"""실험 선택, CLI 덮어쓰기, 마지막 batch·기존 로그 호환 회귀 검사."""
from pathlib import Path
import runpy
import json
import tempfile
import unittest
from unittest.mock import patch
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]/'00_patch_size_tradeoff'
EXPERIMENT = runpy.run_path(str(ROOT/'experiments/main.py'))
RUNNER = EXPERIMENT['run_case'].__globals__


class Experiments(unittest.TestCase):
    def cases(self, argv):
        return list(EXPERIMENT['iter_cases'](EXPERIMENT['get_args'](argv)))

    def test_baseline_runs_six_combinations_with_memory_defaults(self):
        cases=self.cases(['--experiment','1'])
        self.assertEqual([(a.patch_sizes[0],a.attacks[0],a.batch_size) for a in cases],
                         [(8,'pgd',16),(8,'lavan',16),(8,'patch_fool',8),
                          (16,'pgd',64),(16,'lavan',64),(16,'patch_fool',64)])
        self.assertTrue(all(a.num_samples==1000 for a in cases))

    def test_cli_override_and_four_ablations(self):
        args=self.cases(['--experiment','1','--attacks','patch_fool','--patch_sizes','8','--batch_size','2','--seed','123'])
        self.assertEqual(len(args),1)
        self.assertEqual((args[0].batch_size,args[0].seed),(2,123))
        for number in range(2,6):
            args=self.cases(['--experiment',str(number)])[0]
            if number==2:self.assertEqual(args.lavan_loc,'center')
            if number in (3,4):self.assertEqual(args.pf_num_patch,4)
            if number==3:self.assertEqual(args.pf_patch_select,'Attn')
            if number==4:self.assertEqual(args.pf_patch_select,'Contiguous')
            if number==5:self.assertEqual(args.pf_patch_select,'Rand')

    def test_partial_final_batch_and_visualizer_result_compatibility(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                return torch.stack((x[:,0,0,0],-x[:,0,0,0]),1)
        with tempfile.TemporaryDirectory() as tmp:
            args=self.cases(['--experiment','2','--patch_sizes','16','--batch_size','2','--num_samples','5','--result-dir',tmp])[0]
            loader=DataLoader(TensorDataset(torch.ones(5,3,2,2),torch.zeros(5,dtype=torch.long)),batch_size=2)
            def get_loader(**kwargs):
                self.assertEqual(kwargs['num_samples'],5)
                return loader,None
            with patch.dict(RUNNER,get_device=lambda:'cpu',get_dataloader=get_loader,
                            load_vit_model=lambda *a,**kw:Model(),lavan_attack=lambda m,x,y,d,**kw:x):
                result=EXPERIMENT['run_case'](args)
            self.assertEqual(result['valid'],5)
            self.assertEqual(result['CA'],100.)
            record=json.loads(next(Path(tmp).glob('*.json')).read_text())
            self.assertEqual(record['metrics']['valid'],5)
            reader=runpy.run_path(str(ROOT/'fig_common/results_io.py'))['collect_results']
            self.assertEqual(reader(tmp,tag='fixedloc')['lavan'][16]['CA'],[100.])
            self.assertFalse(list(Path(tmp).glob('*.txt')))
