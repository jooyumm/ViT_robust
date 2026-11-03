"""P16 공격/블라인드 P8 영역 방어 경계와 공식 PCGrad 선택 검사."""
import ast
import importlib
import inspect
import json
from pathlib import Path
import sys
import unittest
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'02_detection_defense'))
from src.data.protocol import vanilla_p16_config, validate_attack_manifest
from src.attacks.patch_fool import _pcgrad, patch_fool_config
from src.cohort import COHORT,SAMPLE_FILE
from src.provenance import sha256


class SeparatedProtocol(unittest.TestCase):
    def manifest(self):
        return dict(complete=True,schema_version=1,sample_indices=[s['index'] for s in COHORT['samples']],
                    sample_paths=[s['path'] for s in COHORT['samples']],cohort_sha256=sha256(SAMPLE_FILE),
                    model_sha256=COHORT['model_sha256'],attack_target='undefended_vit_b16',
                    defense_aware=False,patchfool_protocol=vanilla_p16_config())

    def test_p16_one_patch_official_settings_are_pinned(self):
        c=vanilla_p16_config()
        self.assertEqual(c,patch_fool_config(**c))
        self.assertEqual((c['patch_size_model'],c['num_patch'],c['patch_select']),(16,1,'Attn'))
        self.assertEqual((c['attack_mode'],c['train_attack_iters'],c['pcgrad_mode']),('Attention',250,'official'))
        self.assertTrue(c['clip_valid_range'])
        self.assertEqual((c['mild_l_inf'],c['sparse_pixel_num']),(0.,0))
        validate_attack_manifest(self.manifest())

    def test_reject_defense_aware_wrong_model_and_modified_budget(self):
        for field,value in [('defense_aware',True),('attack_target','vit_b8'),('complete',False)]:
            m=self.manifest();m[field]=value
            with self.assertRaises(ValueError):validate_attack_manifest(m)
        m=self.manifest();m['patchfool_protocol']['num_patch']=4
        with self.assertRaises(ValueError):validate_attack_manifest(m)

    def test_attack_process_has_no_defense_imports(self):
        tree=ast.parse((ROOT/'src/data/generate_patchfool.py').read_text())
        imports=[n.module or '' for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
        self.assertFalse(any('region_switch' in n or 'defense_common' in n for n in imports))
        fn=importlib.import_module('region_switch.core.inference').predict_defended
        self.assertEqual(list(inspect.signature(fn).parameters),['model','kernel8','images','detector'])
        # protocol validation itself does not import attack/defense implementations.
        tree=ast.parse((ROOT/'src/data/protocol.py').read_text())
        self.assertFalse(any(isinstance(n,ast.ImportFrom) and 'attacks' in (n.module or '') for n in ast.walk(tree)))

    def test_official_pcgrad_uses_norm_and_old_default_is_unchanged(self):
        a=torch.tensor([[-2.,1.],[1.,2.]])
        c=torch.tensor([[2.,0.],[1.,1.]])
        sim=torch.tensor([-1.,1.])
        official=a.clone();official[0]=a[0]-(a[0]*c[0]).sum()/c[0].norm()*c[0]
        torch.testing.assert_close(_pcgrad(a,c,sim,a.shape,'official'),official)
        projected=a.clone();projected[0]=a[0]-(a[0]*c[0]).sum()/c[0].square().sum()*c[0]
        torch.testing.assert_close(_pcgrad(a,c,sim,a.shape),projected)
