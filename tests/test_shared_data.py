"""공통 이미지/GT 경계와 어텐션 추출의 공격 재생성 금지 계약."""
import ast,json,sys,tempfile,unittest
from pathlib import Path
import numpy as np
from src.data.paired import load_images
from src.provenance import sha256
from src.patches import coverage_from_images
from src.ground_truth.coverage import gt_coverage

ROOT=Path(__file__).resolve().parents[1]

class SharedData(unittest.TestCase):
    def test_attention_extraction_does_not_generate_attacks(self):
        tree=ast.parse((ROOT/'src/data/extract_attention.py').read_text())
        imports=[n.module or '' for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
        self.assertFalse(any('attacks' in m or 'generate' in m for m in imports))
        names={n.func.id for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)}
        self.assertFalse(names & {'patch_fool_attack','lavan_attack','attention_fool_attack'})

    def test_image_loader_needs_no_attention_and_checks_hashes(self):
        with tempfile.TemporaryDirectory() as t:
            folder=Path(t);clean=np.zeros((2,3,4,4),np.float32);attacked=clean.copy();attacked[:,:,0,0]=1
            path=folder/'chunk.npz'
            np.savez(path,sample_index=[7,12],label=[1,0],clean__images=clean,
                     patchfool__images=attacked,patchfool__gt_coverage=coverage_from_images(clean,attacked,2))
            manifest=dict(complete=True,sample_indices=[7,12],chunks=[dict(path='chunk.npz',n=2,sha256=sha256(path))])
            (folder/'manifest.json').write_text(json.dumps(manifest))
            _,data=load_images(folder)
            np.testing.assert_array_equal(data['patchfool__images'],attacked)
            path.write_bytes(path.read_bytes()+b'corrupt')
            with self.assertRaises(ValueError):load_images(folder)

    def test_single_image_gt_reuses_shared_pixel_coverage(self):
        import torch
        clean=torch.zeros(1,3,8,8);attacked=clean.clone();attacked[:,:,:2,:2]=1
        np.testing.assert_array_equal(gt_coverage(clean[0],attacked[0],patch_size=4,ppl=2),
                                      coverage_from_images(clean,attacked,4)[0])

    def test_default_attention_cache_rejects_another_attack_protocol(self):
        from src.cohort import COHORT,validate_default_manifest
        from src.data.paths import PATCHFOOL
        from src.data.protocol import vanilla_p16_config
        cache=dict(sample_indices=[s['index'] for s in COHORT['samples']],sample_paths=[s['path'] for s in COHORT['samples']],
                   model_sha256=COHORT['model_sha256'],source_manifest_sha256=sha256(PATCHFOOL/'manifest.json'),
                   patchfool_protocol=vanilla_p16_config(),attack_generation=False,attacks=['patchfool'])
        validate_default_manifest(cache)
        cache['patchfool_protocol']['pcgrad_mode']='projected'
        with self.assertRaises(ValueError):validate_default_manifest(cache)
        cache['patchfool_protocol']=vanilla_p16_config();cache['source_manifest_sha256']='other attack data'
        with self.assertRaises(ValueError):validate_default_manifest(cache)
