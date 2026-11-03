"""표본 분모, 지표 경계, 블라인드 평가·해시 검증, 공유 좌표 회귀 검사."""
import importlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'02_detection_defense'))
from defense_common.evaluation import auroc, summarize, evaluate_pairs
from src.patches import coverage_from_images, patch_region_index, patch_to_region
from src.provenance import sha256
from region_switch.experiments.model_support import reusable_results, PROTOCOL


class EvaluationContracts(unittest.TestCase):
    def fixture(self):
        labels = np.zeros(6, dtype=int)
        base = dict(clean=np.array([0,0,0,0,1,1]), patchfool=np.array([1,1,0,0,1,0]))
        defended = dict(clean=np.array([1,0,0,0,1,1]), patchfool=np.array([0,1,1,0,1,0]))
        decision = dict(clean=dict(flag=np.array([1,0,0,0,0,0]),score=np.arange(6)),
                        patchfool=dict(flag=np.array([1,0,1,1,1,1]),score=np.arange(6),region=np.array([0,0,1,0,0,0])))
        return labels, base, defended, decision, np.zeros(6,dtype=int)

    def test_populations_and_damage_denominators(self):
        r = summarize(*self.fixture())
        self.assertAlmostEqual(r['attack_detection_rate'], 5/6)
        self.assertEqual(r['successful_attack_detection_rate'], .5)
        self.assertEqual(r['failed_attack_detection_rate'], 1.)
        self.assertEqual(r['recovery_rate'], .5)
        self.assertEqual(r['clean_damage_rate'], .25)
        self.assertAlmostEqual(r['flagged_region_accuracy'], 4/5)
        self.assertAlmostEqual(r['end_to_end_region_accuracy'], 4/6)
        self.assertEqual(r['populations']['clean_already_wrong'], 2)
        self.assertEqual(r['clean_accuracy'], 4/6)

    def test_empty_conditional_population_is_null_not_zero(self):
        labels, base, defended, decisions, gt = self.fixture()
        base['patchfool'] = base['clean'].copy()
        r = summarize(labels,base,defended,decisions,gt)
        self.assertIsNone(r['recovery_rate'])
        self.assertIsNone(r['successful_attack_detection_rate'])
        self.assertIsNone(r['successful_attack_detector_auroc'])
        self.assertEqual(auroc([1,1],[1,1]), .5)

    def test_auroc_matches_pairwise_reference_including_ties(self):
        p, n = np.array([1.,2.,2.,5.]), np.array([1.,1.,2.,4.])
        expected = ((p[:,None] > n) + .5*(p[:,None] == n)).mean()
        self.assertEqual(auroc(p,n), expected)

    def test_blind_mixed_evaluation_and_corrupt_chunk(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp); path=run/'chunk.npz'
            clean=np.zeros((2,3,2,2),dtype=np.float32);clean[1]=1
            attacked=1-clean
            clean_logits=np.eye(2,dtype=np.float32)*5
            coverage=np.zeros((2,196));coverage[:,0]=1
            np.savez(path,sample_index=np.array([10,20]), label=np.array([0,1]),
                     clean__images=clean,patchfool__images=attacked,
                     clean__logits=clean_logits, patchfool__logits=clean_logits[::-1],
                     clean__prediction=np.array([0,1]),patchfool__prediction=np.array([1,0]),
                     patchfool__gt_coverage=coverage)
            manifest=dict(sample_indices=[10,20],chunks=[dict(path='chunk.npz',n=2,sha256=sha256(path))])
            (run/'manifest.json').write_text(json.dumps(manifest))
            calls=[]
            def predict(images):
                calls.append(len(images))
                logits=torch.eye(2)[images[:,0,0,0].long()]*5
                decision={k:np.zeros(len(images)) for k in ['score','flag','patch','region']}
                return logits,logits.clone(),decision
            r,_=evaluate_pairs(predict,manifest,run=run,batch_size=3)
            self.assertEqual(sum(calls),4)
            self.assertEqual((r['clean_accuracy'],r['attack_accuracy'],r['recovery_rate']),(1.,0.,0.))
            def perturbed(images):
                logits, corrected, decision = predict(images)
                return logits + .007, corrected + .007, decision
            tolerant,_=evaluate_pairs(perturbed,manifest,run=run)
            self.assertEqual(tolerant['baseline_prediction_mismatches'],0)
            self.assertAlmostEqual(tolerant['baseline_max_logit_error'],.007,places=5)
            def changed_prediction(images):
                logits, _, decision = predict(images)
                return logits.flip(1), logits.flip(1), decision
            with self.assertRaisesRegex(ValueError,'예측'):
                evaluate_pairs(changed_prediction,manifest,run=run)
            path.write_bytes(path.read_bytes()+b'tampered')
            with self.assertRaisesRegex(ValueError,'해시'):
                evaluate_pairs(predict,manifest,run=run)

    def test_reuse_rejects_changed_protocol_or_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'report.log'
            report=dict(results={},attack_iters=250,batch_size=1,seed=21,
                        attack_protocol=PROTOCOL,source_manifest_sha256='data',source_sha256={'a':'hash'})
            p.write_text('progress\n'+json.dumps(report,indent=2))
            kwargs=dict(attack_iters=250,batch_size=1,seed=21,manifest_hash='data',sources={'a':'hash'},n=500)
            self.assertEqual(reusable_results(p,**kwargs),{})
            with self.assertRaisesRegex(ValueError,'source_sha256'):
                reusable_results(p,**{**kwargs,'sources':{'a':'changed'}})

    def test_both_defenses_import_without_run_or_data_io_collision(self):
        first=importlib.import_module('region_switch.run')
        second=importlib.import_module('region_switch_v2.run')
        self.assertIsNot(first.predict_defended,second.predict_defended)
        self.assertIs(first.evaluate_pairs,second.evaluate_pairs)


class PatchGeometryContracts(unittest.TestCase):
    def test_region_coverage_and_tensor_numpy_equivalence(self):
        clean=np.zeros((1,3,224,224),np.float32);adv=clean.copy()
        adv[:,:,0:8,0:8]=1
        np_cov=coverage_from_images(clean,adv)
        torch_cov=coverage_from_images(torch.from_numpy(clean),torch.from_numpy(adv))
        np.testing.assert_array_equal(np_cov,torch_cov)
        self.assertEqual(np_cov[0,0],.25)
        self.assertEqual(np_cov.sum(),.25)
        np.testing.assert_array_equal(patch_region_index(),patch_to_region(np.arange(196)))
        np.testing.assert_array_equal(np.bincount(patch_region_index()),[49]*4)


class ExperimentProtocolContracts(unittest.TestCase):
    def test_expanded_patchfool_config_tracks_actual_defaults(self):
        from src.attacks.patch_fool import patch_fool_config
        config=patch_fool_config(patch_size_model=8,num_patch=4,patch_select='Contiguous')
        self.assertEqual(config['train_attack_iters'],250)
        self.assertEqual(config['attack_mode'],'Attention')
        self.assertEqual(config['attention_source'],'incoming')
        self.assertEqual(config['initialization'],'replacement')
        self.assertEqual(config['num_patch'],4)
        with self.assertRaises(ValueError):patch_fool_config(unknown_option=3)

