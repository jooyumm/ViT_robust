"""적응형 후보 선택, 공통 공격 엔진 연결, 비차별 추론 경계 검사."""
import inspect
from pathlib import Path
import sys
import unittest
import torch
from timm.models.vision_transformer import VisionTransformer

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'02_detection_defense'))
from region_switch.experiments.compare_defenses import BestCandidate,score_inputs
from region_switch.core.inference import forward_defended
from src.attacks import patch_fool_attack,patch_fool_config


class DefenseComparison(unittest.TestCase):
    def test_best_candidate_prioritizes_actual_misclassification(self):
        best=BestCandidate(torch.zeros(1,1),torch.tensor([0]))
        best(torch.tensor([[1.]]),torch.tensor([[-.01,0.,-9.]]))
        best(torch.tensor([[2.]]),torch.tensor([[.01,0.,0.]]))
        self.assertEqual(best.images.item(),1.)
        self.assertTrue(best.success.item())
        best(torch.tensor([[3.]]),torch.tensor([[-3.,2.,0.]]))
        self.assertEqual(best.images.item(),3.)

    def test_shared_patchfool_uses_defended_ce_and_keeps_patch_budget(self):
        torch.set_num_threads(1);torch.manual_seed(7)
        model=VisionTransformer(img_size=16,patch_size=4,embed_dim=24,depth=4,num_heads=3,num_classes=3).eval().requires_grad_(False)
        x=torch.rand(2,3,16,16)*2-1;y=torch.tensor([0,1]);seen=[];calls=[]
        def objective(images):
            calls.append(torch.is_grad_enabled())
            return model(images).flip(1)
        def candidate(images,logits):
            with torch.no_grad():torch.testing.assert_close(logits,objective(images))
            seen.append(images.clone())
        attacked,_=patch_fool_attack(model,x,y,'cpu',patch_size_model=4,attn_layer_idx=1,train_attack_iters=2,
                                    classification_forward=objective,initial_delta=torch.zeros_like(x),candidate_callback=candidate,
                                    selected_patches=torch.tensor([[0],[15]]))
        self.assertEqual(len(seen),3)
        self.assertTrue(any(calls));self.assertFalse(any(m._forward_hooks or m._forward_pre_hooks for m in model.modules()))
        changed=(attacked-x).abs().sum(1,keepdim=True)>0
        patches=changed.unfold(2,4,4).unfold(3,4,4).any(-1).any(-1).flatten(1).sum(1)
        torch.testing.assert_close(patches,torch.ones(2,dtype=patches.dtype))
        cells=changed.unfold(2,4,4).unfold(3,4,4).any(-1).any(-1).flatten(1)
        torch.testing.assert_close(cells.long().argmax(1),torch.tensor([0,15]))
        self.assertTrue(bool((attacked>=-1).all() and (attacked<=1).all()))
        self.assertFalse(any(k in patch_fool_config() for k in ['classification_forward','initial_delta','candidate_callback','selected_patches']))

    def test_policy_forward_has_no_attack_label_or_gt_input(self):
        self.assertEqual(list(inspect.signature(forward_defended).parameters),['model','kernel8','images','detector'])
        self.assertNotIn('labels',inspect.signature(score_inputs).parameters)
