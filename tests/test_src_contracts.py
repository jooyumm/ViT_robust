"""공격의 예산/상태 보존과 재현성, 공유 로더의 입력 계약 회귀 검사.

실행: python -m unittest discover -s tests -p 'test_src_contracts.py' -v
사전학습 가중치 다운로드나 대규모 공격 재생성은 하지 않는다.
"""
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from PIL import Image
from timm.models.vision_transformer import VisionTransformer

from src import dataset, models
from src.attack_utils import attack_context
from src.attacks import attention_fool_attack, lavan_attack, patch_fool_attack
from src.attacks.patch_fool import _pcgrad, _attention, _select_patches
from src.attack_utils import attention_terms


def tiny_vit():
    torch.manual_seed(11)
    return VisionTransformer(img_size=16, patch_size=4, embed_dim=24, depth=4,
                             num_heads=3, num_classes=3, qk_norm=True)


def snapshot(model):
    return ([module.training for module in model.modules()],
            [None if p.grad is None else p.grad.clone() for p in model.parameters()],
            [p.detach().clone() for p in model.parameters()])


class AttackContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def assert_state(self, model, state):
        modes, grads, weights = state
        self.assertEqual(modes, [module.training for module in model.modules()])
        for p, grad, weight in zip(model.parameters(), grads, weights):
            torch.testing.assert_close(p, weight, atol=0, rtol=0)
            if grad is None:
                self.assertIsNone(p.grad)
            else:
                torch.testing.assert_close(p.grad, grad, atol=0, rtol=0)
        self.assertTrue(all(not block.attn._forward_hooks for block in getattr(model, 'blocks', [])))

    def test_pcgrad_removes_only_conflicting_component(self):
        a = torch.tensor([[-1., 1.], [1., 2.]])
        c = torch.tensor([[2., 0.], [3., 0.]])
        projected = _pcgrad(a, c, torch.tensor([-1., 1.]), a.shape)
        torch.testing.assert_close(projected, torch.tensor([[0., 1.], [1., 2.]]))
        self.assertEqual(float((projected[0] * c[0]).sum()), 0.)

    def test_rng_isolation_only_when_seed_is_explicit(self):
        model = tiny_vit().train()
        torch.manual_seed(13)
        state = torch.get_rng_state().clone()
        with attack_context(model, seed=7): first = torch.rand(8)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        with attack_context(model, seed=7): second = torch.rand(8)
        torch.testing.assert_close(first, second, atol=0, rtol=0)
        with attack_context(model): first = torch.rand(8)
        with attack_context(model): second = torch.rand(8)
        self.assertFalse(torch.equal(first, second))
        self.assertTrue(model.training)

    def test_seeded_unsupported_device_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'CPU/CUDA'):
            with attack_context(tiny_vit(), seed=7, device='mps'):
                self.fail('지원하지 않는 seed 격리가 허용됐습니다.')

    def test_lavan_can_reach_last_valid_position_without_changing_model(self):
        model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3*8*8, 3)).train()
        for p in model.parameters(): p.grad = torch.ones_like(p)
        state = snapshot(model)
        x = torch.zeros(1, 3, 8, 8)
        with mock.patch('src.attacks.lavan.torch.randint', side_effect=lambda low, high, size: torch.tensor([high-1])):
            with torch.no_grad():
                adv = lavan_attack(model, x, torch.tensor([0]), 'cpu', patch_ratio=1/16, steps=1)
        mask = torch.zeros_like(x, dtype=torch.bool); mask[:, :, -2:, -2:] = True
        torch.testing.assert_close(adv[~mask], x[~mask], atol=0, rtol=0)
        self.assertTrue((adv[mask] != x[mask]).any())
        self.assert_state(model, state)

    def test_lavan_full_image_and_tiny_patch(self):
        model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3*8*8, 3))
        for ratio in (1., 1e-12):
            x = torch.zeros(1, 3, 8, 8)
            adv = lavan_attack(model, x, torch.tensor([0]), 'cpu', patch_ratio=ratio, steps=1)
            self.assertTrue(torch.isfinite(adv).all())
            self.assertTrue((adv != x).any())
            self.assertLessEqual(float(adv.abs().max()), 1.)

    def test_patchfool_modes_preserve_gradients_modes_and_budget(self):
        for mode in ('CE_loss', 'Attention'):
            with self.subTest(mode=mode):
                model = tiny_vit().train(); model.blocks[0].attn.eval()
                for i, p in enumerate(model.parameters()):
                    p.grad = torch.ones_like(p) if i % 2 else None
                state = snapshot(model)
                x = torch.zeros(2, 3, 16, 16); y = torch.tensor([0, 1])
                with torch.inference_mode():
                    adv, attention = patch_fool_attack(model, x, y, 'cpu', patch_size_model=4,
                        attn_layer_idx=1, train_attack_iters=2, attack_mode=mode,
                        mild_l_inf=.1, clip_valid_range=True)
                peak = attention[1].mean((1, 2))[:, 1:].argmax(-1)
                allowed = torch.zeros_like(x, dtype=torch.bool)
                for i, index in enumerate(peak.tolist()):
                    r, c = divmod(index, 4); allowed[i, :, r*4:(r+1)*4, c*4:(c+1)*4] = True
                torch.testing.assert_close(adv[~allowed], x[~allowed], atol=0, rtol=0)
                self.assertLessEqual(float((adv-x).abs().max()), .100001)
                self.assertTrue(torch.isfinite(adv).all())
                self.assert_state(model, state)

    def test_patchfool_qk_norm_hook_matches_actual_attention(self):
        model = tiny_vit().eval()
        model.blocks[0].attn.fused_attn = False
        actual = []
        handle = model.blocks[0].attn.attn_drop.register_forward_pre_hook(lambda m, inputs: actual.append(inputs[0].detach()))
        try:
            with torch.no_grad(), attention_terms(model, _attention) as collected:
                model(torch.randn(1, 3, 16, 16))
            torch.testing.assert_close(collected[0], actual[0], atol=1e-6, rtol=1e-5)
        finally:
            handle.remove()

    def test_patchfool_incoming_and_cls_are_explicit_choices(self):
        a = torch.zeros(1, 1, 5, 5)
        a[:, :, :, 3] = .7
        a[:, :, 0, 1] = .9
        x = torch.zeros(1, 3, 8, 8)
        for source, expected in [('incoming', 2), ('cls', 0)]:
            selected = _select_patches(a, None, x, torch.tensor([0]), 4, 1, 'Attn', source)
            self.assertEqual(selected.item(), expected)

    def test_patchfool_saliency_and_contiguous_use_same_attack_loop(self):
        for method in ('Saliency', 'Contiguous', 'Rand'):
            with self.subTest(method=method):
                model = tiny_vit(); x = torch.zeros(1,3,16,16)
                adv, _ = patch_fool_attack(model, x, torch.tensor([0]), 'cpu', patch_size_model=4,
                    attn_layer_idx=1, train_attack_iters=1, patch_select=method, mild_l_inf=.1)
                self.assertTrue(torch.isfinite(adv).all())
                self.assertLessEqual(float(adv.abs().max()), .100001)

    def test_patchfool_failure_restores_state_and_hooks(self):
        model = tiny_vit().train(); state = snapshot(model)
        with mock.patch.object(model, 'forward', side_effect=RuntimeError('intentional')):
            with self.assertRaisesRegex(RuntimeError, 'intentional'):
                patch_fool_attack(model, torch.zeros(1,3,16,16), torch.tensor([0]), 'cpu',
                                  patch_size_model=4, attn_layer_idx=1, train_attack_iters=1)
        self.assert_state(model, state)

    def test_invalid_attack_parameters_fail_explicitly(self):
        model = tiny_vit(); x = torch.zeros(1,3,16,16); y = torch.tensor([0])
        with self.assertRaises(ValueError): lavan_attack(model, x, y, 'cpu', targeted=True)
        with self.assertRaises(ValueError): lavan_attack(model, x, y, 'cpu', fixed_loc='typo')
        for kwargs in ({'attack_mode':'typo'}, {'num_patch':0}, {'train_attack_iters':0}, {'attn_layer_idx':99}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                patch_fool_attack(model, x, y, 'cpu', patch_size_model=4, **kwargs)

    def test_attention_fool_preserves_mask_state_and_reproducibility(self):
        model = tiny_vit().train(); x = torch.zeros(1,3,16,16); y = torch.tensor([0])
        for p in model.parameters(): p.grad = torch.ones_like(p)
        state = snapshot(model)
        kwargs = dict(patch_size=4, fixed_loc='center', steps=2, eps=.1, seed=19, return_info=True, layers=[1])
        first, info = attention_fool_attack(model, x, y, 'cpu', **kwargs)
        second, _ = attention_fool_attack(model, x, y, 'cpu', **kwargs)
        torch.testing.assert_close(first, second, atol=0, rtol=0)
        allowed = info['mask'].expand_as(x)
        torch.testing.assert_close(first[~allowed], x[~allowed], atol=0, rtol=0)
        self.assertLessEqual(float(((first-x)*.5).abs().max()), .100001)
        self.assert_state(model, state)


class LoaderContracts(unittest.TestCase):
    def test_dataset_rejects_invalid_sizes_before_loading(self):
        for kwargs in ({'num_samples':-1},{'num_samples':0},{'batch_size':0},{'num_workers':-1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError): dataset.get_dataloader(**kwargs)
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(FileNotFoundError): dataset.get_dataloader(dataset_root=Path(temporary)/'absent')

    def test_dataset_same_seed_different_batch_sizes_same_sample_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            for cls in ('a','b'):
                folder = Path(temporary)/cls; folder.mkdir()
                for i in range(4): Image.new('RGB',(256,256),(i*20,40,60)).save(folder/f'{i}.png')
            first, _ = dataset.get_dataloader(batch_size=2,num_samples=6,seed=3,dataset_root=temporary,num_workers=0,pin_memory=False)
            second, _ = dataset.get_dataloader(batch_size=3,num_samples=6,seed=3,dataset_root=temporary,num_workers=0,pin_memory=False)
            self.assertEqual(first.dataset.indices,second.dataset.indices)
            self.assertEqual(len(set(first.dataset.indices)),6)
            self.assertEqual(tuple(next(iter(first))[0].shape),(2,3,224,224))

    def test_model_failure_strict_mode_preserves_original_error(self):
        with mock.patch.object(models.timm,'create_model',side_effect=RuntimeError('missing checkpoint')):
            self.assertIsNone(models.load_vit_model(16,'cpu'))
            with self.assertRaisesRegex(RuntimeError,'missing checkpoint'): models.load_vit_model(16,'cpu',strict=True)
        with self.assertRaises(ValueError): models.load_vit_model(7,'cpu',strict=True)




class SharedPGDContracts(unittest.TestCase):
    def test_pgd_preserves_state_and_pixel_budget_inside_no_grad(self):
        from src.attacks import pgd_attack
        model=tiny_vit().train()
        for p in model.parameters():p.grad=torch.ones_like(p)
        modes=[m.training for m in model.modules()]
        grads=[p.grad.clone() for p in model.parameters()]
        x=torch.rand(2,3,16,16)*2-1;y=torch.tensor([0,1])
        with torch.no_grad():adv=pgd_attack(model,x,y,'cpu',steps=2)
        self.assertEqual(modes,[m.training for m in model.modules()])
        for p,g in zip(model.parameters(),grads):torch.testing.assert_close(p.grad,g,atol=0,rtol=0)
        self.assertLessEqual(float(((adv-x)*.5).abs().max()),8/255+1e-6)
        self.assertTrue(bool((adv>=-1).all() and (adv<=1).all()))

    def test_projects_do_not_define_their_own_attack_implementations(self):
        import ast
        root=Path(__file__).resolve().parents[1]
        for project in ['00_patch_size_tradeoff','01_attention_characterization','02_detection_defense']:
            for p in (root/project).rglob('*.py'):
                if any(x in p.parts for x in ('reruns','runs','results','__pycache__')):continue
                names={n.name for n in ast.walk(ast.parse(p.read_text())) if isinstance(n,ast.FunctionDef)}
                self.assertFalse(names & {'pgd_attack','patch_fool_attack','lavan_attack','attention_fool_attack'},str(p))


if __name__ == '__main__': unittest.main()
