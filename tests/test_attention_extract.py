"""공식 attention 추출기의 값, 단일 forward, 상태·hook 복원 검사."""
import unittest
from unittest.mock import patch

import torch
from timm.models.vision_transformer import VisionTransformer

from src.attention import extract_attention


class AttentionExtraction(unittest.TestCase):
    def setUp(self):
        self.model = VisionTransformer(img_size=16, patch_size=4, embed_dim=24,
                                       depth=2, num_heads=3, num_classes=3).eval()
        self.images = torch.randn(2, 3, 16, 16)

    def state(self):
        return ([b.attn.fused_attn for b in self.model.blocks],
                [(len(m._forward_hooks), len(m._forward_pre_hooks))
                 for m in self.model.modules()])

    def test_probabilities_logits_and_cleanup(self):
        state = self.state()
        with torch.no_grad():
            expected = self.model(self.images)
            for _ in range(2):
                with patch.object(self.model, 'forward', wraps=self.model.forward) as forward:
                    logits, attention = extract_attention(self.model, self.images)
                    self.assertEqual(forward.call_count, 1)
                torch.testing.assert_close(logits, expected, atol=1e-6, rtol=1e-5)
                self.assertEqual(len(attention), 2)
                for a in attention:
                    self.assertEqual(a.shape, (2, 3, 17, 17))
                    torch.testing.assert_close(a.sum(-1), torch.ones_like(a.sum(-1)))
                self.assertEqual(self.state(), state)

    def test_cleanup_after_failed_forward(self):
        state = self.state()
        with self.assertRaises(AssertionError):
            extract_attention(self.model, torch.randn(1, 3, 8, 8))
        self.assertEqual(self.state(), state)

    def test_training_model_rejected_without_state_change(self):
        self.model.train()
        state = self.state()
        with self.assertRaises(ValueError):
            extract_attention(self.model, self.images)
        self.assertTrue(self.model.training)
        self.assertEqual(self.state(), state)
