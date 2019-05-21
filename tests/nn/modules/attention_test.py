from source.nn import *
import unittest
import torch


class GateTest(unittest.TestCase):
    def test_output_shape(self):
        batch_size = 5
        emb_size = 3
        gate = Gate(emb_size)
        batch = torch.rand((batch_size, emb_size))
        self.assertEqual(batch.shape, gate(batch).shape)


class ScaledDotProductAttentionTest(unittest.TestCase):
    def test_output_shape(self):
        p_length, batch_size, emb_size = 4, 5, 3
        output_size = 6
        p_batch = torch.rand((batch_size, p_length, emb_size))
        attn = ScaledDotProductAttention(emb_size)
        self.assertEqual(torch.Size([batch_size, p_length, emb_size]), attn(p_batch, p_batch, p_batch).shape)


class GatedDotProductTest(unittest.TestCase):
    def test_output_shape(self):
        batch_size, emb_size = 5, 3
        hidden_size = 6
        p_length, q_length = 4, 2
        p_batch = torch.rand((p_length, batch_size, emb_size))
        q_batch = torch.rand((q_length, batch_size, emb_size))
        attn = GatedAttentionBasedRNNWithDotProduct(emb_size, hidden_size)
        self.assertEqual(torch.Size([p_length, batch_size, hidden_size]), attn(p_batch, q_batch).shape)


class SelfMatchingAttentionDotProductTest(unittest.TestCase):
    def test_output_shape(self):
        batch_size, emb_size = 5, 3
        hidden_size = 6
        p_length = 4
        p_batch = torch.rand((p_length, batch_size, emb_size))
        attn = SelfMatchingAttentionWithDotProduct(emb_size, hidden_size)
        self.assertEqual(torch.Size([p_length, batch_size, 2*hidden_size]), attn(p_batch).shape)


class GatedAttentionBasedRNNTest(unittest.TestCase):
    def test_output_shape(self):
        batch_size, emb_size = 5, 3
        hidden_size = 6
        p_length, q_length = 4, 2
        p_batch = torch.rand((p_length, batch_size, emb_size))
        q_batch = torch.rand((q_length, batch_size, emb_size))
        attn = GatedAttentionBasedRNN(emb_size, hidden_size)
        self.assertEqual(torch.Size([p_length, batch_size, hidden_size]), attn(p_batch, q_batch).shape)


class SelfMatchingAttentionTest(unittest.TestCase):
    def test_output_shape(self):
        p_length, batch_size, emb_size = 4, 5, 3
        output_size = 6
        p_batch = torch.rand((p_length, batch_size, emb_size))
        attn = SelfMatchingAttention(emb_size, output_size)
        self.assertEqual(torch.Size([p_length, batch_size, 2*output_size]), attn(p_batch).shape)
