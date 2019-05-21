from source.nn.modules.output_layer import *
import unittest
import torch


class PointerNetworkTest(unittest.TestCase):
    def test_initial_state_shape(self):
        length, batch_size, emb_size = 4, 5, 3
        question = torch.rand((length, batch_size, emb_size))
        p_n = PointerNetwork(emb_size)
        self.assertEqual(torch.Size([batch_size, emb_size]), p_n.get_initial_state(question).shape)

    def test_output_shape(self):
        length, batch_size, emb_size = 4, 5, 3
        question = torch.rand((length, batch_size, emb_size))
        passage_length = 6
        batch = torch.rand((passage_length, batch_size, emb_size))
        p_n = PointerNetwork(emb_size)
        p1, p2 = p_n(batch, question)
        self.assertEqual(torch.Size([batch_size, passage_length, 1]), p1.shape, "Error in shape of p1")
        self.assertEqual(torch.Size([batch_size, passage_length, 1]), p1.shape, "Error in shape of p1")
