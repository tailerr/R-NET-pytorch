import torch.nn as nn
import torch


class PQEncoding(nn.Module):
    """
    Passage and question encoding.
    output of shape (batch, seq_len, num_directions * hidden_size)
    """
    def __init__(self, input_size, hidden_size, num_layers, bidirectional, batch_first=False):
        super(PQEncoding, self).__init__()
        self.ch_dropout = 0.05
        self.w_dropout = 0.1
        self.encoding = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            bidirectional=bidirectional,
            num_layers=num_layers,
            dropout=self.w_dropout,
            batch_first=batch_first
        )

    def forward(self, ch_emb, w_emb):
        nn.functional.dropout(ch_emb, self.ch_dropout, self.training, True)
        nn.functional.dropout(w_emb, self.w_dropout, self.training, True)
        ch_emb, _ = torch.max(ch_emb, dim=2)
        emb = torch.cat((w_emb, ch_emb), 2)
        return self.encoding(emb)[0].transpose(0, 1)
