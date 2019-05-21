import torch
import torch.nn as nn
import numpy as np


class Gate(nn.Module):
    def __init__(self, input_size, dropout=0.2):
        """ To determine the importance of passage parts and
            attend to the ones relevant to the question, this Gate was added
            to the input of RNNCell in both Gated Attention-based Recurrent
            Network and Self-Matching Attention.

            Args:
                input_size(int): size of input vectors
                dropout (float, optional): dropout probability

            Input:
                - **input** of shape `(batch, input_size)`: a float tensor containing concatenated
                  passage representation and attention vector both calculated for each word in the passage

            Output:
                - **output** of shape `(batch, input_size)`: a float tensor containing gated input
        """
        super(Gate, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.W = nn.Linear(input_size, input_size, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        result = self.W(input)
        self.dropout(result)
        result = self.sigmoid(result)
        return result * input


class ScaledDotProductAttention(nn.Module):
    """
    Input:
        value: float tensor of shape (batch, seq_len, dim_m)
        key: float tensor of shape (batch, seq_len, dim_q)
        query: float tensor of shape (batch, q_len, dim_q)
        mask: float tensor of shape (batch, q_len, seq_len)
    Output:
        attention: a float tensor of shape (batch, q_len, dim_m)
    """
    def __init__(self, input_size):
        super(ScaledDotProductAttention, self).__init__()
        self.input_size = input_size
        self.scale_factor = np.power(input_size, -0.5)

    def forward(self, value, key, query, mask=None):
        outputs = torch.bmm(value, key.permute([0, 2, 1]))*self.scale_factor
        if mask is not None:
            outputs.masked_fill_(mask, -float('inf'))
        attention = nn.functional.softmax(outputs, 2)
        attention = torch.bmm(attention, key)
        return attention


class GatedAttentionBasedRNNWithDotProduct(nn.Module):
    """ Instead of additive attn use Scaled dot-product attn.

            Args:
                input_size(int): size of input tensor (emb_size)
                dropout (float, optional): dropout probability

            Input:
                - **question** of shape `(question_length, batch, input_size)`: a float tensor containing question
                  representation
                - **passage** of shape `(passage_length, batch, input_size)`: a float tensor containing passage
                  representation

            Output:
                - **V** of shape `(passage_legth, batch, output_size)`: a float tensor in which we determine
                the importance of information in the passage regarding a question
        """
    def __init__(self, input_size, output_size, dropout=0.2):
        super(GatedAttentionBasedRNNWithDotProduct, self).__init__()
        self.gru = nn.GRU(input_size, output_size)
        self.attention = ScaledDotProductAttention(input_size)
        self.input_size = input_size
        self.gate = Gate(self.input_size * 2, dropout)
        
    def forward(self, passage, question):
        c = self.attention(passage, passage, question)
        g = torch.cat((passage, c), 2)
        g = self.gate(g)
        _, c = torch.split(g, (self.input_size, self.input_size), 2)
        return self.gru(c)[0]


class SelfMatchingAttentionWithDotProduct(nn.Module):
    ''' Instead of additive attn use Scaled dot-product attn.
        Args:
                    input_size(int): size of input tensor
                    output_size(int): output size
                    dropout (float, optional): dropout probability

                Input:
                    - **V** of shape `(passage_length, batch, input_size)`:  a float tensor in which we determine
                    the importance of information in the passage regarding a question

                Output:
                    - **H** of shape `(passage_legth, batch, 2*output_size)`: a float tensor in which we determine
                    the importance of information in the passage regarding a question

    '''
    def __init__(self, input_size, output_size, dropout=0.2):
        super(SelfMatchingAttentionWithDotProduct, self).__init__()
        self.gru = nn.GRU(input_size*2, output_size, bidirectional=True)
        self.attention = ScaledDotProductAttention(input_size)
        self.gate = Gate(input_size * 2, dropout)

    def forward(self, H):
        c = self.attention(H, H, H)
        g = torch.cat((H, c), 2)
        g = self.gate(g)
        return self.gru(g)[0]


class GatedAttentionBasedRNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=75, dropout=0.2):
        """ Gated Attention Based RNN.

            Args:
                input_size(int): size of input tensor (emb_size)
                hidden_size(int): hidden size and output size
                dropout (float, optional): dropout probability

            Input:
                - **question** of shape `(batch, question_length, input_size)`: a float tensor containing question
                  representation
                - **passage** of shape `(batch, passage_length, input_size)`: a float tensor containing passage
                  representation

            Output:
                - **V** of shape `(passage_legth, batch, output_size)`: a float tensor in which we determine
                the importance of information in the passage regarding a question
        """
        super(GatedAttentionBasedRNN, self).__init__()
        self.input_size = input_size
        self.Wuq = nn.Linear(self.input_size, hidden_size, bias=False)
        self.Wup = nn.Linear(self.input_size, hidden_size, bias=False)
        self.v = nn.Linear(hidden_size, 1, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.gate = Gate(self.input_size * 2)

        self.rnn = nn.GRU(input_size, output_size)

    def forward(self, passage, question):
        p, batch_size, _ = passage.size()
        q, _, _ = question.size()
        Wp = self.Wup(passage)
        self.dropout(Wp)
        Wq = self.Wuq(question)
        self.dropout(Wq)
        Wq = Wq.repeat(p, 1, 1, 1).permute([1, 0, 2, 3])
        Wp = Wp.repeat(q, 1, 1, 1)
        s = Wq+Wp
        torch.tanh_(s)
        s = self.v(s)
        a = nn.functional.softmax(s, 0)
        u = question.repeat(p, 1, 1, 1).permute([1, 0, 2, 3])
        c = a*u
        c = c.sum(0)
        g = torch.cat((passage, c), 2)
        g = self.gate(g)
        _, c = torch.split(g, (self.input_size, self.input_size), 2)
        v, _ = self.rnn(c)
        return v


class SelfMatchingAttention(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=75, dropout=0.2):
        """ Self Matching Attention.

                Args:
                    input_size(int): size of input tensor
                    output_size(int): output size
                    dropout (float, optional): dropout probability

                Input:
                    - **V** of shape `(passage_length, batch, input_size)`:  a float tensor in which we determine
                    the importance of information in the passage regarding a question

                Output:
                    - **H** of shape `(passage_legth, batch, 2*output_size)`: a float tensor in which we determine
                    the importance of information in the passage regarding a question

        """
        super(SelfMatchingAttention, self).__init__()
        self.input_size = input_size
        self.Wvp1 = nn.Linear(input_size, hidden_size, bias=False)
        self.Wvp2 = nn.Linear(input_size, hidden_size, bias=False)
        self.v = nn.Linear(hidden_size, 1, bias=False)
        self.gate = Gate(2 * input_size)
        self.dropout = nn.Dropout(dropout)

        self.rnn = nn.GRU(2*input_size, output_size, bidirectional=True)

    def forward(self, v):
        p, batch_size, _ = v.size()
        Wv1 = self.Wvp1(v)
        self.dropout(Wv1)
        Wv2 = self.Wvp2(v)
        self.dropout(Wv2)
        Wv1 = Wv1.repeat(p, 1, 1, 1)
        Wv2 = Wv2.repeat(p, 1, 1, 1).permute([1, 0, 2, 3])
        s = Wv2 + Wv1
        torch.tanh_(s)
        s = self.v(s)
        a = nn.functional.softmax(s, 0)
        c = a*v.repeat(p, 1, 1, 1)
        c = c.sum(0)
        g = torch.cat((v, c), 2)
        g = self.gate(g)
        h, _ = self.rnn(g)
        return h
