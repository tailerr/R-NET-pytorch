from nn import *


class RNet(nn.Module):
    def __init__(self, word_mat, char_mat, emb_word_size, emb_char_size, v_size, h_size,
                 model_dim, num_layers, attn_type, pretrained_char, dropout=0.2):
        """
        The gated self-matching networks for reading comprehension style question answering, which aims to answer
        questions from a given passage.
        :param vocab_size: vocabulary size(int)
        :param emb_size: embedding size(int, optional)
        :param embedding_weights: float tensor of shape `(vocab_size, dim_m)`, containing
                embedding weights. Embedding size value would inherited from shape of `embedding_weights` tensor.
        :param hidden_size: hidden size(int)
        :param dropout:
        """
        super(RNet, self).__init__()
        self.char_emb = nn.Embedding.from_pretrained(torch.Tensor(char_mat), freeze=pretrained_char)
        self.word_emb = nn.Embedding.from_pretrained(torch.Tensor(word_mat))
        self.p_encoder = PQEncoding(emb_word_size+emb_char_size, model_dim, num_layers, True)
        self.q_encoder = PQEncoding(emb_word_size+emb_char_size, model_dim, num_layers, True)
        if attn_type == "additive":
            self.gated_attn = GatedAttentionBasedRNN(2*model_dim, v_size)
            self.self_matching_attn = SelfMatchingAttention(v_size, h_size)
        else:
            self.gated_attn = GatedAttentionBasedRNNWithDotProduct(2*model_dim, v_size, dropout)
            self.self_matching_attn = SelfMatchingAttentionWithDotProduct(v_size, h_size, dropout)
        self.output_layer = PointerNetwork(2*h_size, 2*model_dim)
        self.criterion = nn.NLLLoss()

    def forward(self, passage, passage_ch, question, question_ch):
        passage_w_emb, passage_c_emb = self.word_emb(passage), self.char_emb(passage_ch)
        question_w_emb, question_c_emb = self.word_emb(question), self.char_emb(question_ch)
        passage = self.p_encoder(passage_c_emb, passage_w_emb)
        question = self.q_encoder(question_c_emb, question_w_emb)
        v = self.gated_attn(passage, question)
        h = self.self_matching_attn(v)
        start, end = self.output_layer(h, question)
        return start, end

    def train_step(self, batch, y1, y2, optim, scheduler):
        self.train()
        optim.zero_grad()
        p1, p2 = self.forward(*batch)
        loss1 = self.criterion(p1, y1)
        loss2 = self.criterion(p2, y2)
        loss = (loss1+loss2)/2
        loss.backward()
        optim.step()
        scheduler.step()

        return loss.item(), p1, p2

    def evaluate(self, batch, y1, y2):
        self.eval()
        with torch.no_grad():
            p1, p2 = self.forward(*batch)
            loss1 = self.criterion(p1, y1)
            loss2 = self.criterion(p2, y2)
            loss = (loss1+loss2)/2
        return loss.item(), p1, p2
