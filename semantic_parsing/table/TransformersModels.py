from __future__ import division
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
import torch.nn.functional as F
from table.modules.Transformer import PositionalEmbedding
import table
from table.Utils import aeq, sort_for_pack
from table.modules.embed_regularize import embedded_dropout
from table.modules.cross_entropy_smooth import onehot
from table.Tokenize import SrcVocab
from table.Trainer import _debug_batch_content
from torch.nn import TransformerDecoder


class CopyGenerator(nn.Module):
    """Generator module that additionally considers copying
    words directly from the source. For each source sentence we have a `src_map` that maps each source word to an index in `tgt_dict` if it known, or else to an extra word. The copy generator is an extended version of the standard generator that computse three values.
    * :math:`p_{softmax}` the standard softmax over `tgt_dict`
    * :math:`p(z)` the probability of instead copying a word from the source, computed using a bernoulli
    * :math:`p_{copy}` the probility of copying a word instead. taken from the attention distribution directly.
    The model returns a distribution over the extend dictionary, computed as
    :math:`p(w) = p(z=1)  p_{copy}(w)  +  p(z=0)  p_{softmax}(w)`
    Args:
       hidden_size (int): size of input representation
       tgt_dict (Vocab): output target dictionary
    """

    def __init__(self, dropout, hidden_size, context_size, tgt_dict, ext_dict, copy_prb):
        super(CopyGenerator, self).__init__()
        self.copy_prb = copy_prb
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size, len(tgt_dict))
        if copy_prb == 'hidden':
            self.linear_copy = nn.Linear(hidden_size, 1)
        elif copy_prb == 'hidden_context':
            self.linear_copy = nn.Linear(hidden_size + context_size, 1)
        else:
            raise NotImplementedError
        self.tgt_dict = tgt_dict
        self.ext_dict = ext_dict

    def forward(self, hidden, attn, copy_to_ext):
        """
        Compute a distribution over the target dictionary extended by the dynamic dictionary implied by compying source words.
        Args:
           hidden (`FloatTensor`): hidden outputs `[tlen * batch, hidden_size]`
           attn (`FloatTensor`): attn for each `[tlen * batch, src_len]`
           copy_to_ext (`FloatTensor`): A sparse indicator matrix mapping each source word to its index in the "extended" vocab containing. `[src_len, batch]`
           copy_to_tgt (`FloatTensor`): A sparse indicator matrix mapping each source word to its index in the target vocab containing. `[src_len, batch]`
        """
        dec_seq_len = hidden.size(0)
        batch_size = hidden.size(1)
        # -> (targetL_ * batch_, rnn_size)
        hidden = hidden.contiguous().view(dec_seq_len * batch_size, -1)
        # -> (targetL_ * batch_, sourceL_)
        attn = attn.contiguous().view(dec_seq_len * batch_size, -1)

        # CHECKS
        batch_by_tlen, _ = hidden.size()
        batch_by_tlen_, slen = attn.size()
        slen_, batch = copy_to_ext.size()
        aeq(batch_by_tlen, batch_by_tlen_)
        aeq(slen, slen_)
        copy = F.sigmoid(self.linear_copy(hidden))
        hidden = self.dropout(hidden)

        # Original probabilities.
        logits = self.linear(hidden)
        # logits[:, self.tgt_dict.stoi[table.IO.PAD_WORD]] = -float('inf')
        prob_log = F.log_softmax(logits)
        # return prob_log.view(dec_seq_len, batch_size, -1)

        # Probability of copying p(z=1) batch.
        # copy = F.sigmoid(self.linear_copy(hidden))

        def safe_log(v):
            return torch.log(v.clamp(1e-3, 1 - 1e-3))

        # Probibility of not copying: p_{word}(w) * (1 - p(z))
        out_prob_log = prob_log + safe_log(copy).expand_as(prob_log)
        mul_attn = torch.mul(attn, 1.0 - copy.expand_as(attn))
        # copy to extend vocabulary
        copy_to_ext_onehot = onehot(
            copy_to_ext, N=len(self.ext_dict), ignore_index=self.ext_dict.stoi[table.IO.UNK_WORD]).float()
        ext_copy_prob = torch.bmm(mul_attn.view(-1, batch, slen).transpose(0, 1),
                                  copy_to_ext_onehot.transpose(0, 1)).transpose(0, 1).contiguous().view(-1, len(
            self.ext_dict))
        ext_copy_prob_log = safe_log(ext_copy_prob)

        return torch.cat([out_prob_log, ext_copy_prob_log], 1).view(dec_seq_len, batch_size, -1)

        # copy to target vocabulary
        # copy_to_tgt_onehot = onehot(
        #     copy_to_tgt, N=len(self.tgt_dict), ignore_index=self.tgt_dict.stoi[table.IO.UNK_WORD]).float()
        # tgt_add_copy_prob = torch.bmm(mul_attn.view(-1, batch, slen).transpose(0, 1),
        #                               copy_to_tgt_onehot.transpose(0, 1)).transpose(0, 1).contiguous().view(-1, len(self.tgt_dict))
        # out_prob = torch.exp(out_prob_log) + tgt_add_copy_prob
        #
        # return torch.log(torch.cat([out_prob, ext_copy_prob], 1)).view(dec_seq_len, batch_size, -1)


class ParserModel(nn.Module):
    def __init__(self, q_encoder, lay_embedding, lay_decoder, lay_classifier, lay_encoder, tgt_embeddings, tgt_pos_encoder, tgt_decoder, tgt_classifier,
                 model_opt):
        super(ParserModel, self).__init__()
        if model_opt.seprate_encoder:
            self.q_encoder, self.q_tgt_encoder = q_encoder
        else:
            self.q_encoder = q_encoder
        self.lay_embedding = lay_embedding
        self.lay_decoder = lay_decoder
        self.lay_classifier = lay_classifier
        self.lay_encoder = lay_encoder
        self.tgt_embeddings = tgt_embeddings
        self.tgt_pos_encoder = tgt_pos_encoder
        self.tgt_decoder = tgt_decoder
        self.tgt_classifier = tgt_classifier
        self.opt = model_opt

    def run_lay_decoder(self, encoder_out, lay_input, encoder_padding):
        '''
        encoder_out: seq_len * batch * hidden_dim
        lay_input: batch * seq_len
        encoder_padding:
        '''
        def generate_square_subsequent_mask(sz):
            mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
            mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
            return mask
        #encoder_out = encoder_out.transpose(0,1)
        seq_len = lay_input.size(1)
        self_attn_mask = generate_square_subsequent_mask(seq_len).cuda()
        lay_input = self.lay_embedding(lay_input)
        lay_input = lay_input.transpose(0,1)
        output = self.lay_decoder(tgt = lay_input, memory = encoder_out, tgt_mask = self_attn_mask, memory_key_padding_mask = encoder_padding)
        output = self.lay_classifier(output)
        return output

    def forward(self, q, q_len, bert_input, bert_len, attention_mask, lay, lay_e, lay_len, lay_index, tgt_mask, tgt,
                lay_parent_index, tgt_loss, copy_to_ext, copy_to_ext_wordpiece, wordpiece_index):
        # bert_input, q, attention_mask size: seq_len*batch_size
        vocab = SrcVocab('bert-base-uncased')
        batch_size = q.size(1)
        bert_token_input = [vocab.decode_token2token(seq.tolist()) for seq in bert_input.transpose(0, 1)]
        # print('bert_token_input', bert_token_input)
        # print('q_len size', q_len.size())
        #print('q size', q.size())
        # print('bert_input size', bert_input.size())
        # print('bert_len size', bert_len)
        # print('attention mask size', attention_mask)
        bert_input = bert_input.transpose(0, 1)
        attention_mask = attention_mask.transpose(0, 1)
        # bert_input, attention_mask size: batch*seq_len
        # print('bert_input {}, attention_mask {}'.format(bert_input.size(), attention_mask.size()))
        # q_encoder_out size: batch * seq_len * hidden_dim
        q_encoder_out = self.q_encoder(bert_input, attention_mask)
        #q_encoder_out = self.q_encoder(bert_input,None)
        # print('encoder_out', q_encoder_out.size())

        # decode lay
        lay_input = lay[:-1].transpose(0, 1)

        # print('lay size{}, lay input size {}'.format(lay.size(), lay_input.size()))
        # q_encoder_out size: seq_len * batch *hidden_dim
        q_encoder_out = q_encoder_out.transpose(0, 1)
        encoder_padding = wordpiece_index.contiguous().transpose(0,1).bool()
        another_padding = (1-attention_mask).bool()
        lay_dec_out = self.run_lay_decoder(q_encoder_out, lay_input, another_padding)
        #print('lay_dec_out', lay_dec_out.size())
        # lay_dec_out = self.lay_decoder(lay_input, q_encoder_out, False, None, True, None, encoder_padding)
        # seq_len = lay_dec_out[0].size(1)
        # sketch_out = lay_dec_out[0].contiguous().view(seq_len * batch_size, -1)
        # sketch_out = self.lay_classifier(sketch_out)
        # sketch_out = sketch_out.contiguous().view(seq_len, batch_size, -1)
        # lay_dec_out: forward_state: batch * seq_len * len(vocab)
        #             attn_state: batch * tgt_len * src_len
        #             inner_state: list: seq_len * batch * num_feature
        # layout_encoding
        #sketch_out = lay_dec_out[0].contiguous().transpose(0,1)
        #print('sketch', sketch_out.shape)
        lay_e_len = lay_len - 2
        lay_e = lay_e.transpose(0, 1)
        #print(lay_e.size())
        lay_encoder_out = self.lay_encoder(lay_e)
        # lay_encoder_out: src_out: seq_len * batch * hidden_dim
        #                 src_embeddings: batch * seq_len * hidden_dim
        #                 src_padding_mask: batch * seq_len
        # print(lay_encoder_out['src_padding_mask'])
        lay_out = lay_encoder_out['src_out']
        batch_index = torch.LongTensor(range(batch_size)).unsqueeze_(
            0).cuda().expand(lay_index.size(0), lay_index.size(1))
        # (tgt_len, batch, lay_size)
        lay_select = lay_out[lay_index.data, batch_index, :]
        # print('lay_select size ', lay_select.size())
        # (tgt_len, batch, lay_size)
        tgt_inp_emb = self.tgt_embeddings(tgt[:-1])
        # (tgt_len, batch) -> (tgt_len, batch, lay_size)
        tgt_mask_expand = tgt_mask.unsqueeze(2).expand_as(tgt_inp_emb)
        dec_inp = tgt_inp_emb.mul(tgt_mask_expand) + \
                  lay_select.mul(1 - tgt_mask_expand)
        #print('lay_select {}, tgt_inp_emb {}, tgt_mask_expand {}, inp {}'.format(lay_select.size(), tgt_inp_emb.size(), tgt_mask_expand.size(), dec_inp.size()))

        # fake = lay_index + tgt[:-1]
        # print('fake is {} \n tgt_loss is {}'.format(fake, tgt_loss))
        # fake_pos = embed_positions(fake)
        # print(fake_pos == tgt_loss_pos)
        # print('pos {}, embed {}'.format(tgt_pos.size(), tgt_inp_emb.size()))
        #tgt_inp = self.tgt_pos_encoder(dec_inp)
        # embed_positions = PositionalEmbedding(
        #     self.tgt_decoder.embed_dim, padding_idx=self.tgt_decoder.padding_idx,
        #     init_size=self.tgt_decoder.max_tgt_positions + self.tgt_decoder.padding_idx + 1
        # )
        # embed_positions.cuda()
        # # fake_pos = embed_positions(fake)
        # tgt_pos = embed_positions(tgt_loss[:-1])
        # # print(fake_pos == tgt_loss_pos)
        # # print('pos {}, embed {}'.format(tgt_pos.size(), tgt_inp_emb.size()))
        # tgt_inp = tgt_pos + dec_inp
        #print('dec inp ', dec_inp.size())
        tgt_inp = self.tgt_pos_encoder(dec_inp)
        tgt_token = tgt_loss[:-1].transpose(0, 1)
        attn_padding_mask = tgt_token.eq(self.tgt_decoder.padding_idx) if tgt_token.eq(
            self.tgt_decoder.padding_idx).any() else None
        #print('atten_padding_mask size {}, attention mask size {}, wordpiece_index size {}'.format(
        #    attn_padding_mask.size(), attention_mask.size(), wordpiece_index.size()))
        tgt_inp = tgt_inp.contiguous().transpose(0, 1)
        tgt_out = self.tgt_decoder(tgt_inp, q_encoder_out, True, None, True, attn_padding_mask, encoder_padding)
        # print('tgt_out ', tgt_out[1]['attn_state'].size())
        # print('q_len {}\nq {} \n copy_to_ext {}\n'.format(q_len,q.size(), copy_to_ext.size()))
        #print('copy_to_ext_wordpiece {} \n bert_input {} \n word_piece_index {}'.format(copy_to_ext_wordpiece.size(),
        #                                                                                bert_input.size(),
        #                                                                                wordpiece_index.size()))
        # print('copy_to_ext_wordpiece {} \n word_piece_index {} \n attention mask {}'.format(copy_to_ext_wordpiece,wordpiece_index, attention_mask))
        tgt_attn = tgt_out[1]['attn_state']
        tgt_hidden = tgt_out[0]
        tgt_hidden = tgt_hidden.contiguous().transpose(0, 1)
        #out: seq_len * batch * len(dic)
        out = self.tgt_classifier(tgt_hidden, tgt_attn, copy_to_ext_wordpiece)
        #print('sketch out ', sketch_out)
        return lay_dec_out, out, None, None