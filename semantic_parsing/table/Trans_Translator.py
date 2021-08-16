import torch
import math
from torch.autograd import Variable
import torch.nn.functional as F

import table
import table.IO
import table.ModelConstructor
import table.TransfomersModelConstructor
import table.Models
import table.modules
from table.Utils import add_pad, argmax, topk
from table.ParseResult import ParseResult
from table.Models import encode_unsorted_batch
from table.modules.Transformer import PositionalEmbedding, PositionalEncoding

def v_eval(a):
    return Variable(a, volatile=True)


def cpu_vector(v):
    return v.clone().view(-1).cpu()


def recover_layout_token(pred_list, vocab, max_sent_length):
    r_list = []
    for i in range(max_sent_length):
        r_list.append(vocab.itos[pred_list[i]])

        if r_list[-1] == table.IO.EOS_WORD:
            r_list = r_list[:-1]
            break
    return r_list


def recover_target_token(lay_skip, pred_list, vocab_tgt, vocab_copy_ext, max_sent_length):
    r_list = []
    pred_list = pred_list.tolist()
    for i in range(max_sent_length):
        if i < len(lay_skip) and lay_skip[i] not in set([table.IO.SKP_WORD]):
            tk = lay_skip[i]
        else:
            if pred_list[i] < len(vocab_tgt):
                tk = vocab_tgt.itos[pred_list[i]]
            else:
                tk = vocab_copy_ext.itos[pred_list[i] - len(vocab_tgt)]
            # filter topk results using layout information
            # k = pred_list[i].size(0)
            # for j in range(k):
            #     if pred_list[i][j] < len(vocab_tgt):
            #         tk = vocab_tgt.itos[pred_list[i][j]]
            #     else:
            #         tk = vocab_copy_ext.itos[pred_list[i][j] - len(vocab_tgt)]

            #     if i < len(lay_skip):
            #         if lay_skip[i] == 'NUMBER':
            #             is_number = True
            #             try:
            #                 __ = float(tk)
            #             except:
            #                 is_number = False
            #             if is_number:
            #                 break
            #             else:
            #                 continue
            #         else:
            #             break
            #     else:
            #         break
        r_list.append(tk)

        if r_list[-1] == table.IO.EOS_WORD:
            r_list = r_list[:-1]
            break
    return r_list


def get_decode_batch_length(dec, batch_size, max_sent_length):
    r_list = []
    for b in range(batch_size):
        find_len = None
        for i in range(max_sent_length):
            if dec[i, b] == table.IO.EOS:
                find_len = i
                break
        if find_len is None:
            r_list.append(max_sent_length)
        else:
            r_list.append(find_len)
    assert (len(r_list) == batch_size)
    return torch.LongTensor(r_list)


# ['(airline:e@1', '(argmin', '(and', 'flight@1', 'from@2', 'to@2', 'day_number@2', 'month@2', ')', 'fare@1', ')', ')']
def expand_layout_with_skip(lay_list):
    op_list = ['o', 'p', 'v', 's', 't']
    lay_skip_list, tgt_mask_list, lay_index_list = [], [], []
    for lay in lay_list:
        lay_skip = []
        for tk_lay in lay:
            if len(tk_lay) == 2 and tk_lay[0] in op_list and tk_lay[1].isdigit():
                lay_skip.append(tk_lay[0])
                for i in range(int(tk_lay[1])):
                    lay_skip.append(table.IO.SKP_WORD)
            elif len(tk_lay) == 3 and tk_lay[0] in op_list and tk_lay[1:].isdigit():
                lay_skip.append(tk_lay[0])
                for i in range(int(tk_lay[1:])):
                    lay_skip.append(table.IO.SKP_WORD)
            else:
                lay_skip.append(tk_lay)
        # print('lay is {} lay_skip is {}'.format(lay, lay_skip))
        lay_skip_list.append(lay_skip)
        tgt_mask_list.append(table.IO.get_tgt_mask(lay_skip))
        lay_index_list.append(table.IO.get_lay_index_(lay_skip))
    tgt_mask_seq = add_pad(tgt_mask_list, 1).float().t()
    lay_index_seq = add_pad(lay_index_list, 0).t()
    return lay_skip_list, tgt_mask_seq, lay_index_seq


class TransformerTranslator(object):
    def __init__(self, opt, dummy_opt={}):
        # Add in default model arguments, possibly added since training.
        self.opt = opt
        checkpoint = torch.load(opt.model,
                                map_location=lambda storage, loc: storage)
        self.fields = table.IO.TableDataset.load_fields(checkpoint['vocab'])

        model_opt = checkpoint['opt']
        model_opt.pre_word_vecs = opt.pre_word_vecs
        for arg in dummy_opt:
            if arg not in model_opt:
                model_opt.__dict__[arg] = dummy_opt[arg]

        self.model = table.TransfomersModelConstructor.make_transformer_base_model(
            model_opt, self.fields, checkpoint)
        self.model.eval()

        if model_opt.moving_avg > 0:
            for p, avg_p in zip(self.model.parameters(), checkpoint['moving_avg']):
                p.data.copy_(avg_p)

        if opt.attn_ignore_small > 0:
            self.model.lay_decoder.attn.ignore_small = opt.attn_ignore_small
            self.model.tgt_decoder.attn.ignore_small = opt.attn_ignore_small

    def run_test_lay_decoder(self, model, q, q_enc, max_dec_len, encoder_mask):
        batch_size = q.size(0)
        dec_list = []
        inp = torch.LongTensor(batch_size, 1).fill_(table.IO.BOS).cuda()
        for i in range(max_dec_len):
            inp = v_eval(inp)
            with torch.no_grad():
                # Compute the decoder output by repeatedly feeding it the decoded sentence prefix
                decoder_out = model.run_lay_decoder(q_enc, inp, encoder_mask)
            next_word = decoder_out[-1, :, :]
            next_word = argmax(next_word).unsqueeze(1)
            inp = torch.cat([inp, next_word], dim=1)
        inp = inp[:, 1:]
        inp = inp.transpose(0, 1)
        return inp
    def run_test_tgt_decoder(self, lay_all, lay_skip_list, tgt_mask_seq, lay_index_seq, q_encoder_out, q,
                             max_dec_len, tgt_not_copy_vocab, tgt_embeddings, tgt_pos_encoder, tgt_decoder, encoder_padding, copy_to_ext_wordpiece):
        batch_size = q.size(0)
        dec_list = []
        inp = torch.LongTensor(1, batch_size).fill_(table.IO.BOS).cuda()
        batch_index = torch.LongTensor(range(batch_size)).unsqueeze_(0).cuda()
        for i in range(min(max_dec_len, lay_index_seq.size(0))):
            # (1, batch)
            lay_index = lay_index_seq[i].unsqueeze(0)
            lay_select = lay_all[lay_index, batch_index, :]
            #inp.masked_fill_(inp.ge(len(tgt_not_copy_vocab)), table.IO.UNK)
            tgt_inp_emb = tgt_embeddings(v_eval(inp).long())
            tgt_mask_expand = v_eval(tgt_mask_seq[i].unsqueeze(
                0).unsqueeze(2).expand_as(tgt_inp_emb))
            tgt_inp = tgt_inp_emb.mul(tgt_mask_expand) + \
                  lay_select.mul(1 - tgt_mask_expand)
            #print('lay_select {}, tgt_inp_emb {}, tgt_mask_expand {}, inp {}'.format(lay_select.size(), tgt_inp_emb.size(), tgt_mask_expand.size(), tgt_inp.size()))
            tgt_inp = tgt_pos_encoder(tgt_inp)
            tgt_inp = tgt_inp.transpose(0, 1)
            tgt_out = tgt_decoder(tgt_inp, encoder_out=q_encoder_out, input_embedding=True, incremental_state=None, features_only=True,
                                  attn_padding_mask=None, encoder_padding_mask = encoder_padding)
            tgt_attn = tgt_out[1]['attn_state']
            tgt_hidden = tgt_out[0]
            tgt_hidden = tgt_hidden.contiguous().transpose(0, 1)
            #out: seq_len * batch * len(dic)
            out = self.model.tgt_classifier(tgt_hidden, tgt_attn, copy_to_ext_wordpiece)
            #print('out.size()',out.size())
            next_words = argmax(out)
            next_words = next_words[-1,:].unsqueeze(0)
            #print(next_words.size())
            inp = torch.cat([inp, next_words], dim = 0)
            #print('inp ', inp.size())
        inp = inp[1:,:]
        return inp
    def translate(self, batch):
        #print('hahah')
        bert_input, bert_len = batch.bert_input
        bert_input = bert_input.transpose(0, 1)
        batch_size = bert_input.size(0)
        layout_token_prune_list = [None for b in range(batch_size)]
        attention_mask = batch.attention_mask.transpose(0, 1)
        encoder_padding = batch.wordpiece_index.contiguous().transpose(0, 1).bool()
        q_encoder_out = self.model.q_encoder(bert_input, attention_mask)
        #q_encoder_out = q_encoder_out.transpose(0,1)
        # seq_len * batch * hidden
        lay_dec_out = self.run_test_lay_decoder(self.model, bert_input, q_encoder_out, self.opt.max_lay_len,
                                                encoder_padding)
        lay_dec_out = lay_dec_out.cpu()
        q_encoder_out = q_encoder_out.transpose(0,1)
        lay_list = []
        for b in range(batch_size):
            lay_field = 'lay'
            lay = recover_layout_token([lay_dec_out[i, b] for i in range(
                lay_dec_out.size(0))], self.fields[lay_field].vocab, lay_dec_out.size(0))
            lay_list.append(lay)
        lay_len = torch.LongTensor([len(lay_list[b])
                                    for b in range(batch_size)])
        # data used for layout encoding
        lay_dec = torch.LongTensor(
            lay_len.max(), batch_size).fill_(table.IO.PAD)
        for b in range(batch_size):
            for i in range(lay_len[b]):
                lay_dec[i, b] = self.fields['lay'].vocab.stoi[lay_list[b][i]]
        lay_dec = v_eval(lay_dec.cuda())
        lay_skip_list, tgt_mask_seq, lay_index_seq = expand_layout_with_skip(
            lay_list)
        #print('lay_index ', lay_index_seq.size())
        lay_dec = lay_dec.transpose(0,1)
        lay_out = self.model.lay_encoder(lay_dec)
        #lay_all: seq_len * batch * hidden
        lay_all = lay_out['src_out']
        tgt_pos_encoder = PositionalEncoding(d_model=512, dropout=0.1,
                                           max_len=512)
        tgt_pos_encoder.cuda()
        #tgt_out: seq_len * batch
        tgt_out = self.run_test_tgt_decoder(lay_all, lay_skip_list, tgt_mask_seq, lay_index_seq, q_encoder_out, bert_input,
                                       self.opt.max_tgt_len, self.fields['tgt'].vocab, 
                                       self.model.tgt_embeddings, tgt_pos_encoder, self.model.tgt_decoder, encoder_padding, batch.copy_to_ext_wordpiece)
        tgt_out = tgt_out.cpu()
        tgt_list = []
        for b in range(batch_size):
            tgt = recover_target_token(lay_skip_list[b], tgt_out[:, b], self.fields['tgt_not_copy'].vocab,
                                       self.fields['copy_to_ext'].vocab, tgt_out.size(0))
            tgt_list.append(tgt)
        # for i in range(len(lay_list)):
        #     print('lay: {} \n tgt: {} \n'.format(lay_list[i], tgt_list[i]))
        indices = cpu_vector(batch.indices.data)
        return [ParseResult(idx, lay, tgt, token_prune)
                for idx, lay, tgt, token_prune in zip(indices, lay_list, tgt_list, layout_token_prune_list)]
