import torch
import math
from torch.autograd import Variable
import torch.nn.functional as F

import table
import table.IO
import table.ModelConstructor
import table.BARTModelConstructor
import table.Models
import table.modules
from table.Utils import add_pad, argmax, topk
from table.Tokenize import SrcVocab
from table.ParseResult import ParseResult
from table.Models import encode_unsorted_batch
from table.modules.Transformer import PositionalEmbedding, PositionalEncoding


def add_pad(b_list, pad_index, return_tensor=True):
    max_len = max((len(b) for b in b_list))
    r_list = []
    for b in b_list:
        r_list.append(b + [pad_index] * (max_len - len(b)))
    if return_tensor:
        return torch.LongTensor(r_list).cuda()
    else:
        return r_list


def v_eval(a):
    return Variable(a, volatile=True)


def cpu_vector(v):
    return v.clone().view(-1).cpu()



def modify_tgt(seq):
    modified = []
    seq = seq[1:]
    for i in range(len(seq)):
        if seq[i]!= '</s>' and seq[i]!='<s>':
            modified.append(seq[i])
        else:
            break
    return modified
def modify_target(seq):
    modified = []
    seq = seq[2:]
    for i in range(len(seq)):
        if seq[i] != '<s>' and seq[i] != '</s>':
            modified.append(seq[i])
        else:
            break
    return modified


def modify_lay(seq):
    eval_list = []
    modified = []
    lay_skip = []
    seq = seq[1:]
    prev_key = ''
    for i in range(len(seq)):
        if seq[i] == '<s>' or seq[i] == '</s>':
            break
        elif seq[i] == '{' or seq[i] == 'Ġ{':
            key = '{H' + seq[i + 1].replace('Ġ', '')
            modified.append(key)
            lay_skip.append(key)
            eval_list.append(seq[i])
        elif seq[i] == 'ĠS' or seq[i] == 'ĠV' or seq[i] == 'ĠP' or seq[i] == 'ĠO' or seq[i] == 'ĠT':
            if seq[i] == prev_key:
                lay_skip.append(table.IO.SKP_WORD)
                eval_list.append(seq[i])
            elif seq[i] == 'ĠV':
                prev_key = seq[i]
                modified.append('V')
                lay_skip.append('V')
                eval_list.append(seq[i])
            else:
                prev_key = seq[i]
                modified.append(seq[i])
                lay_skip.append(seq[i])
                eval_list.append(seq[i])
        elif seq[i] == 'Ġ}' or seq[i] == 'Ġsv' or seq[i] == 'Ġov':
            modified.append(seq[i])
            lay_skip.append(seq[i])
            eval_list.append(seq[i])
        else:
            #print('seq[i]', seq[i])
            eval_list.append(seq[i])
            if i >= 1:
                if seq[i - 1] != '{' and seq[i - 1] != 'Ġ{':
                    #print('seq[i******', seq[i])
                    lay_skip.append(table.IO.SKP_WORD)
            else:
                #print('seqseqseqseqseq', seq[i])
                lay_skip.append(table.IO.SKP_WORD)
    return modified, lay_skip, eval_list

def modify_lay_(seq):
    eval_list = []
    modified = []
    lay_skip = []
    seq = seq[1:]
    for token in seq:
        if token == '<s>' or token == '</s>':
            break
        elif token.isdigit():
            eval_list.append(token)
            for i in range(int(token)):
                lay_skip.append(table.IO.SKP_WORD)
        else:
            eval_list.append(token)
            lay_skip.append(token)
            modified.append(token)
    return modified, lay_skip, eval_list



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
        self.vocab = SrcVocab()
        checkpoint = torch.load(opt.model,
                                map_location=lambda storage, loc: storage)
        self.fields = table.IO.TableDataset.load_fields(checkpoint['vocab'])

        model_opt = checkpoint['opt']
        model_opt.pre_word_vecs = opt.pre_word_vecs
        for arg in dummy_opt:
            if arg not in model_opt:
                model_opt.__dict__[arg] = dummy_opt[arg]

        self.model = table.BARTModelConstructor.make_bart_base_model(
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
                             max_dec_len, tgt_not_copy_vocab, tgt_embeddings, tgt_pos_encoder, tgt_decoder,
                             encoder_padding, copy_to_ext_wordpiece):
        batch_size = q.size(0)
        dec_list = []
        inp = torch.LongTensor(1, batch_size).fill_(table.IO.BOS).cuda()
        batch_index = torch.LongTensor(range(batch_size)).unsqueeze_(0).cuda()
        for i in range(min(max_dec_len, lay_index_seq.size(0))):
            # (1, batch)
            lay_index = lay_index_seq[i].unsqueeze(0)
            lay_select = lay_all[lay_index, batch_index, :]
            # inp.masked_fill_(inp.ge(len(tgt_not_copy_vocab)), table.IO.UNK)
            tgt_inp_emb = tgt_embeddings(v_eval(inp).long())
            tgt_mask_expand = v_eval(tgt_mask_seq[i].unsqueeze(
                0).unsqueeze(2).expand_as(tgt_inp_emb))
            tgt_inp = tgt_inp_emb.mul(tgt_mask_expand) + \
                      lay_select.mul(1 - tgt_mask_expand)
            # print('lay_select {}, tgt_inp_emb {}, tgt_mask_expand {}, inp {}'.format(lay_select.size(), tgt_inp_emb.size(), tgt_mask_expand.size(), tgt_inp.size()))
            tgt_inp = tgt_pos_encoder(tgt_inp)
            tgt_inp = tgt_inp.transpose(0, 1)
            tgt_out = tgt_decoder(tgt_inp, encoder_out=q_encoder_out, input_embedding=True, incremental_state=None,
                                  features_only=True,
                                  attn_padding_mask=None, encoder_padding_mask=encoder_padding)
            tgt_attn = tgt_out[1]['attn_state']
            tgt_hidden = tgt_out[0]
            tgt_hidden = tgt_hidden.contiguous().transpose(0, 1)
            # out: seq_len * batch * len(dic)
            out = self.model.tgt_classifier(tgt_hidden, tgt_attn, copy_to_ext_wordpiece)
            # print('out.size()',out.size())
            next_words = argmax(out)
            next_words = next_words[-1, :].unsqueeze(0)
            # print(next_words.size())
            inp = torch.cat([inp, next_words], dim=0)
            # print('inp ', inp.size())
        inp = inp[1:, :]
        return inp
    def self_translate_lay(self, src,attention_mask, batch_size):
        encoder = self.model.enc_dec.get_encoder()
        decoder = self.model.enc_dec.get_decoder()
        encoder_out = encoder(src)
        encoder_out = (encoder_out.last_hidden_state,)
        inp = torch.LongTensor(1,batch_size).fill_(table.IO.BOS).cuda()
        for i in range(80):
            inp = inp.transpose(0,1)
            tgt_inp_emb = decoder.embed_tokens(inp) * decoder.embed_scale
  
            tgt_outs = self.model.enc_dec(attention_mask = attention_mask, encoder_outputs=encoder_out,
                                        decoder_inputs_embeds=tgt_inp_emb)
            tgt_out = tgt_outs.logits.transpose(0,1)
                #print('tgt_out ',tgt_out.size())
            next_words = argmax(tgt_out)
                #print('next_words', next_words.size())
            next_words = next_words[-1, :].unsqueeze(0)
            inp = inp.transpose(0,1)
            inp = torch.cat([inp, next_words], dim=0)
            #print('inp size ', inp.size())
        tgt_list = []
        for b in range(inp.size(1)):
            tgt = self.vocab.tokenizer.convert_ids_to_tokens(inp[:,b])
            tgt_list.append(tgt)
        return tgt_list, encoder_out
            #print('tgt is ', tgt)

    def translate(self, batch):
        batch_size = batch.bart_src.size(1)
        bart_src = batch.bart_src.transpose(0, 1)
        # print('src ', bart_src.size())
        attention_mask = batch.attention_mask.transpose(0, 1)
        # lay_index = lay_index.transpose(0,1)
        # tgt_mask = tgt_mask.transpose(0,1)
        # print('tgt_loss ', tgt_loss.size()

        layout_token_prune_list = [None for b in range(batch_size)]
        lay_list = [None for b in range(batch_size)]
        tgts, _ = self.self_translate_lay(bart_src, attention_mask, batch_size)
        tgt_list = []
        for tgt in tgts:
            modified = modify_tgt(tgt)
            tgt_list.append(modified)
        indices = cpu_vector(batch.indices.data)
            #print('tgt_list is ', tgt_list)
            # print('tgt_list is ',tgt_list)
        return [ParseResult(idx, lay, tgt, token_prune)
                for idx, lay, tgt, token_prune in zip(indices, lay_list, tgt_list, layout_token_prune_list)]







