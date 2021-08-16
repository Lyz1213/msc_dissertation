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
from table.Utils import argmax

class ParserModel(nn.Module):
    def __init__(self, enc_dec1, enc_dec2,
                 model_opt):
        super(ParserModel, self).__init__()
        self.enc_dec1 = enc_dec1
        self.enc_dec2 = enc_dec2
        self.opt = model_opt
    def forward(self, bart_src, bart_tgt, bart_lay_train, bart_lay_test, lay_index, tgt_mask, attention_mask, tgt_loss, sketch_train):
        batch_size = bart_src.size(1)

        bart_src = bart_src.transpose(0,1)
        #print('src ', bart_src.size())
        attention_mask = attention_mask.transpose(0,1)
        bart_tgt = bart_tgt.transpose(0,1)
        bart_lay_train = bart_lay_train.transpose(0,1)
        bart_lay_test = bart_lay_test.transpose(0,1)
        #lay_index = lay_index.transpose(0,1)
        #tgt_mask = tgt_mask.transpose(0,1)
        tgt_loss = tgt_loss.transpose(0,1)
        #dec_inp = sketch_train.transpose(0,1)[:,:-1]
        dec_inp = tgt_loss[:,:-1]
        #print('tgt_loss ', tgt_loss.size())
        lay_out = bart_lay_train[:,:-1]
        #dec_inp = bart_lay_train[:,:-1]
        lay_in = lay_out
        #lay_in = bart_lay_test[:,:-1]
        if self.opt.model_type == 'onestage':
            output = self.enc_dec1(bart_src,attention_mask,dec_inp)
            #sketch = self.enc_dec1.generate(bart_src)
            #sketch = sketch.transpose(0,1)
            #print(sketch.size())
            sketch = None
            return output.logits.transpose(0,1), sketch
        else:
            #lay_output = self.enc_dec1(bart_src, attention_mask, sketch_train.transpose(0,1)[:,:-1], output_attentions = True)
            lay_output = self.enc_dec1(bart_src, attention_mask, lay_out, output_attentions = True)
            #for key in lay_output:
            #    print('key is ',key)
            #batch*seq_len*vocab_len
            sketch = lay_output.logits.transpose(0,1)
            #print('cross attention', lay_output.cross_attentions[-1].size())
            #print('encoder attention', lay_output.encoder_attentions[-1].size())
            encoder_out = (lay_output.encoder_last_hidden_state,)
            encoder_outs = {'last_hidden_state':encoder_out[0]}
            #print('encoder_out', encoder_out[0].size())
            
            encoder2 = self.enc_dec2.get_encoder()
            lay_encode = encoder2(bart_lay_test)
            #for key in lay_encode:
                #print('2 key is ', key)
            #print('lay_encode ',lay_encode.last_hidden_state.size())
            lay_encode = lay_encode.last_hidden_state.transpose(0,1)
            #print('TODO')
            decoder2 = self.enc_dec2.get_decoder()
            inp = torch.LongTensor(1, batch_size).fill_(table.IO.BOS).cuda()

            tgt_embed = decoder2.embed_tokens(bart_tgt[:,:-1]) * decoder2.embed_scale
            tgt_embed = tgt_embed.transpose(0,1)
            #print('tgt_embed', tgt_embed.size())
            batch_index = torch.LongTensor(range(batch_size)).unsqueeze_(
            0).cuda().expand(lay_index.size(0), lay_index.size(1))

            #print('batch_index ', batch_index.size())
            lay_select = lay_encode[lay_index.data, batch_index, :]
            #print('lay_selet ',lay_select.size())
            # (tgt_len, batch) -> (tgt_len, batch, lay_size)
            tgt_mask_expand = tgt_mask.unsqueeze(2).expand_as(tgt_embed)
            dec_inp = tgt_embed.mul(tgt_mask_expand) + \
            lay_select.mul(1 - tgt_mask_expand)

            #print('dec_inp',dec_inp.size())
            dec_inp = dec_inp.transpose(0,1)
            #print('mask{}\nencoder_out{}\ndec_inp{}'.format(attention_mask.size(), encoder_out.size(), dec_inp.size()))
            tgt_out = self.enc_dec2(attention_mask = attention_mask, encoder_outputs = encoder_out, decoder_inputs_embeds = dec_inp)

            #for key in tgt_out:
            #    print('key 3 ', key)
            alist = tgt_out.logits.transpose(0,1)
            sketch_ge = self.enc_dec1.generate(bart_src, attention_mask = attention_mask).transpose(0,1)
            #return alist, sketch
            return alist, (sketch, sketch_ge)
            





