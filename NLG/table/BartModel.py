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

class ParserModel(nn.Module):
    def __init__(self, enc_dec,
                 model_opt):
        super(ParserModel, self).__init__()
        self.enc_dec = enc_dec
        self.opt = model_opt
    def forward(self, input_ids, attention_mask, dec_inp):
        input_ids = input_ids.transpose(0,1)
        attention_mask = attention_mask.transpose(0,1)
        dec_inp = dec_inp.transpose(0,1)
        dec_inp = dec_inp[:, :-1]
       
        #print('input{} mask {} dec{}'.format(input_ids.size(), attention_mask.size(),dec_inp.size()))
        output = self.enc_dec(input_ids = input_ids, attention_mask=attention_mask, decoder_input_ids=dec_inp)
        #print('output is ', argmax(output.logits).size())
        return output.logits.transpose(0,1)