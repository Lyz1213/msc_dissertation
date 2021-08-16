import torch.nn as nn
import table.modules
import table.Models
from transformers import BartModel, BartTokenizer, BartForConditionalGeneration, BartConfig
from table.BartModel import ParserModel
import torchtext.vocab
from table.modules.BATRModel import BART
from table.Tokenize import SrcVocab

def make_bart_base_model(opt, fields, checkpoint = None):
    #enc_dec = BART()
    vocab = SrcVocab()
    enc_dec = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
    enc_dec.resize_token_embeddings(len(vocab.added_token) + len(vocab.tokenizer))
    model = ParserModel(enc_dec, opt)
    if checkpoint is not None:
        print('Loading model')
        model.load_state_dict(checkpoint['model'])
    model.cuda()
    return model