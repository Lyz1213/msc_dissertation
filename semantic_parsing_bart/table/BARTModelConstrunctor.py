import torch.nn as nn
import table.modules
import table.Models
from table.BARTModel import ParserModel
from transformers import BartModel, BartTokenizer, BartForConditionalGeneration, BartConfig
import torchtext.vocab
from table.modules.BART_model import BART
from table.Tokenize import SrcVocab

def make_bart_base_model(opt, fields, checkpoint = None):
    vocab = SrcVocab()
    if opt.model_type == 'onestage':
        enc_dec = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
        enc_dec.resize_token_embeddings(len(vocab.added_token) + len(vocab.tokenizer))
        model = ParserModel(enc_dec, None, opt)
    else:
        enc_dec1 = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
        enc_dec1.resize_token_embeddings(len(vocab.added_token) + len(vocab.tokenizer))
        enc_dec2 = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
        enc_dec2.resize_token_embeddings(len(vocab.added_token) + len(vocab.tokenizer))
        model = ParserModel(enc_dec1, enc_dec2, opt)
    if checkpoint is not None:
        print('Loading model')
        model.load_state_dict(checkpoint['model'])
    model.cuda()
    return model