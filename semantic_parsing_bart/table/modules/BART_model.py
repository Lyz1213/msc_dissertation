from transformers import BartModel, BartTokenizer, BartForConditionalGeneration, BartConfig
import os
import torch.nn as nn

class BART(nn.Module):
    def __init__(self):
        super(BART, self).__init__()
        #model = BartModel.from_pretrained('facebook/bart-large')
        self.model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')

    def forward(self, src, mask, tgt):
        output = self.model(input_ids=src, attention_mask=mask, decoder_input_ids = tgt)


        return output