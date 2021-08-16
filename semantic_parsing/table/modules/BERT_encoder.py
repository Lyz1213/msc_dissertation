from transformers import BertModel
import os
import torch.nn as nn


class BERT(nn.Module):
    def __init__(self, bert_name='base', d_model=512):
        super(BERT, self).__init__()
        if bert_name == 'base':
            path = os.path.join('BERT_pretrained_models', 'bert-base-uncased')
            name = 'bert-base-uncased'
        else:
            path = os.path.join('BERT_pretrained_models', 'bert-large-uncased')
            name = 'bert-large-uncased'
        unfreeze_layers = ['layer.10','layer.11','bert.pooler','out.']
        # for name ,param in model.named_parameters():
        #     param.requires_grad = False
        #     for ele in unfreeze_layers:
        #         if ele in name:
        #             param.requires_grad = True
        #             break
                
        self.model = BertModel.from_pretrained(name, cache_dir=os.path.join('BERT_pretrained_models', path))

        self.output_dim = 768 if bert_name == 'base' else 1024
        #self.linear_mapping = nn.Linear(self.output_dim, d_model,
                                        #bias=False)
        #nn.init.xavier_uniform(self.linear_mapping.weight)

    def forward(self, input, mask):
        output = self.model(input_ids=input, attention_mask=mask)
        #output = self.linear_mapping(output[0])
        return output[0]
