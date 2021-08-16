from transformers import BertTokenizer, BertModel
import warnings
# import table.IO
import os
import torch
class SrcVocab(object):
    def __init__(self, bertModel):
        self.tokenizer = BertTokenizer.from_pretrained(bertModel)
        self.tokenizer.ids_to_tokens[1] = '<s>'
        self.tokenizer.vocab['<s>'] = 1
        self.tokenizer.vocab['</s>'] = 2
        self.tokenizer.ids_to_tokens[2] = '</s>'
        self.tokenizer.vocab['<sk>'] = 3
        self.tokenizer.ids_to_tokens[3] = '<sk>'
        self.bos_token = '<s>'
        self.bos_token_id = 1
        self.eos_token = '</s>'
        self.eos_token_id = 2

    def encodeSeqs(self, seqs):
        # for sequence in seqs:
        #     if '[UNK]' in self.tokenizer(text = sequence, is_split_into_words = True):
        #         warnings.warn('[UNK] in target sequence tokenization: You need to add the corresponding items to the vocabulary')
        # token_sequences = [self.tokenizer.tokenize(text = sequence, is_split_into_words = True) for sequence in seqs]
        token_sequences = self.tokenizer(text=seqs, is_split_into_words = True, padding = True, return_token_type_ids = False)
        return token_sequences

    def seq2ID(self, seq):
        return [self.tokenizer.convert_tokens_to_ids(token) for token in seq]

    def decode_token2token(self, idseq):
        # decoded = []
        # for id in idseq:
        #     print(self.tokenizer.convert_ids_to_tokens(id))
        #     decoded.append(self.tokenizer.convert_ids_to_tokens(id))
        # return decoded
        return [self.tokenizer.convert_ids_to_tokens(ids) for ids in idseq]

    def word_piece_index(self, src, seq):
        word_piece = [1 if token.startswith('##') else 0 for token in seq]
        word_piece[0] = 1
        word_piece[-1] = 1
        lenth = sum([1 - index for index in word_piece])
        if lenth != len(src):
            word_piece = []
            # flag = False
            # for token in seq:
            #     if token not in src:
            #         if token == '[CLS]' or token == '[SEP]':
            #             word_piece.append(1)
            #         else:
            #             if not flag:
            #                 word_piece.append(0)
            #                 flag = True
            #             else:
            #                 word_piece.append(1)
            #         #print('token is {}, new word_piece is {}'.format(token,word_piece))
            #     else:
            #         flag = False
            #         word_piece.append(0)
            src_index = 0
            #print('src is {} seq is {}'.format(src, seq))
            for token in seq:
                if src_index >= len(src):
                    word_piece.append(1)
                else:
                    if token == ['CLS'] or token == '[SEP]':
                        word_piece.append(1)
                    else:
                        if token == src[src_index] or src[src_index].startswith(token):
                            word_piece.append(0)
                            src_index += 1
                        else:
                            word_piece.append(1)
            length = sum([1 - index for index in word_piece])
            if length != len(src):
                #print('src is {} \n seq is {} \n word_piece {} \n'.format(src, seq, word_piece))
                return None
        return word_piece
if __name__ == '__main__':
    vocab = SrcVocab('bert-base-uncased')
    # js_list = table.IO.read_txt('/Users/liyanzhou/Desktop/Edinburgh/Dissertation/semantic_parsing/data_model/test.txt')
    # srcseqs = js_list[0]['src']
    # srcseqs.append('embeddings')
    # batch_seqs = vocab.encodeSeqs(srcseqs)
    # print(batch_seqs)
    # decoded = vocab.decode_token2token(batch_seqs['input_ids'])
    # print(decoded)
    # word_piece_index = vocab.word_piece_index(decoded)
    # print(word_piece_index)
    # print(vocab.tokenizer.vocab_size)
    # id_seq = vocab.seq2ID(srcseqs[0])
    # print('original is {} and converted is {}'.format(srcseqs[0], id_seq))
    # d = vocab.decode_token2token(id_seq)
    # print(d)