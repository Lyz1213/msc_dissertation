import torch.nn as nn
import table.modules
import table.Models
from table.TransformersModels import CopyGenerator, ParserModel
import torchtext.vocab
from table.modules.Embeddings import PartUpdateEmbedding
from table.modules.BERT_encoder import BERT
from table.modules.Transformer import transformerDecoder, transformerEncoder, PositionalEncoding, DecoderEmbeddings
from torch.nn import TransformerDecoder, TransformerDecoderLayer, TransformerEncoder, TransformerEncoderLayer
def make_word_embeddings(opt, word_dict, fields):
    word_padding_idx = word_dict.stoi[table.IO.PAD_WORD]
    num_word = len(word_dict)
    emb_word = nn.Embedding(num_word, opt.word_vec_size,
                            padding_idx=word_padding_idx)

    if len(opt.pre_word_vecs) > 0:
        if opt.word_vec_size == 150:
            dim_list = ['100', '50']
        elif opt.word_vec_size == 250:
            dim_list = ['200', '50']
        else:
            dim_list = [str(opt.word_vec_size), ]
        vectors = [torchtext.vocab.GloVe(
            name="6B", cache=opt.pre_word_vecs, dim=it) for it in dim_list]
        word_dict.load_vectors(vectors)
        emb_word.weight.data.copy_(word_dict.vectors)

    if opt.fix_word_vecs:
        # <unk> is 0
        num_special = len(table.IO.special_token_list)
        # zero vectors in the fixed embedding (emb_word)
        emb_word.weight.data[:num_special].zero_()
        emb_special = nn.Embedding(
            num_special, opt.word_vec_size, padding_idx=word_padding_idx)
        emb = PartUpdateEmbedding(num_special, emb_special, emb_word)
        return emb
    else:
        return emb_word

def make_embeddings(word_dict, vec_size):
    word_padding_idx = word_dict.stoi[table.IO.PAD_WORD]
    num_word = len(word_dict)
    w_embeddings = nn.Embedding(
        num_word, vec_size, padding_idx=word_padding_idx)
    return w_embeddings


def make_bert_encoder(opt):
    return BERT(opt.bert_name, opt.dim_size)

def make_transformer_encoder(opt, vocab):
    return transformerEncoder(dropout = opt.dropout, attention_dropout = opt.attention_dropout,
                              activation_dropout = opt.attention_dropout, encoder_embed_dim = opt.encoder_embed_dim,
                              max_src_positions = 512, no_scale_embedding = opt.no_scale_embedding, encoder_layers = opt.encoder_layers,
                              encoder_attention_heads = opt.encoder_attention_heads, encoder_ffn_embed_dim = opt.encoder_ffn_embed_dim, dictionary = vocab)

def make_transformer_decoder(opt, vocab):
    return transformerDecoder(dropout = opt.dropout, attention_dropout = opt.attention_dropout,
                              activation_dropout = opt.attention_dropout, decoder_embed_dim = opt.decoder_embed_dim, encoder_embed_dim = opt.encoder_embed_dim,
                              max_tgt_positions = 512, no_scale_embedding = opt.no_scale_embedding, decoder_layers = opt.decoder_layers,
                              decoder_attention_heads = opt.decoder_attention_heads, decoder_ffn_embed_dim = opt.decoder_ffn_embed_dim, dictionary = vocab)

def make_decoder(opt, type, vocab_tgt, vocab_ext):
    transformer_decoder = make_transformer_decoder(opt, vocab_tgt)
    if type == 'tgt':
        classifier = CopyGenerator(opt.dropout, opt.decoder_embed_dim, opt.decoder_embed_dim, vocab_tgt,
                                   vocab_ext, opt.copy_prb)
    else:
        classifier = nn.Sequential(
            nn.Dropout(opt.dropout),
            nn.Linear(opt.decoder_embed_dim, len(vocab_tgt)),
            nn.LogSoftmax())
    return transformer_decoder, classifier
def make_input_embedding(opt, vocab):
    return nn.Sequential(DecoderEmbeddings(vocab=vocab, embed_size= opt.decoder_embed_dim),
                                                     PositionalEncoding(d_model = opt.decoder_embed_dim, dropout=opt.dropout,
                                                                        max_len=512))
def make_src_embedding(opt, embedding):
    return nn.Sequential(embedding,
                         PositionalEncoding(d_model=opt.encoder_embed_dim, dropout=opt.dropout,
                                            max_len=512)
                         )
def generate_linear(opt, vocab, bias):
    m = nn.Linear(opt.decoder_embed_dim, len(vocab), bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.)
    return m
def make_torch_transformer_decoder(opt):
    decoder_layer = nn.TransformerDecoderLayer(d_model=opt.decoder_embed_dim, nhead=opt.decoder_attention_heads, dim_feedforward=opt.decoder_ffn_embed_dim, dropout=opt.dropout)
    transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=opt.decoder_layers)
    return transformer_decoder

def make_torch_transformer_encoder(opt):
    encoder_layer = nn.TransformerEncoderLayer(d_model=opt.encoder_embed_dim, nhead=opt.encoder_attention_heads, dim_feedforward=opt.decoder_ffn_embed_dim, dropout=opt.dropout)
    transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=opt.encoder_layers)
    return transformer_encoder


def make_transformer_base_model(opt, fields, checkpoint = None):
    w_embeddings = make_word_embeddings(opt, fields["src"].vocab, fields)
    q_embedding = make_src_embedding(opt, w_embeddings)
    q_encoder = make_torch_transformer_encoder(opt)
    #q_encoder = make_bert_encoder(opt)
    #lay_decoder, lay_classifier = make_decoder(opt, 'lay', fields['lay'].vocab, None)
    lay_embedding = make_input_embedding(opt, fields['lay'].vocab)
    lay_decoder = make_torch_transformer_decoder(opt)
    lay_encoder = make_transformer_encoder(opt, fields['lay'].vocab)
    lay_classifier = generate_linear(opt, fields['lay'].vocab,True)
    tgt_decoder, tgt_classifier = make_decoder(opt, 'tgt', fields['tgt_not_copy'].vocab,fields['copy_to_ext'].vocab)
    tgt_embeddings = make_embeddings(
        fields['tgt'].vocab, opt.decoder_embed_dim)
    model = ParserModel(q_encoder, lay_embedding, lay_decoder, lay_classifier, lay_encoder, tgt_embeddings, tgt_decoder, tgt_classifier, opt)
    if checkpoint is not None:
        print('Loading model')
        model.load_state_dict(checkpoint['model'])

    model.cuda()


    return model
