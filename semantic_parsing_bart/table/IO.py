import torch
import torchtext.data
import torchtext.vocab
from collections import Counter, defaultdict
from itertools import chain
from table.Tokenize import SrcVocab

PAD_WORD = '<pad>'
PAD = 1
BOS_WORD = '<s>'
BOS = 0
EOS_WORD = '</s>'
EOS = 2
UNK_WORD = '<unk>'
UNK = 3
SKP_WORD = '<sk>'
tgt_not_copy_list = ['{Hlist', '{Hvalue', '{Hcount', '{Hgreater', '{Hequal', '{Hless', '{His','{Hmax','{Hmin', 'O', 'P', 'V', 'T', 'S', 'X', 'XX', 'A', 'AA', '}', 'type', 'location', 'contains', 'starts_with', 'ov', 'sv','label']
def load_template(path):
    templates = []
    with open(path) as fread:
        lines = fread.readlines()
        for line in lines:
            templates.append(line.strip())
    return templates


def get_lay_skip(src, lay_token):
    non_skip = ['S', 'P', 'O', 'T', '}', 'V']
    non_skip.extend(lay_token)
    non_skip = set(non_skip)
    lay_skip = []
    tgt_ = []
    skecth_train = []
    prev_token = 'v'
    for i in range(len(src)):
        if not src[i].startswith('{H') and src[i] not in non_skip:
            lay_skip.append('<sk>')
            tgt_.append(src[i])
            skecth_train.append(prev_token)
        elif src[i].startswith('{H'):
            tokens = src[i].split('H')
            skecth_train.append(tokens[0])
            skecth_train.append(tokens[1])
            lay_skip.append(src[i])
            tgt_.append(PAD_WORD)
        else:
            nothing = None
            pre = src[i].replace('Ġ',"")
            skecth_train.append(src[i].replace('Ġ',""))
            if pre == 'V':
                prev_token = 'variable'
            elif pre == 'S':
                prev_token = 'subject'
            elif pre == 'O':
                prev_token = 'object'
            elif pre == 'T':
                prev_token = 'time'
            elif pre == 'P':
                prev_token = 'predicate'
            elif pre == 'ov' or pre =='sv' or pre == '}':
                nothing = pre
            else:
                print('something went wrong, pre is ',pre)
            lay_skip.append(src[i])
            tgt_.append(PAD_WORD)
    #print('skecth_train', skecth_train)
    return lay_skip, tgt_, skecth_train


def list2str(alist, type):
    #print('alist is ', alist)
    alist_list = []
    lay_test = []
    alist = str(alist).strip().replace("}",", }'").replace('"','').split("', ")

    if type == 'lay':
        for term in alist:
            term = term.replace("'","")
            if term.startswith("sv: {") or term.startswith("ov: {"):
            #if term.startswith("?x0: {") or term.startswith("?x1: {") or term.startswith("$y0: {") or term.startswith("$y1: {"):
                term = term.split(': ')
                alist_list.append(term[0].strip())
                alist_list.append(term[1].strip()+ term[2].strip())
                lay_test.append(term[0].strip())
                lay_test.append(term[1].strip() + term[2].strip())
            elif term == "}":
                alist_list.append(term)
                lay_test.append(term)
            else:
                if term.startswith('{H'):
                    lay_test.append(term.replace(': ', ''))
                else:
                    lay_test.append(term.split(': ')[0].strip())
                alist_list.append(term.replace(': ', ''))

    else:
        for term in alist:
            term = term.replace("'","")
            if term.strip().startswith("{H"):
                alist_list.append(term.replace(": ",""))
            #elif term.startswith("?x") or term.startswith("$y"):
            elif term.startswith("sv:") or term.startswith("ov:"):
                newterm = term.strip().split(': ')
                if len(newterm) == 3:
                    alist_list.append(newterm[0].strip())
                    alist_list.append(newterm[1].strip() + newterm[2].strip())
                elif len(newterm) == 2:
                    alist_list.append(newterm[0])
                    alist_list.append(newterm[1])
                else:
                    print("something went wrong")
            elif term == "}":
                alist_list.append(term)
            else:
                term = term.strip().split(": ")
                if len(term) < 2:
                    return None, None
                alist_list.append(term[0].strip())
                if term[1] == 'X,XX':
                    alist_list.append("X")
                    alist_list.append("XX")
                else:
                    words = term[1].strip().split(" ")
                    for word in words:
                        alist_list.append(word)
    #print('alist_list is ',alist_list)
    return alist_list, lay_test
def get_parent_index(tk_list):
    stack = [0]
    r_list = []
    for i, tk in enumerate(tk_list):
        r_list.append(stack[-1])
        if tk.startswith('{'):
            # +1: because the parent of the top level is 0
            stack.append(i+1)
        elif tk =='}':
            stack.pop()
    # for EOS (</s>)
    r_list.append(0)
    return r_list
def get_lay_index(lay_skip, data):
    length = len(data)
    # with a <s> token at the first position
    r_list = [0]
    k = 0
    for tk in lay_skip:
        if tk in (SKP_WORD, ):
            r_list.append(0)
        else:
            r_list.append(k)
            if k == length:
                print('not same length\n\n\n\n\n\n\n\n\n\n\n')
                #print('alist is {} \n tgt_ is {} \n length is {} \n lay is {} \n lay_skip is {} \n rlist {} \n'.format(data['alist'], data['tgt_'], length, data['lay'], lay_skip, r_list))
                print('lay skip is {}\n lay is {}\n'.format(lay_skip, data))
                return None
            k += 1
    #print('k is {} length is {}'.format(k, length))
    if k!=length:
        print('not same\n\n\n\n\n\n\n\n')
        return None
    return r_list

def get_lay_index_(lay_skip):
    r_list = [0]
    k = 0
    for tk in lay_skip:
        if tk in (SKP_WORD,):
            r_list.append(0)
        else:
            r_list.append(k)
            k+=1
    return r_list

def get_tgt_loss(line, mask_target_loss):
    r_list = []
    for tk_tgt, tk_lay_skip in zip(line['tgt_'], line['lay_skip']):
        if tk_lay_skip in (SKP_WORD,):
            r_list.append(tk_tgt)
        else:
            if mask_target_loss:
                r_list.append(PAD_WORD)
            else:
                r_list.append(tk_tgt)
    return r_list

def get_tgt_mask(lay_skip):
    # 0: use layout encoding vectors; 1: use target word embeddings;
    # with a <s> token at the first position
    return [1] + [1 if tk in (SKP_WORD,) else 0 for tk in lay_skip]

def get_tgt_not_copy(src, tgt):
    mask_list = []
    src_set = set(src)
    for tk_tgt in tgt:
        if tk_tgt in src_set:
            mask_list.append(UNK_WORD)
        else:
            if tk_tgt not in tgt_not_copy_list:
                #print('tk_tgt is {}\nsrc is {}\ntgt is {}\n'.format(tk_tgt, src, tgt))
                return None
            mask_list.append(tk_tgt)
    return mask_list

def get_copy_ext_wordpiece(src, wordpiece_index):
    #print('src is {} wordpiece is {}'.format(src, wordpiece_index))
    i = 0
    paded_src = []
    for wordpiece in wordpiece_index:
        if wordpiece:
            paded_src.append(PAD_WORD)
            pass
        else:
            paded_src.append(src[i])
            i+=1
    return paded_src


def create_lay_alist(alist, vocab):
    value2key = {}
    lay_alist = {}
    _alist = {}
    bart_alist = {}
    for key, value in alist.items():
        if isinstance(value, str):
            if key != 'h' and key != 'v':
                if key.startswith("?") or key.startswith("$"):
                    if key in value2key:
                        new_key = str(value2key[key]) + 'v'
                        lay_alist[new_key] = str((len(vocab.tokenizer('what ' + value.strip())['input_ids']) - 3))
                        _alist[new_key] = alist[key]
                    # new_key = str(value2key[key] + 'v')
                    # lay_alist[new_key] = str(len(value.strip().split(' ')))
                    #lay_alist[key] = str(len(value.strip().split(' ')))
                    else:
                        print('something went wrong ************************************************')
                else:
                    if alist[key].__contains__('?x') or alist[key].__contains__('$y'):
                        value2key[value] = key
                        if alist[key] == '?x0':
                            _alist[key.upper()] = 'X'
                            value = 'X'
                        elif alist[key] == '?x1':
                            _alist[key.upper()] = 'XX'
                            value = 'XX'
                        elif alist[key] == '$y0':
                            _alist[key.upper()] = 'A'
                            value = 'A'
                        elif alist[key] == '$y1':
                            _alist[key.upper()] = 'AA'
                            value = 'AA'
                    else:
                        _alist[key.upper()] = alist[key]
                    lay_alist[key.upper()] = str((len(vocab.tokenizer('what ' + value.strip())['input_ids']) - 3))
                bart_alist[key.upper()] = (' '.join(vocab.get_bart_token(alist[key])), vocab.tokenizer(alist[key])['input_ids'][1:-1])

            elif key == 'v':

                if alist[key] == '?x0':
                    _alist[key.upper()] = 'X'
                    lay_alist[key.upper()] = '1'

                elif alist[key] == '?x1':
                    _alist[key.upper()] = 'XX'
                    lay_alist[key.upper()] = '1'
                elif alist[key] == '$y0':
                    _alist[key.upper()] = 'A'
                    lay_alist[key.upper()] = '1'
                elif alist[key] == '$y1':
                    _alist[key.upper()] = 'AA'
                    lay_alist[key.upper()] = '1'
                elif alist[key] == '?x0,?x1':
                    _alist[key.upper()] = 'X,XX'
                    lay_alist[key.upper()] = '2'
                else:
                    print('went wrong ', alist[key])
                bart_alist[key.upper()] = (' '.join(vocab.get_bart_token(alist[key])), vocab.tokenizer(alist[key])['input_ids'][1:-1])
            else:
                lay_alist[key.upper()] = alist[key]
                _alist[key.upper()] = alist[key]
                bart_alist[key.upper()] = (' '.join(vocab.get_bart_token(alist[key])), vocab.tokenizer(alist[key])['input_ids'][1:-1])
        else:
            if key in value2key:
                new_key = str(value2key[key]) + 'v'
                lay_alist[new_key], _alist[new_key], bart_alist[new_key]= create_lay_alist(alist[key], vocab)
    return lay_alist, _alist, bart_alist
def get_bart(vocab, src, tgt_list):
    tgt = ' '.join(tgt_list)
    bart_src = vocab.tokenizer(tgt)['input_ids']
    bart_dec_inp = vocab.tokenizer(src)['input_ids']
    #print('src {} dec {} other {}'.format(bart_src, bart_dec_inp, vocab.tokenizer(tgt_list)['input_ids']))
    return bart_src, bart_dec_inp

def modify_data(data, bart_vocab):
    if data['alist'].__contains__('[') or data['alist'] == '{}':
        return None
    else:
        data['src_'] = data.pop('query').lower().replace("?","").replace("{","").replace("}","").replace("(max","").replace("(","").replace("(min","").replace(",", "").replace(")","").replace("<", "").replace(">","").replace("'s","").replace('"','').strip()
        data['bart_src'] = bart_vocab.tokenizer(data['src_'])['input_ids']
        data['bart_src_tokens'] = bart_vocab.tokenizer.convert_ids_to_tokens(data['bart_src'])
        data['src'] = data['src_'].split(' ')
        data['src'] = [token for token in data['src'] if token != ""]
        lay_alist, alist, bart_list = create_lay_alist(eval(data['alist']), bart_vocab)
        #print('lay alist {}\nalist {}\n bart list {}'.format(lay_alist, alist, bart_list))
        data['tgt_'], _ = list2str(alist, 'tgt')
        #print('src is {}\n tgt {}\n'.format(data['src'], data['tgt_']))
        data['lay'], data['lay_test'] = list2str(lay_alist, 'lay')
        #print(data['lay_test'])
        if data['tgt_'] == None or data['lay'] == None:
            return None
        #print('lay_test is ', data['lay_test'])
        data['bart_tgt_loss'] = bart_vocab.tokenizer(' '.join(data['tgt_']))['input_ids']
        data['tgt_tokens_'] = bart_vocab.tokenizer.convert_ids_to_tokens(data['bart_tgt_loss'])
        #print('alist is {}\nlay_alist is {}\ntgt_tokens {}'.format(data['alist'], lay_alist, data['tgt_tokens_']))
        assert len(data['bart_tgt_loss']) == len(data['tgt_tokens_'])
        #print('tgt_tokens {}'.format( data['tgt_tokens_']))
        #print('src', data['bart_src_tokens'])
        data['bart_lay_test'] = bart_vocab.tokenizer(' '.join(data['lay_test']))['input_ids'][1:-1]
        data['bart_lay_train'] = bart_vocab.tokenizer(' '.join(data['lay']))['input_ids']
        data['lay_test_tokens'] = bart_vocab.tokenizer.convert_ids_to_tokens(data['bart_lay_test'])
        data['lay_train_tokens'] = bart_vocab.tokenizer.convert_ids_to_tokens(data['bart_lay_train'])
        #print('lay_train',data['lay_train_tokens'])
        assert len(data['lay_train_tokens']) == len(data['bart_lay_train'])
        assert len(data['lay_test_tokens']) == len(data['bart_lay_test'])
        #print('bart_lay_train {}'.format(data['lay_train_tokens']))
        #print('bart_lay_test {}'.format(bart_vocab.get_bart_token(' '.join(data['lay_test']))))
        data['lay_skip'], data['tgt'], data['sketch_train_tokens'] = get_lay_skip(data['tgt_tokens_'][1:-1], data['lay_test_tokens'])
        data['sketch_train'] = bart_vocab.tokenizer(' '.join(data['sketch_train_tokens']))['input_ids']
        data['sketch_train_tokens'] = bart_vocab.tokenizer.convert_ids_to_tokens(data['sketch_train'])[1:-1]
        #print('sketch', data['sketch_train_tokens'])
        #print('sketch_tokens',data['sketch_train_tokens'])
        #print('sketch tokens {} \nconverted sketch tokens {}'.format(data['sketch_train_tokens'],bart_vocab.tokenizer.convert_ids_to_tokens(data['sketch_train'])))
        data['tgt'].insert(0, BOS_WORD)
        data['tgt'].append(EOS_WORD)
        data['bart_tgt'] = bart_vocab.tokenizer.convert_tokens_to_ids(data['tgt'])
        assert len(data['bart_tgt']) == len(data['tgt'])
        #print('tgt_loss {} \ntgt {}'.format(data['bart_tgt_loss'], data['bart_tgt']))
        #print('skip {}'.format(data['lay_skip']))
        data['lay_index'] = get_lay_index(data['lay_skip'], data['lay_test_tokens'])
        if data['lay_index'] == None:
            return None
        #data['lay_index'].append(0)
        data['tgt_mask'] = get_tgt_mask(data['lay_skip'])
        #data['tgt_mask'].append(1)
        # print('tgt mask {}\n bart_tgt {}\n tgt_loss {}\n'.format(data['tgt_mask'], data['tgt'], data['tgt_tokens_']))
        #assert len(data['tgt_mask']) == len(data['bart_tgt_loss'])
        #print(data['lay_index'])
        data['tgt_not_copy'] = get_tgt_not_copy(data['src'], data['tgt_'])
        if data['tgt_not_copy'] == None:
            return None
        data['attention_mask'] = [1 for token in data['bart_src']]
        #print('lay_skip is {}\n tgt_mask is {}\ntgt_loss is {} \n tgt is {} \n lay_index{} \n*********'.format(data['lay_skip'], data['tgt_mask'], data['tgt_'], data['tgt'], data['lay_index']))
        return data


def read_txt(path):
    datas = []
    bart_vocab = SrcVocab()
    with open(path) as f:
        lines = f.readlines()
        data = {}
        for line in lines:
            if line is not None:
                line = line.strip()
                if line.startswith('Q'):
                    data['query'] = line.split('Q:')[1].strip().lower()
                elif line.startswith('SPARQL'):
                    data['sparql'] = line.split('SPARQL:')[1].strip().lower()
                elif line.startswith('ALIST'):
                    data['alist'] = line.split('ALIST:')[1].strip().lower()
                elif line.startswith('TEMPLATE:'):
                    data['template'] = line.split('TEMPLATE: ')[1].strip()
                if 'query' in data and 'sparql' in data and 'alist' in data and 'template' in data:
                    data = modify_data(data, bart_vocab)
                    if data is not None:
                        datas.append(data)
                    data = {}
    return datas


def __getstate__(self):
    return dict(self.__dict__, stoi=dict(self.stoi))


def __setstate__(self, state):
    self.__dict__.update(state)
    self.stoi = defaultdict(lambda: 0, self.stoi)


torchtext.vocab.Vocab.__getstate__ = __getstate__
torchtext.vocab.Vocab.__setstate__ = __setstate__

def filter_counter(freqs, min_freq):
    cnt = Counter()
    for k, v in freqs.items():
        if (min_freq is None) or (v >= min_freq):
            cnt[k] = v
    return cnt

# def merge_vocabs(vocabs, min_freq=0, vocab_size=None):
#     """
#     Merge individual vocabularies (assumed to be generated from disjoint
#     documents) into a larger vocabulary.
#
#     Args:
#         vocabs: `torchtext.vocab.Vocab` vocabularies to be merged
#         vocab_size: `int` the final vocabulary size. `None` for no limit.
#     Return:
#         `torchtext.vocab.Vocab`
#     """
#     merged = Counter()
#     for vocab in vocabs:
#         merged += filter_counter(vocab.freqs, min_freq)
#     return torchtext.vocab.Vocab(merged,
#                                  specials=list(special_token_list),
#                                  max_size=vocab_size, min_freq=min_freq)


def _tgt_copy_ext(line):
    r_list = []
    mask_list = []
    src_set = set(line['src'])
    for tk_tgt in line['tgt_']:
        if tk_tgt in src_set:
            r_list.append(tk_tgt)
        else:
            r_list.append(UNK_WORD)

    return r_list

def _tgt_not_copy(line):
    mask_list = []
    src_set = set(line['src'])
    for tk_tgt in line['tgt_']:
        if tk_tgt in src_set:
            mask_list.append(UNK_WORD)
        else:
            if tk_tgt not in tgt_not_copy_list:
                print('tk_tgt is {}, src is {}, tgt is {}'.format(tk_tgt, line['src'], line['tgt_']))
            mask_list.append(tk_tgt)
    return mask_list

def join_dicts(*args):
    """
    args: dictionaries with disjoint keys
    returns: a single dictionary that has the union of these keys
    """
    return dict(chain(*[d.items() for d in args]))


class OrderedIterator(torchtext.data.Iterator):
    def create_batches(self):
        if self.train:
            self.batches = torchtext.data.pool(
                self.data(), self.batch_size,
                self.sort_key, self.batch_size_fn,
                random_shuffler=self.random_shuffler)
        else:
            self.batches = []
            for b in torchtext.data.batch(self.data(), self.batch_size,
                                          self.batch_size_fn):
                self.batches.append(sorted(b, key=self.sort_key))


class TableDataset(torchtext.data.Dataset):
    """Defines a dataset for machine translation."""

    @staticmethod
    def sort_key(ex):
        "Sort in reverse size order"
        return -len(ex.bart_src)

    def __init__(self, path, fields, opt, **kwargs):
        """
        Create a TranslationDataset given paths and fields.

        anno: location of annotated data / js_list
        filter_ex: False - keep all the examples for evaluation (should not have filtered examples); True - filter examples with unmatched spans;
        """
        if isinstance(path, str):
            datas = read_txt(path)
        else:
            datas = path

        bart_src_data = self._read_annotated_file(datas, 'bart_src')
        bart_src_examples = self._construct_examples(bart_src_data, 'bart_src')

        bart_tgt_data = self._read_annotated_file(datas, 'bart_tgt')
        bart_tgt_examples = self._construct_examples(bart_tgt_data, 'bart_tgt')

        bart_tgt_loss_data = self._read_annotated_file(datas, 'bart_tgt_loss')
        bart_tgt_loss_examples = self._construct_examples(bart_tgt_loss_data, 'bart_tgt_loss')

        attention_mask_data = self._read_annotated_file(datas, 'attention_mask')
        attention_mask_examples = self._construct_examples(attention_mask_data, 'attention_mask')

        bart_lay_train_data = self._read_annotated_file(datas, 'bart_lay_train')
        bart_lay_train_examples = self._construct_examples(bart_lay_train_data, 'bart_lay_train')

        bart_lay_test_data = self._read_annotated_file(datas, 'bart_lay_test')
        bart_lay_test_examples = self._construct_examples(bart_lay_test_data, 'bart_lay_test')

        lay_index_data = self._read_annotated_file(datas, 'lay_index')
        lay_index_examples = self._construct_examples(lay_index_data, 'lay_index')

        tgt_mask_data = self._read_annotated_file(datas, 'tgt_mask')
        tgt_mask_examples = self._construct_examples(tgt_mask_data, 'tgt_mask')

        sketch_train_data = self._read_annotated_file(datas, 'sketch_train')
        sketch_train_examples = self._construct_examples(sketch_train_data, 'sketch_train')



        # examples: one for each src line or (src, tgt) line pair.
        examples = [join_dicts(*it) for it in
                    zip(bart_src_examples,bart_tgt_examples, bart_tgt_loss_examples, attention_mask_examples,
                        bart_lay_train_examples, bart_lay_test_examples, lay_index_examples, tgt_mask_examples, sketch_train_examples)]
        # the examples should not contain None
        len_before_filter = len(examples)
        examples = list(filter(lambda x: all(
            (v is not None for k, v in x.items())), examples))
        len_after_filter = len(examples)
        num_filter = len_before_filter - len_after_filter
        #print("examples is ", [[example['bart_src'], example['bart_dec_inp']]for example in examples])
        #print('exmaples is ', examples)
        # Peek at the first to see which fields are used.
        ex = examples[0]
        keys = ex.keys()
        fields = [(k, fields[k])
                  for k in (list(keys) + ["indices"])]

        def construct_final(examples):
            for i, ex in enumerate(examples):
                s = torchtext.data.Example.fromlist(
                    [ex[k] for k in keys] + [i],
                    fields)
                #print("exmaple is {}, preprocessed is {}".format(examples[i]['tgt_loss'], s.tgt_loss))
                yield torchtext.data.Example.fromlist(
                    [ex[k] for k in keys] + [i],
                    fields)

        def filter_pred(example):
            return True

        super(TableDataset, self).__init__(
            construct_final(examples), fields, filter_pred)

    def _read_annotated_file(self, data_list, field):
        """
        path: location of a src or tgt file
        truncate: maximum sequence length (0 for unlimited)
        """
        if field in ('copy_to_tgt','copy_to_ext'):
            lines = (line['src'] for line in data_list)
        elif field in ('tgt_copy_ext',):
            lines = (_tgt_copy_ext(line) for line in data_list)
        elif field in ('tgt_not_copy',):
                #lines = (_tgt_not_copy(line) for line in data_list)
                lines = (line['tgt_not_copy'] for line in data_list)
        elif field in ('tgt_loss',):
            lines = (get_tgt_loss(line, False) for line in data_list)
        else:
            lines = (line[field] for line in data_list)
        for line in lines:
            yield line

    def _construct_examples(self, lines, side):
        for words in lines:
            #print('side is {}, words is {}'.format(side, words))
            example_dict = {side: words}
            yield example_dict

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, d):
        self.__dict__.update(d)

    def __reduce_ex__(self, proto):
        "This is a hack. Something is broken with torch pickle."
        return super(TableDataset, self).__reduce_ex__()

    @staticmethod
    def load_fields(vocab):
        vocab = dict(vocab)
        fields = TableDataset.get_fields()
        for k, v in vocab.items():
            # Hack. Can't pickle defaultdict :(
            v.stoi = defaultdict(lambda: 0, v.stoi)
            fields[k].vocab = v
        return fields

    @staticmethod
    def save_vocab(fields):
        vocab = []
        for k, f in fields.items():
            if 'vocab' in f.__dict__:
                f.vocab.stoi = dict(f.vocab.stoi)
                vocab.append((k, f.vocab))
        return vocab

    @staticmethod
    def get_fields():
        fields = {}
        fields["bart_src"] = torchtext.data.Field(
            use_vocab=False, pad_token=PAD)
        fields['bart_tgt'] = torchtext.data.Field(use_vocab=False, pad_token=PAD)
        fields['bart_tgt_loss'] = torchtext.data.Field(use_vocab=False, pad_token=PAD)
        fields["attention_mask"] = torchtext.data.Field(use_vocab=False, pad_token=0)
        fields["bart_lay_train"] = torchtext.data.Field(
            use_vocab=False, pad_token=PAD)
        fields['sketch_train'] = torchtext.data.Field(use_vocab = False, pad_token=PAD)
        fields['bart_lay_test'] = torchtext.data.Field(
            use_vocab=False, pad_token=PAD)
        fields['lay_index'] = torchtext.data.Field(
            use_vocab=False, pad_token=0)
        fields["tgt_mask"] = torchtext.data.Field(
            use_vocab=False, pad_token=0)
        fields["indices"] = torchtext.data.Field(
            use_vocab=False, sequential=False)
        return fields

    @staticmethod
    def build_vocab(train, dev, test, opt):
        print('ohho')

#if __name__ == "__main__":
    #print(load_template('/Users/liyanzhou/Desktop/Edinburgh/Dissertation/semantic_parsing/data_model/templates.txt'))