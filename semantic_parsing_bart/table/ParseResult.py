import json
from collections import defaultdict
def is_code_eq(t1, t2, not_layout=False):
    if t1 is None:
        return 0
    not_structure_list = ['Ġobject', 'Ġvariable', 'Ġsubject', 'Ġpredicate', 'Ġtime']
    # t1 = str(t1)
    # t2 = str(t2)
    # t1 = [it for it in t1.split(' ')]
    # t2 = [it for it in t2.split(' ')]
    #print('t1 is {} \nt2 is {}'.format(t1, t2))
    if not_layout:
        if len(t1) == len(t2):
            for tk1,tk2 in zip(t1,t2):
            # if not (tk1 == tk2 or tk1 == '<unk>' or tk2 == '<unk>'):
                if tk1.lower() != tk2.lower():
                    return False
            return True
        else: 
            return False
    else:
        if len(t1) == len(t2):
            for tk1, tk2 in zip(t1, t2):
                if tk2 not in not_structure_list:
                    if tk1.lower() != tk2.lower():
                        return False
            return True
        else:
            return False

    return t1==t2

def jaccard_similarity(tokens1, tokens2):
    if tokens1 is None:
        return 0
    #print('tokens 1{}\n, tokens2{}'.format(tokens1, tokens2))
    set1 = set([token.lower() for token in tokens1])
    set2 = set([token.lower() for token in tokens2])
    score = len(set1 & set2) / len(set1 | set2)
    #print(score)
    return score

class ParseResult(object):
    def __init__(self, idx, lay, tgt, token_prune):
        self.idx = idx
        self.lay = lay
        self.tgt = tgt
        self.token_prune = token_prune
        self.correct = defaultdict(lambda: 0)
        self.jaccard = defaultdict(lambda: 0)
        self.incorrect_prune = set()

    def eval(self, gold):
        if is_code_eq(self.lay, gold['sketch_train_tokens'], not_layout=False):
        #if is_code_eq(self.lay, gold['lay_train_tokens'][1:-1], not_layout=False):
            #print('lay 1')
            self.correct['lay'] = 1
        # else:
        #     print(' '.join(gold['src']))
        #     print('pred:', self.lay)
        #     print('gold:', gold['lay'])
        #     print('')

        if is_code_eq(self.tgt, gold['tgt_tokens_'][1:-1], not_layout=True):
            #print('tgt 1')
            self.correct['tgt'] = 1
        self.jaccard['lay'] = jaccard_similarity(self.lay, gold['sketch_train_tokens'])
        #self.jaccard['lay'] = jaccard_similarity(self.lay, gold['lay_train_tokens'][1:-1])
        self.jaccard['tgt'] = jaccard_similarity(self.tgt, gold['tgt_tokens_'][1:-1])

        # if self.correct['lay'] == 1 and self.correct['tgt'] == 1 and ('NUMBER' in self.lay and 'STRING' in self.lay and 'NAME' in self.lay):
        # if self.correct['lay'] == 1 and self.correct['tgt'] == 0:
        #     print(' '.join(gold['src']))
        #     print('pred_lay:', ' '.join(self.lay))
        #     print('gold_lay:', ' '.join(gold['lay']))
        #     print('pred_tgt:', ' '.join(self.tgt))
        #     print('gold_tgt:', ' '.join(gold['tgt']))
        #     print('')
