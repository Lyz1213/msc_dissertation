import json
from collections import defaultdict
def is_code_eq(t1, t2, not_layout=False):
    # t1 = str(t1)
    # t2 = str(t2)
    # t1 = [it for it in t1.split(' ')]
    # t2 = [it for it in t2.split(' ')]
    #print('t1 is {}, t2 is {}'.format(t1, t2))
    if len(t1) == len(t2):
        for tk1,tk2 in zip(t1,t2):
            # if not (tk1 == tk2 or tk1 == '<unk>' or tk2 == '<unk>'):
            if tk1 != tk2:
                return False
        return True
    else:
        return False
    return t1==t2

def jaccard_similarity(tokens1, tokens2):
    #print('tokens 1{}\n, tokens2{}'.format(tokens1, tokens2))
    set1 = set(tokens1)
    set2 = set(tokens2)
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
        self.incorrect_prune = set()
        self.jaccard = defaultdict(lambda: 0)

    def eval(self, gold):
        if is_code_eq(self.lay, gold['lay'], not_layout=False):
            #print('lay 1')
            self.correct['lay'] = 1
    
        # else:
        #     print(' '.join(gold['src']))
        #     print('pred:', self.lay)
        #     print('gold:', gold['lay'])
        #     print('')

        if is_code_eq(self.tgt, gold['tgt_'], not_layout=True):
            #print('tgt 1')
            self.correct['tgt'] = 1
        self.jaccard['lay'] = jaccard_similarity(self.lay, gold['lay'])
        self.jaccard['tgt'] = jaccard_similarity(self.tgt, gold['tgt_'])

        # if self.correct['lay'] == 1 and self.correct['tgt'] == 1 and ('NUMBER' in self.lay and 'STRING' in self.lay and 'NAME' in self.lay):
        # if self.correct['lay'] == 1 and self.correct['tgt'] == 0:
        #     print(' '.join(gold['src']))
        #     print('pred_lay:', ' '.join(self.lay))
        #     print('gold_lay:', ' '.join(gold['lay']))
        #     print('pred_tgt:', ' '.join(self.tgt))
        #     print('gold_tgt:', ' '.join(gold['tgt']))
        #     print('')
