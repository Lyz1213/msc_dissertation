import json
from collections import defaultdict
from nltk.translate.bleu_score import sentence_bleu

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
def bleu(candidate, reference):
    #print('reference:{}\ncandidate:{}'.format(reference, candidate))
    bleu = sentence_bleu(reference, candidate)
    bleu1 = sentence_bleu(reference, candidate, weights=(1, 0, 0, 0))
    bleu2 = sentence_bleu(reference, candidate, weights=(0.5, 0.5, 0, 0))
    bleu3 = sentence_bleu(reference, candidate, weights=(0.33, 0.33, 0.33, 0))
    bleu4 = sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25))
    return bleu, bleu1, bleu2, bleu3, bleu4
class ParseResult(object):
    def __init__(self, idx, lay, tgt, token_prune):
        self.idx = idx
        self.lay = lay
        self.tgt = tgt
        self.token_prune = token_prune
        self.correct = defaultdict(lambda: 0)
        self.jaccard = defaultdict(lambda: 0)
        self.bleu = defaultdict(lambda: 0)
        self.bleu1 = defaultdict(lambda: 0)
        self.bleu2 = defaultdict(lambda: 0)
        self.bleu3 = defaultdict(lambda: 0)
        self.bleu4 = defaultdict(lambda: 0)
        self.incorrect_prune = set()

    def eval(self, gold):
        if is_code_eq(self.lay, gold['src'], not_layout=False):
        #if is_code_eq(self.lay, gold['lay_train_tokens'][1:-1], not_layout=False):
            #print('lay 1')
            self.correct['lay'] = 1
        # else:
        #     print(' '.join(gold['src']))
        #     print('pred:', self.lay)
        #     print('gold:', gold['lay'])
        #     print('')

        if is_code_eq(self.tgt, gold['tgt'][1:-1], not_layout=True):
            #print('tgt 1')
            self.correct['tgt'] = 1
        #self.jaccard['lay'] = jaccard_similarity(self.lay, gold['tgt'][1:-1])
        #self.jaccard['lay'] = jaccard_similarity(self.lay, gold['lay_train_tokens'][1:-1])
        #self.jaccard['tgt'] = jaccard_similarity(self.tgt, gold['src'])
        self.bleu['tgt'], self.bleu1['tgt'], self.bleu2['tgt'], self.bleu3['tgt'], self.bleu4['tgt'] = bleu(self.tgt, [gold['tgt'][1:-1]])
        # if self.correct['lay'] == 1 and self.correct['tgt'] == 1 and ('NUMBER' in self.lay and 'STRING' in self.lay and 'NAME' in self.lay):
        # if self.correct['lay'] == 1 and self.correct['tgt'] == 0:
        #     print(' '.join(gold['src']))
        #     print('pred_lay:', ' '.join(self.lay))
        #     print('gold_lay:', ' '.join(gold['lay']))
        #     print('pred_tgt:', ' '.join(self.tgt))
        #     print('gold_tgt:', ' '.join(gold['tgt']))
        #     print('')
