import os
import argparse
import torch
from path import Path

import table
import table.IO
import opts
from table.Utils import set_seed
from table.template_parser import Parser

parser = argparse.ArgumentParser(description='preprocess.py')


# **Preprocess Options**
parser.add_argument('-config', help="Read options from this file")

parser.add_argument('-src_vocab',
                    help="Path to an existing source vocabulary")
parser.add_argument('-tgt_vocab',
                    help="Path to an existing target vocabulary")
parser.add_argument('-seed', type=int, default=123,
                    help="Random seed")
parser.add_argument('-report_every', type=int, default=100000,
                    help="Report status every this many sentences")

opts.preprocess_opts(parser)

opt = parser.parse_args()
set_seed(opt.seed)

opt.train_anno = os.path.join(opt.root_dir, 'train.txt')
opt.valid_anno = os.path.join(opt.root_dir, 'test.txt')
opt.test_anno = os.path.join(opt.root_dir, 'test.txt')
opt.save_data = os.path.join(opt.root_dir)
def preprocess_alist(alist):
    if alist == None:
        return None
    modified = {}
    for term in alist:
        if term.startswith('$x') or term.startswith('?y'):
            if alist[term] != [] and alist[term] != None:
                if term.startswith('?y'):
                    modified['?x0'] = preprocess_alist(alist[term])
                elif term.startswith('$x'):
                    modified['$y0'] = preprocess_alist(alist[term])
        elif term.startswith('$filter'):
            if len(alist[term])>1:
                return None
            else:
                modified['?x0'] = preprocess_alist(alist[term][0])
        else:
            if alist[term].startswith('?y'):
                modified[term] = '?x0'
            elif alist[term].startswith('$x'):
                modified[term] = '$y0'
            else:
                modified[term] = alist[term]
    return modified

def jaccard_similarity(tokens1, tokens2):
    if tokens1 == None:
        return 0
    #print('tokens 1{}\n, tokens2{}'.format(tokens1, tokens2))
    set1 = set(tokens1)
    set2 = set(tokens2)
    score = len(set1 & set2) / len(set1 | set2)
    #print(score)
    return score
def strict_jaccard_similarity(tokens1, tokens2):
    if tokens1 == None:
        return 0
    #print('tokens 1{}\n, tokens2{}'.format(tokens1, tokens2))
    set1 = set(tokens1)
    set2 = set(tokens2)
    score = len(set1 & set2) / len(set1 | set2)
    if score == 1:
        return 1
    else:
        return 0


def eval(datas):
    parser = Parser()
    jaccard = 0.0
    strict_jaccard = 0.0
    i = 0
    for data in datas:
        alist = parser.find_templates(' '.join(data['src']))
        alist = preprocess_alist(alist['alist'])
        alist = table.IO.list2str(alist, 'tgt')[0]
        ground_truth = table.IO.list2str(data['alist'], 'tgt')[0]
        temp_jaccard = jaccard_similarity(alist, ground_truth)
        if temp_jaccard != 0:
            i+=1
        jaccard += temp_jaccard
        strict_jaccard += strict_jaccard_similarity(alist, ground_truth)
    if i == 0:
        return jaccard/len(datas), strict_jaccard/len(datas), 0, 0, 0
    return jaccard/len(datas), strict_jaccard/len(datas), jaccard/i, strict_jaccard/i, float(i)/float(len(datas))
def main():
    question = 'what is the population in UK'
    parser = Parser()
    alist = parser.find_templates(question)
    print(alist['alist'])
    # print('all')
    # datas = table.IO.read_txt(opt.test_anno)
    # jac_all, sjac_all, jac_sup, sjac_sup, prop = eval(datas)
    # print('jac_all{}, sjac_all{}, jac_sup{}, sjac_sup{}, translated data{}\n'.format(jac_all, sjac_all, jac_sup, sjac_sup, prop))
    # templates = table.IO.load_template(os.path.join(opt.root_dir, 'templates.txt'))
    # template_js_list = {}
    # for template in templates:
    #     template_js_list[template] = []
    #     for data in datas:
    #         if str(data['template']).strip() == template:
    #             template_js_list[template].append(data)
    # for template in templates:
    #     datas = template_js_list[template]
    #     if len(datas)!=0:
    #         print('template ', template)
    #         print('datas:', len(datas))
    #         jac_all, sjac_all, jac_sup, sjac_sup, prop = eval(datas)
    #         print('jac_all{}, sjac_all{}, jac_sup{}, sjac_sup{}, translated data{}\n'.format(jac_all, sjac_all, jac_sup,
    #                                                                                          sjac_sup, prop))




if __name__ == "__main__":
    main()
