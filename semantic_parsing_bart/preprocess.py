# -*- coding: utf-8 -*-
import os
import argparse
import torch
from path import Path

import table
import table.IO
import opts
from table.Utils import set_seed

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


def main():
    datas = table.IO.read_txt(opt.train_anno)

    print('Preparing training ...')
    fields = table.IO.TableDataset.get_fields()
    print("Building Training...")
    train = table.IO.TableDataset(
        opt.train_anno, fields, opt)
    print('train is ',train)

    if Path(opt.valid_anno).exists():
        print("Building Valid...")
        valid = table.IO.TableDataset(
            opt.valid_anno, fields, opt)
    else:
        print('valid is none')
        valid = None

    if Path(opt.test_anno).exists():
        print("Building Test...")
        test = table.IO.TableDataset(
            opt.test_anno, fields, opt)
    else:
        test = None

    print("Building Vocab...")
    table.IO.TableDataset.build_vocab(train, valid, test, opt)

    print("Saving train/valid/fields")
    # Can't save fields, so remove/reconstruct at training time.
    torch.save(table.IO.TableDataset.save_vocab(fields),
               open(os.path.join(opt.save_data, 'vocab.pt'), 'wb'))
    train.fields = []
    torch.save(train, open(os.path.join(opt.save_data, 'train.pt'), 'wb'))
    
    if Path(opt.valid_anno).exists():
        valid.fields = []
        torch.save(valid, open(os.path.join(opt.save_data, 'valid.pt'), 'wb'))
    
    if Path(opt.test_anno).exists():
        test.fields = []
        torch.save(test, open(os.path.join(opt.save_data, 'test.pt'), 'wb'))


if __name__ == "__main__":
    main()
