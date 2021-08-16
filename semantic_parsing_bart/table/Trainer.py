"""
This is the loadable seq2seq trainer library that is
in charge of training details, loss compute, and statistics.
"""
from __future__ import division
import os
import time
import sys
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
from copy import deepcopy

import table
import table.modules
from table.Utils import argmax
from table.Tokenize import SrcVocab


class Statistics(object):
    def __init__(self, loss, eval_result):
        self.loss = loss
        self.eval_result = eval_result
        self.start_time = time.time()

    def update(self, stat):
        self.loss += stat.loss
        for k, v in stat.eval_result.items():
            if k in self.eval_result:
                v0 = self.eval_result[k][0] + v[0]
                v1 = self.eval_result[k][1] + v[1]
                self.eval_result[k] = (v0, v1)
            else:
                self.eval_result[k] = (v[0], v[1])

    def accuracy(self, return_str=False):
        d = sorted([(k, v)
                    for k, v in self.eval_result.items()], key=lambda x: x[0])
        if return_str:
            return '; '.join((('{}: {:.2%}'.format(k, float(v[0]) / float(v[1]),)) for k, v in d))
        else:
            return dict([(k, 100.0 * v[0] / v[1]) for k, v in d])

    def elapsed_time(self):
        return time.time() - self.start_time

    def output(self, epoch, batch, n_batches, start):
        print(("Epoch %2d, %5d/%5d; %s; %.0f s elapsed") %
              (epoch, batch, n_batches, self.accuracy(True), time.time() - start))
        sys.stdout.flush()

    def log(self, split, logger, lr, step):
        pass


def count_accuracy(scores, target, mask=None, row=False):
    pred = argmax(scores)
    #print('target size {} pred size {}'.format(target.size(), pred.size()))
    if mask is None:
        m_correct = pred.eq(target)
        num_all = m_correct.numel()
    elif row:
        m_correct = pred.eq(target).masked_fill_(
            mask, 1).prod(0, keepdim=False)
        num_all = m_correct.numel()
    else:
        non_mask = mask.ne(1)
        m_correct = pred.eq(target).masked_select(non_mask)
        num_all = non_mask.sum()
    return (m_correct, num_all)


def count_token_prune_accuracy(scores, target, _mask, row=False):
    # 0 -> 0.5 by sigmoid
    pred = scores.gt(0).long()
    target = target.long()
    mask = torch.ByteTensor(_mask).cuda().unsqueeze(1).expand_as(target)
    if row:
        m_correct = pred.eq(target).masked_fill_(
            mask, 1).prod(0, keepdim=False)
        num_all = m_correct.numel()
    else:
        non_mask = mask.ne(1)
        m_correct = pred.eq(target).masked_select(non_mask)
        num_all = non_mask.sum()
    return (m_correct, num_all)


def aggregate_accuracy(r_dict, metric_name_list):
    m_list = []
    for metric_name in metric_name_list:
        m_list.append(r_dict[metric_name][0])
    agg = torch.stack(m_list, 0).prod(0, keepdim=False)
    return (agg.sum(), agg.numel())


def _debug_batch_content(vocab, ts_batch, gold_batch):
    batch_size = ts_batch.size(1)
    for b in range(batch_size):
        pred = vocab.tokenizer.convert_ids_to_tokens(ts_batch[:,b])
        gold = vocab.tokenizer.convert_ids_to_tokens(gold_batch[:,b])
        print('lay gold {} \nlay pred {}\n'.format(gold, pred))

def _debug_batch_content_(vocab, ts_batch,ge_batch, gold_batch):
    batch_size = ts_batch.size(1)
    for b in range(batch_size):
        pred = vocab.tokenizer.convert_ids_to_tokens(ts_batch[:,b])
        generated = vocab.tokenizer.convert_ids_to_tokens(ge_batch[:,b])
        gold = vocab.tokenizer.convert_ids_to_tokens(gold_batch[:,b])
        print('lay gold {} \nlay pred {}\nlay gene {}\n'.format(gold, pred, generated))
def _debug_seq(vocab, seq):
    return [vocab.itos[token] for token in seq]

def _debug_batch_content_tgt(vocabtgt, vocabext, ts_batch, gold_batch):
    seq_len = ts_batch.size(0)
    batch_size = ts_batch.size(1)
    for b in range(batch_size):
        tk_list = []
        gold_list = []
        for i in range(seq_len):
            if gold_batch[i,b] < len(vocabtgt):
                gtk = vocabtgt.itos[gold_batch[i,b]]
            else:
                gtk = vocabext.itos[gold_batch[i,b]-len(vocabtgt)]
            gold_list.append(gtk)
            if ts_batch[i,b] < len(vocabtgt):
                tk = vocabtgt.itos[ts_batch[i,b]]
            else:
                tk = vocabext.itos[ts_batch[i,b]-len(vocabtgt)]
            tk_list.append(tk)
        print('tgt gold {} \n tgt pred {}'.format(gold_list, tk_list))


class Trainer(object):
    def __init__(self, model, train_iter, valid_iter,
                 train_loss, valid_loss, optim):
        """
        Args:
            model: the seq2seq model.
            train_iter: the train data iterator.
            valid_iter: the validate data iterator.
            train_loss: the train side LossCompute object for computing loss.
            valid_loss: the valid side LossCompute object for computing loss.
            optim: the optimizer responsible for lr update.
        """
        # Basic attributes.
        self.model = model
        self.train_iter = train_iter
        self.valid_iter = valid_iter
        self.train_loss = train_loss
        self.valid_loss = valid_loss
        self.optim = optim
        self.srcvocab = SrcVocab()

        if self.model.opt.moving_avg > 0:
            self.moving_avg = deepcopy(
                list(p.data for p in model.parameters()))
        else:
            self.moving_avg = None

        # Set model in training mode.
        self.model.train()

    def forward(self, epoch, batch, criterion, fields):
        # 1. F-prop.


        #print('q_batch is ', q_batch)
        out, sketch = self.model(batch.bart_src, batch.bart_tgt, batch.bart_lay_train, batch.bart_lay_test, batch.lay_index, batch.tgt_mask, batch.attention_mask, batch.bart_tgt_loss, batch.sketch_train)
        if self.model.opt.model_type == 'onestage':
            # 2. Compute loss.
            pred = {'tgt': out}
            gold = {}
            mask_loss = {}
            gold['tgt'] = batch.bart_tgt_loss[1:]
            #gold['tgt'] = batch.sketch_train[1:]
            #_debug_batch_content_(self.srcvocab, argmax(out), sketch, gold['tgt'])
            #_debug_batch_content(self.srcvocab, argmax(out), gold['tgt'])

            # if self.model.opt.coverage_loss > 0 and epoch > 10:
            #     gold['cover'] = loss_coverage * self.model.opt.coverage_loss

            loss = criterion.compute_loss(pred, gold, mask_loss)

            # 3. Get the batch statistics.
            r_dict = {}
            for metric_name in ('tgt',):
                p = pred[metric_name].data
                g = gold[metric_name].data
                # token accuracy
                r_dict[metric_name + '-token'] = count_accuracy(
                    p, g, mask=g.eq(1), row=False)
                # sentence accuracy
                r_dict[metric_name] = count_accuracy(
                    p, g, mask=g.eq(1), row=True)
            st = dict([(k, (v[0].sum(), v[1])) for k, v in r_dict.items()])
            st['all'] = aggregate_accuracy(r_dict, ('tgt',))
            # aggregate the accrucy of lay and tgt: 1 iff lay and tgt are both 1, else 0
            if self.model.opt.coverage_loss > 0 and epoch > 10:
                st['attn_impor_loss'] = (gold['cover'].data[0], 1)
            batch_stats = Statistics(loss.data, st)

            return loss, batch_stats
        else:
            sketch_ge = sketch[1]
            sketch = sketch[0]
            #print('sketch size {} alist size {}'.format(sketch.size(), out.size()))
            pred = {'lay': sketch, 'tgt': out, 'token': None}
            gold = {}
            mask_loss = {}
            gold['lay'] = batch.bart_lay_train[1:]
            #_debug_batch_content_(self.srcvocab, argmax(sketch), sketch_ge, gold['lay'])
            # tgt_copy_mask = batch.tgt_copy_ext.ne(
            #     fields['tgt_copy_ext'].vocab.stoi[table.IO.UNK_WORD]).long()[1:]
            # tgt_org_mask = batch.tgt_copy_ext.eq(
            #     fields['tgt_copy_ext'].vocab.stoi[table.IO.UNK_WORD]).long()[1:]
            # gold['tgt'] = torch.mul(tgt_copy_mask, batch.tgt_copy_ext[1:] + len(
            #     fields['tgt_not_copy'].vocab)) + torch.mul(tgt_org_mask, batch.tgt_loss[1:])
            # _debug_batch_content_tgt(fields['tgt_not_copy'].vocab, fields['copy_to_ext'].vocab, argmax(tgt_out.data), gold['tgt'])
            # if self.model.opt.coverage_loss > 0 and epoch > 10:
            #     gold['cover'] = loss_coverage * self.model.opt.coverage_loss
            gold['tgt'] = batch.bart_tgt_loss[1:]
            #_debug_batch_content(self.srcvocab, argmax(out), gold['tgt'])
            #print('gold lay {} gold tgt {}'.format(gold['lay'].size(), gold['tgt'].size()))
            #_debug_batch_content(self.srcvocab, argmax(out), gold['tgt'])

            loss = criterion.compute_loss(pred, gold, mask_loss)

            # 3. Get the batch statistics.
            r_dict = {}
            for metric_name in ('lay', 'tgt'):
                p = pred[metric_name].data
                g = gold[metric_name].data
                # token accuracy
                r_dict[metric_name + '-token'] = count_accuracy(
                    p, g, mask=g.eq(1), row=False)
                # sentence accuracy
                r_dict[metric_name] = count_accuracy(
                    p, g, mask=g.eq(1), row=True)
            st = dict([(k, (v[0].sum(), v[1])) for k, v in r_dict.items()])
            st['all'] = aggregate_accuracy(r_dict, ('lay', 'tgt'))
            # aggregate the accrucy of lay and tgt: 1 iff lay and tgt are both 1, else 0
            if self.model.opt.coverage_loss > 0 and epoch > 10:
                st['attn_impor_loss'] = (gold['cover'].data[0], 1)
            batch_stats = Statistics(loss.data, st)

            return loss, batch_stats


    def train(self, epoch, fields, report_func=None):
        """ Called for each epoch to train. """
        total_stats = Statistics(0, {})
        report_stats = Statistics(0, {})

        for i, batch in enumerate(self.train_iter):
            self.model.zero_grad()
            loss, batch_stats = self.forward(
                epoch, batch, self.train_loss, fields)

            # _debug_batch_content(fields['lay'].vocab, batch.lay.data)

            # Update the parameters and statistics.
            loss.backward()
            self.optim.step()
            total_stats.update(batch_stats)
            report_stats.update(batch_stats)

            if report_func is not None:
                report_stats = report_func(
                    epoch, i, len(self.train_iter),
                    total_stats.start_time, self.optim.lr, report_stats)

            if self.model.opt.moving_avg > 0:
                decay_rate = min(self.model.opt.moving_avg,
                                 (1 + epoch) / (1.5 + epoch))
                for p, avg_p in zip(self.model.parameters(), self.moving_avg):
                    avg_p.mul_(decay_rate).add_(1.0 - decay_rate, p.data)

        return total_stats

    def validate(self, epoch, fields):
        """ Called for each epoch to validate. """
        # Set model in validating mode.
        torch.cuda
        with torch.no_grad():
            self.model.eval()

            stats = Statistics(0, {})
            for batch in self.valid_iter:
                loss, batch_stats = self.forward(
                epoch, batch, self.valid_loss, fields)

            # Update statistics.
            stats.update(batch_stats)

        # Set model back to training mode.
        self.model.train()

        return stats

    def epoch_step(self, eval_metric, epoch):
        """ Called for each epoch to update learning rate. """
        return self.optim.updateLearningRate(eval_metric, epoch)

    def drop_checkpoint(self, opt, epoch, fields, valid_stats):
        """ Called conditionally each epoch to save a snapshot. """

        model_state_dict = self.model.state_dict()
        model_state_dict = {k: v for k, v in model_state_dict.items()
                            if 'generator' not in k}
        checkpoint = {
            'model': model_state_dict,
            'vocab': table.IO.TableDataset.save_vocab(fields),
            'opt': opt,
            'epoch': epoch,
            'optim': self.optim,
            'moving_avg': self.moving_avg
        }
        for key in checkpoint:
            print('key is ',key)
            print('value is ', checkpoint[key])
        eval_result = valid_stats.accuracy()
        torch.save(checkpoint, os.path.join(
            opt.save_path, 'm_%d.pt' % (epoch)))
