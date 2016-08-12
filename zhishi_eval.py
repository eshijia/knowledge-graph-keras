# -*- coding: utf-8 -*-

from __future__ import print_function
import os
import sys
import random
from time import strftime, gmtime
import cPickle as pickle
from keras.optimizers import RMSprop, Adam
from scipy.stats import rankdata
from keras.utils import generic_utils
from keras_models import *
from heapq import nlargest
import codecs

random.seed(42)
os.environ['ZHISHI'] = 'data/zhishi'


class Evaluator:
    def __init__(self, conf=None):
        try:
            data_path = os.environ['ZHISHI']
        except KeyError:
            print("ZHISHI is not set.")
            sys.exit(1)
        self.path = data_path
        self.conf = dict() if conf is None else conf
        self.params = conf.get('training_params', dict())
        self.entity = self.load('zhishi-id2entity.pkl')
        self._vocab = None
        self._reverse_vocab = None
        self._eval_sets = None

    ##### Resources #####

    def load(self, name):
        return pickle.load(open(os.path.join(self.path, name), 'rb'))

    def vocab(self):
        if self._vocab is None:
            self._vocab = self.load('vocabulary')
        return self._vocab

    def reverse_vocab(self):
        if self._reverse_vocab is None:
            vocab = self.vocab()
            self._reverse_vocab = dict((v.lower(), k) for k, v in vocab.items())
        return self._reverse_vocab

    ##### Loading / saving #####

    def save_epoch(self, model, epoch):
        if not os.path.exists('models/zhishi_models/embedding/'):
            os.makedirs('models/zhishi_models/embedding/')
        model.save_weights('models/zhishi_models/embedding/weights_epoch_%d.h5' % epoch, overwrite=True)

    def save_epoch_rt(self, model, epoch):
        if not os.path.exists('models/zhishi_models/embedding_rt/'):
            os.makedirs('models/zhishi_models/embedding_rt/')
        model.save_weights('models/zhishi_models/embedding_rt/weights_epoch_%d.h5' % epoch, overwrite=True)

    def load_epoch(self, model, epoch):
        assert os.path.exists('models/zhishi_models/embedding/weights_epoch_%d.h5' % epoch),\
            'Weights at epoch %d not found' % epoch
        model.load_weights('models/zhishi_models/embedding/weights_epoch_%d.h5' % epoch)

    def load_epoch_rt(self, model, epoch):
        assert os.path.exists('models/zhishi_models/embedding_rt/weights_epoch_%d.h5' % epoch),\
            'Weights at epoch %d not found' % epoch
        model.load_weights('models/zhishi_models/embedding_rt/weights_epoch_%d.h5' % epoch)


    ##### Converting / reverting #####

    def convert(self, words):
        rvocab = self.reverse_vocab()
        if type(words) == str:
            words = words.strip().lower().split(' ')
        return [rvocab.get(w, 0) for w in words]

    def revert(self, indices):
        vocab = self.vocab()
        return [vocab.get(i, 'X') for i in indices]

    ##### Padding #####

    def padq(self, data):
        return self.pad(data, self.conf.get('question_len', None))

    def pada(self, data):
        return self.pad(data, self.conf.get('answer_len', None))

    def pad(self, data, len=None):
        from keras.preprocessing.sequence import pad_sequences
        return pad_sequences(data, maxlen=len, padding='post', truncating='post', value=0)

    ##### Training #####

    def print_time(self):
        print(strftime('%Y-%m-%d %H:%M:%S :: ', gmtime()), end='')

    def train(self, model):
        eval_every = self.params.get('eval_every', None)
        save_every = self.params.get('save_every', None)
        batch_size = self.params.get('batch_size', 128)
        nb_epoch = self.params.get('nb_epoch', 10)

        print("Load train ...")
        training_set = self.load('zhishi-train.pkl')
        print("Load entity ...")
        entity_candidates = self.entity.keys()
        subjects = list()
        relations = list()
        good_objects = list()

        print("Load triplets ...")
        for line in training_set:
            triplet = line.split('\t')
            subjects += [[int(triplet[0])]]
            relations += [[int(triplet[1])]]
            good_objects += [[int(triplet[2])]]

        subjects = np.asarray(subjects)
        relations = np.asarray(relations)
        good_objects = np.asarray(good_objects)

        for i in range(1, nb_epoch+1):
            print('-' * 40)
            print('Epoch', i)
            print('-' * 40)
            progbar = generic_utils.Progbar(len(training_set))
            nb_batch = int(len(training_set) / batch_size)
            for batch_id in xrange(nb_batch):
                begin_index = batch_id * batch_size
                end_index = len(training_set) if (batch_id + 1) * batch_size > len(training_set) else (batch_id + 1) * batch_size
                subjects_batch = subjects[begin_index: end_index]
                relations_batch = relations[begin_index: end_index]
                good_objects_batch = good_objects[begin_index: end_index]
                bad_objects_batch = np.asarray([[int(random.choice(entity_candidates))]
                                                for _ in xrange(end_index - begin_index)])

                loss = model.train_on_batch([subjects_batch, relations_batch, good_objects_batch, bad_objects_batch])
                progbar.add(subjects_batch.shape[0], values=[('train loss', loss)])

            if eval_every is not None and i % eval_every == 0:
                self.get_mrr(model)

            if save_every is not None and i % save_every == 0:
                self.save_epoch(model, i)
        print("Training completed!")

    def train_rt(self, model):
        eval_every = self.params.get('eval_every', None)
        save_every = self.params.get('save_every', None)
        batch_size = self.params.get('batch_size', 128)
        nb_epoch = self.params.get('nb_epoch', 10)

        print("Load train ...")
        training_set = self.load('zhishi-train.pkl')
        print("Load entity ...")
        entity_candidates = self.entity.keys()
        subjects = list()
        relations = list()
        good_objects = list()

        print("Load triplets ...")
        for line in training_set:
            triplet = line.split('\t')
            subjects += [[int(triplet[0])]]
            relations += [[int(triplet[1])]]
            good_objects += [[int(triplet[2])]]

        subjects = np.asarray(subjects)
        relations = np.asarray(relations)
        good_objects = np.asarray(good_objects)

        for i in range(1, nb_epoch+1):
            print('-' * 40)
            print('Epoch', i)
            print('-' * 40)
            progbar = generic_utils.Progbar(len(training_set))
            nb_batch = int(len(training_set) / batch_size)
            for batch_id in xrange(nb_batch):
                begin_index = batch_id * batch_size
                end_index = len(training_set) if (batch_id + 1) * batch_size > len(training_set) else (batch_id + 1) * batch_size
                subjects_batch = subjects[begin_index: end_index]
                bad_subjects_batch = np.asarray([[int(random.choice(entity_candidates))]
                                                for _ in xrange(end_index - begin_index)])
                relations_batch = relations[begin_index: end_index]
                good_objects_batch = good_objects[begin_index: end_index]

                loss = model.train_on_batch_rt([subjects_batch, bad_subjects_batch, relations_batch, good_objects_batch])
                progbar.add(subjects_batch.shape[0], values=[('train loss', loss)])

            if eval_every is not None and i % eval_every == 0:
                self.get_mrr(model)

            if save_every is not None and i % save_every == 0:
                self.save_epoch_rt(model, i)
        print("Training completed!")

    ##### Evaluation #####

    def prog_bar(self, so_far, total, n_bars=20):
        n_complete = int(so_far * n_bars / total)
        if n_complete >= n_bars - 1:
            print('\r[' + '=' * n_bars + ']', end='')
        else:
            s = '\r[' + '=' * (n_complete - 1) + '>' + '.' * (n_bars - n_complete) + ']'
            print(s, end='')

    def eval_sets(self):
        if self._eval_sets is None:
            self._eval_sets = dict([(s, self.load(s)) for s in ['zhishi_hr-test.pkl']])
        return self._eval_sets

    def eval_sets_rt(self):
        if self._eval_sets is None:
            self._eval_sets = dict([(s, self.load(s)) for s in ['zhishi_rt-test.pkl']])
        return self._eval_sets

    def eval_sets_tc(self):
        if self._eval_sets is None:
            self._eval_sets = dict([(s, self.load(s)) for s in ['zhishi_tc-test.pkl']])
        return self._eval_sets

    def make_submit(self, model, submit_file):
        data = self.eval_sets().values()[0]
        target_lines = list()
        for i, d in enumerate(data):
            num_candidate = len(self.entity)
            index_entities = xrange(num_candidate)

            terms = d.split('\t')
            answers = [[idx] for idx in self.entity.keys()]
            subjects = [[terms[0]]] * num_candidate
            relations = [[terms[1]]] * num_candidate

            sims = model.predict([subjects, relations, answers], batch_size=num_candidate).flatten()
            print(len(sims))
            r = rankdata(sims, method='ordinal')
            index_candidates = nlargest(200, index_entities, key=lambda j: r[j])
            one_line = ' '.join(index_candidates)
            target_lines.append(one_line + '\n')
        submit_file.writelines(target_lines)

    def make_submit_rt(self, model, submit_file):
        data = self.eval_sets_rt().values()[0]
        target_lines = list()
        for i, d in enumerate(data):
            num_candidate = len(self.entity)
            index_entities = xrange(num_candidate)

            terms = d.split('\t')
            answers = [[idx] for idx in self.entity.keys()]
            relations = [[terms[0]]] * num_candidate
            objects = [[terms[1]]] * num_candidate

            sims = model.predict_rt([answers, relations, objects], batch_size=num_candidate).flatten()
            print(len(sims))
            r = rankdata(sims, method='ordinal')
            index_candidates = nlargest(200, index_entities, key=lambda j: r[j])
            one_line = ' '.join(index_candidates)
            target_lines.append(one_line + '\n')
        submit_file.writelines(target_lines)

    def make_submit_tc(self, model, submit_file):
        data = self.eval_sets_tc().values()[0]
        target_lines = list()
        for i, d in enumerate(data):

            terms = d.split('\t')
            subjects = [[terms[0]]]
            relations = [[terms[1]]]
            objects = [[terms[2]]]

            sims = model.predict([subjects, relations, objects], batch_size=1).flatten()
            print(len(sims))
            if sims[0] >= 0.55:
                target_lines.append(1)
            else:
                target_lines.append(0)
        submit_file.writelines(' '.join(target_lines))

    def get_mrr(self, model, evaluate_all=False):
        top1s = list()
        mrrs = list()
        for name, data in self.eval_sets().items():
            if evaluate_all:
                self.print_time()
                print('----- %s -----' % name)

            random.shuffle(data)

            if not evaluate_all and 'n_eval' in self.params:
                data = data[:self.params['n_eval']]

            # c_1 for hit@1, c_3 for hit@3, c_10 for hit@10
            c_1, c_3, c_10 = 0, 0, 0
            mean_ranks = list()

            for i, d in enumerate(data):
                triplet = d.split('\t')
                if evaluate_all:
                    self.prog_bar(i, len(data))

                candidate_objects = self.entity.keys()
                candidate_objects.remove(int(triplet[2]))

                subject = np.asarray([[int(triplet[0])]] * (len(candidate_objects)+1))
                relation = np.asarray([[int(triplet[1])]] * (len(candidate_objects)+1))
                objects = np.asarray([[int(triplet[2])]] + [[entity_id] for entity_id in candidate_objects])
                sims = model.predict([subject, relation, objects], batch_size=len(self.entity)).flatten()
                r = rankdata(sims, method='max')

                target_rank = r[0]
                num_candidate = len(sims)
                real_rank = num_candidate - target_rank + 1

                # print(' '.join(self.revert(d['question'])))
                # print(' '.join(self.revert(self.answers[indices[max_r]])))
                # print(' '.join(self.revert(self.answers[indices[max_n]])))

                c_1 += 1 if target_rank == num_candidate else 0
                c_3 += 1 if target_rank + 3 > num_candidate else 0
                c_10 += 1 if target_rank + 10 > num_candidate else 0
                mean_ranks.append(real_rank)
                # c_2 += 1 / float(r[max_r] - r[max_n] + 1)

            hit_at_1 = c_1 / float(len(data))
            hit_at_3 = c_3 / float(len(data))
            hit_at_10 = c_10 / float(len(data))
            avg_rank = np.mean(mean_ranks)

            del data

            if evaluate_all:
                print('Hit@1 Precision: %f' % hit_at_1)
                print('Hit@3 Precision: %f' % hit_at_3)
                print('Hit@10 Precision: %f' % hit_at_10)
                print('Mean Rank: %f' % avg_rank)

            # top1s.append(top1)
            # mrrs.append(mrr)

        # rerun the evaluation if above some threshold
        if not evaluate_all:
            print('Top-1 Precision: {}'.format(top1s))
            print('MRR: {}'.format(mrrs))
            evaluate_all_threshold = self.params.get('evaluate_all_threshold', dict())
            evaluate_mode = evaluate_all_threshold.get('mode', 'all')
            mrr_theshold = evaluate_all_threshold.get('mrr', 1)
            top1_threshold = evaluate_all_threshold.get('top1', 1)

            if evaluate_mode == 'any':
                evaluate_all = evaluate_all or any([x >= top1_threshold for x in top1s])
                evaluate_all = evaluate_all or any([x >= mrr_theshold for x in mrrs])
            else:
                evaluate_all = evaluate_all or all([x >= top1_threshold for x in top1s])
                evaluate_all = evaluate_all or all([x >= mrr_theshold for x in mrrs])

            if evaluate_all:
                return self.get_mrr(model, evaluate_all=True)

if __name__ == '__main__':
    conf = {
        'subject_len': 1,
        'relation_len': 1,
        'object_len': 1,
        'n_words': 648211,  # len(vocabulary) + 1
        'margin': 0.05,

        'training_params': {
            'save_every': 100,
            # 'eval_every': 1,
            'batch_size': 512,
            'nb_epoch': 200,
            'validation_split': 0,
            'optimizer': 'adam',
            # 'optimizer': Adam(clip_norm=0.1),
            # 'n_eval': 100,

            'evaluate_all_threshold': {
                'mode': 'all',
                'top1': 0.4,
            },
        },

        'model_params': {
            'n_embed_dims': 100,
            'n_hidden': 200,

            # convolution
            'nb_filters': 1000, # * 4
            'conv_activation': 'relu',

            # recurrent
            'n_lstm_dims': 141, # * 2

            # 'initial_embed_weights': np.load('word2vec_100_dim.embeddings'),
        },

        'similarity_params': {
            'mode': 'cosine',
            'gamma': 1,
            'c': 1,
            'd': 2,
        }
    }

    evaluator = Evaluator(conf)

    ##### Define model ######
    # model = EmbeddingModel(conf)
    # optimizer = conf.get('training_params', dict()).get('optimizer', 'adam')
    # model.compile(optimizer=optimizer)

    model = EmbeddingModelRt(conf)
    optimizer = conf.get('training_params', dict()).get('optimizer', 'adam')
    model.compile_rt(optimizer=optimizer)

    import numpy as np

    # save embedding layer
    # evaluator.load_epoch(model, 33)
    # embedding_layer = model.prediction_model.layers[2].layers[2]
    # evaluator.load_epoch(model, 100)
    # evaluator.train(model)
    # weights = embedding_layer.get_weights()[0]
    # np.save(open('models/embedding_1000_dim.h5', 'wb'), weights)

    # model for link prediction -> tail
    evaluator.load_epoch(model, 200)
    # evaluator.train(model)
    lp_t = codecs.open('lp_t.txt', 'wb')
    evaluator.make_submit(model, lp_t)

    # model for link prediction -> head
    # evaluator.load_epoch_rt(model, 200)
    # evaluator.train_rt(model)
    # lp_h = codecs.open('lp_h.txt', 'wb')
    # evaluator.make_submit_rt(model, lp_h)

    # model for triplet classification
    # evaluator.load_epoch(model, 200)
    # evaluator.train(model)
    # tc = codecs.open('tc.txt', 'wb')
    # evaluator.make_submit_tc(model, tc)

    # evaluate mrr for a particular epoch
    # evaluator.load_epoch(model, 200)
    # evaluator.get_mrr(model, evaluate_all=True)
