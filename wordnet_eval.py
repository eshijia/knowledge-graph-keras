# -*- coding: utf-8 -*-

from __future__ import print_function
import os
import sys
import random
from time import strftime, gmtime
import cPickle as pickle
from keras.optimizers import RMSprop, Adam, SGD, Adadelta, Adagrad
from scipy.stats import rankdata
from keras_models import *

random.seed(42)
os.environ['WORDNET'] = 'data/wordnet'


class Evaluator:
    def __init__(self, conf=None):
        try:
            data_path = os.environ['WORDNET']
        except KeyError:
            print("WORDNET is not set.")
            sys.exit(1)
        self.path = data_path
        self.conf = dict() if conf is None else conf
        self.params = conf.get('training_params', dict())
        self.entity = self.load('wordnet-id2entity.pkl')
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
        if not os.path.exists('models/wordnet_models/embedding/'):
            os.makedirs('models/wordnet_models/embedding/')
        model.save_weights('models/wordnet_models/embedding/weights_epoch_%d.h5' % epoch, overwrite=True)

    def load_epoch(self, model, epoch):
        assert os.path.exists('models/wordnet_models/embedding/weights_epoch_%d.h5' % epoch),\
            'Weights at epoch %d not found' % epoch
        model.load_weights('models/wordnet_models/embedding/weights_epoch_%d.h5' % epoch)

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
        split = self.params.get('validation_split', 0)

        training_set = self.load('wordnet-train.pkl')
        valid_set = self.load('wordnet-valid.pkl')

        subjects = list()
        relations = list()
        good_objects = list()

        for line in training_set:
            triplet = line.split('\t')
            subjects += [[int(triplet[0])]]
            relations += [[int(triplet[1])]]
            good_objects += [[int(triplet[2])]]

        subjects = np.asarray(subjects)
        relations = np.asarray(relations)
        good_objects = np.asarray(good_objects)
        num_bad = len(good_objects)
        bad_object_candidates = [[int(key)] for key in self.entity.keys()] * 4
        random.shuffle(bad_object_candidates)

        # subjects_valid = list()
        # relations_valid = list()
        # good_objects_valid = list()
        #
        # for line in valid_set:
        #     triplet = line.split('\t')
        #     subjects_valid += [[int(triplet[0])]]
        #     relations_valid += [[int(triplet[1])]]
        #     good_objects_valid += [[int(triplet[2])]]

        # subjects_valid = np.asarray(subjects_valid)
        # relations_valid = np.asarray(relations_valid)
        # good_objects_valid = np.asarray(good_objects_valid)

        val_loss = {'loss': 1., 'epoch': 0}

        for i in range(1, nb_epoch+1):
            # bad_answers = np.roll(good_answers, random.randint(10, len(questions) - 10))
            # bad_answers = good_answers.copy()
            # random.shuffle(bad_answers)
            bad_objects = np.asarray(random.sample(bad_object_candidates, num_bad))

            # shuffle question
            # zipped = zip(questions, good_answers)
            # random.shuffle(zipped)
            # questions[:], good_answers[:] = zip(*zipped)

            print('Epoch %d :: ' % i, end='')
            self.print_time()
            model.fit([subjects, relations, good_objects, bad_objects], nb_epoch=1, batch_size=batch_size)

            # if hist.history['val_loss'][0] < val_loss['loss']:
            #     val_loss = {'loss': hist.history['val_loss'][0], 'epoch': i}
            # print('Best: Loss = {}, Epoch = {}'.format(val_loss['loss'], val_loss['epoch']))

            if eval_every is not None and i % eval_every == 0:
                self.get_mrr(model)

            if save_every is not None and i % save_every == 0:
                self.save_epoch(model, i)

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
            self._eval_sets = dict([(s, self.load(s)) for s in ['wordnet-test.pkl']])
        return self._eval_sets

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

            # c_1 for hit@1, c_3 for hit@3, c_10 for hit@10, rr for mrr
            c_1, c_3, c_10, rr = 0, 0, 0, 0
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
                rr += 1 / float(target_rank + 1)

            hit_at_1 = c_1 / float(len(data))
            hit_at_3 = c_3 / float(len(data))
            hit_at_10 = c_10 / float(len(data))
            avg_rank = np.mean(mean_ranks)
            mrr = rr / float(len(data))

            del data

            if evaluate_all:
                print('Hit@1 Precision: %f' % hit_at_1)
                print('Hit@3 Precision: %f' % hit_at_3)
                print('Hit@10 Precision: %f' % hit_at_10)
                print('Mean Rank: %f' % avg_rank)
                print('MRR: %f' % mrr)

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
        'n_words': 40961,  # len(vocabulary)
        'margin': 0.2,

        'training_params': {
            'save_every': 100,
            # 'eval_every': 1,
            'batch_size': 128,
            'nb_epoch': 1000,
            'validation_split': 0,
            'optimizer': Adam(),
            # 'optimizer': Adam(clip_norm=0.1),
            # 'n_eval': 100,

            'evaluate_all_threshold': {
                'mode': 'all',
                'top1': 0.4,
            },
        },

        'model_params': {
            'n_embed_dims': 1000,
            'n_hidden': 200,

            # convolution
            'nb_filters': 1000, # * 4
            'conv_activation': 'relu',

            # recurrent
            'n_lstm_dims': 141, # * 2

            'initial_embed_weights': np.load('models/wordnet_word2vec_1000_dim.h5'),
        },

        'similarity_params': {
            'mode': 'cosine',
            'gamma': 1,
            'c': 1,
            'd': 2,
        }
    }

    evaluator = Evaluator(conf)

    ##### Embedding model ######
    model = EmbeddingModel(conf)
    optimizer = conf.get('training_params', dict()).get('optimizer', 'adam')

    # TransE model
    # model = TranEModel(conf)
    # optimizer = conf.get('training_params', dict()).get('optimizer', 'adam')

    model.compile(optimizer=optimizer)

    # save embedding layer
    # evaluator.load_epoch(model, 33)
    # embedding_layer = model.prediction_model.layers[2].layers[2]
    # evaluator.load_epoch(model, 100)
    # evaluator.train(model)
    # weights = embedding_layer.get_weights()[0]
    # np.save(open('models/embedding_1000_dim.h5', 'wb'), weights)

    # train the model
    # evaluator.load_epoch(model, 54)
    evaluator.train(model)
    # embedding_matrix = model.prediction_model.layers[3].layers[3].get_weights()[0]
    # print(np.linalg.norm(embedding_matrix[1, :]))
    # print(np.linalg.norm(embedding_matrix[:, 1]))

    # evaluate mrr for a particular epoch
    # evaluator.load_epoch(model, 5)
    # evaluator.get_mrr(model, evaluate_all=True)
