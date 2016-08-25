from __future__ import print_function

import os
import sys
import random

import pickle

from gensim.models import Word2Vec

from keras_models import *

random.seed(42)


def load(path, name):
    return pickle.load(open(os.path.join(path, name), 'rb'))


def revert(triplet):
    return [term for term in triplet.split('\t')]

if __name__ == '__main__':
    try:
        data_path = 'data/wordnet'
    except KeyError:
        print("data_path is not set.")
        sys.exit(1)

    size = 1000
    # assert os.path.exists('models/embedding_%d_dim.h5' % size)

    sentences = list()
    for line in load(data_path, 'wordnet-train.pkl'):
        sentences.append(revert(line))

    print('Training Word2Vec model...')
    model = Word2Vec(sentences, size=size, min_count=1, window=2, sg=1, iter=25)
    weights = model.syn0
    d = dict([(k, v.index) for k, v in model.vocab.items()])

    # this is the stored weights of an equivalent embedding layer
    # there is some commented code in insurance_qa_eval.py for generating this
    emb = np.random.rand(40961, 1000)

    # swap the word2vec weights with the embedded weights
    for i in xrange(40961):
        emb[i, :] = weights[d[str(i)], :]

    np.save(open('models/wordnet_word2vec_%d_dim.h5' % size, 'wb'), emb)
