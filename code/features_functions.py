#!/usr/bin/env python

from __future__ import print_function

import os
import sys
import six.moves.cPickle
import csv
import numpy as np
import random
from sys import argv
from sklearn.model_selection import train_test_split
from itertools import product
from collections import Counter
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard, EarlyStopping
from keras.models import Model
from keras.layers import Dense, LSTM, Dropout, Input, concatenate, Bidirectional, Lambda
from keras.layers.embeddings import Embedding
from keras.layers.pooling import GlobalMaxPooling1D
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.preprocessing import sequence
from keras.utils import np_utils

class GenerateBatch(object):
    def __init__(self, path, windows=(2,4,8), k=4,
            n_batch = 32, n_reads = 250,
            shuffle = True, one_hot = False, seed=43):

        nts = ['A','C','G','T']
        key = {''.join(kmer):i+1 for i,kmer in enumerate(product(nts,repeat=k))}
        key['PAD'] = 0

        kmer_dir = os.path.join(path,'kmers_' + str(k))
        meta_fn = os.path.join(path,'labels.csv')

        fns = {f.split('.pkl')[0]:os.path.join(kmer_dir, f) for f in os.listdir(kmer_dir)
               if os.path.isfile(os.path.join(kmer_dir, f))}

        meta = csv.reader(open(meta_fn,'r'), delimiter=',', quotechar='"')
        meta = [(r,l) for r,l in meta if r in fns]

        train, test = train_test_split(meta,test_size=0.2,random_state=seed,
            shuffle=True,stratify=[l for r,l in meta])
        val, test = train_test_split(test,test_size=0.5,random_state=seed,
            shuffle=True,stratify=[l for r,l in test])

        ids_train = [r for r,l in train]
        ids_val = [r for r,l in val]
        ids_test = [r for r,l in test]

        ids = {'train':ids_train,'val':ids_val,'test':ids_test}
        labels = {r:int(l) for r,l in meta}

        read_len = len(six.moves.cPickle.load(open(next(iter(fns.values())),'rb'))[0])

        n_classes = max(labels.values()) + 1
        class_weights = Counter(labels.values())

        'Initialization'
        self.read_len = read_len
        self.windows = windows
        self.n_vocab = len(key)
        self.n_classes = n_classes
        self.n_batch = n_batch
        self.n_reads = n_reads
        self.n_pad = max(windows) * k
        self.max_len = n_reads * (read_len + self.n_pad)
        self.shuffle = shuffle
        self.one_hot = one_hot
        self.out_type = 'float32'
        self.fns = fns
        self.labels = labels
        self.ids = ids
        self.key = key
        self.rev_key = {v:k for k,v in key.items()}

        if self.one_hot:
            self.in_len = (None, self.n_vocab) #(self.max_len, self.n_vocab,)
            self.in_type = 'float32'
        else:
            self.in_len = (None,) #(self.max_len,)
            self.in_type = 'int32'

    def __get_exploration_order(self, ids):
        'Generates order of exploration'
        # Find exploration order
        if self.shuffle == True:
            random.shuffle(ids)

        return ids

    def __data_generation(self, ids):
        'Generates data of batch_size samples'

        # Initialization
        batch_X = np.zeros((self.n_batch, self.max_len), dtype=np.int32)
        batch_y = np.zeros((self.n_batch, self.n_classes), dtype=np.float32)

        read_idxs_dict = dict()

        for i,id in enumerate(ids):

            batch_y[i][self.labels[id]] = 1.0

            x = six.moves.cPickle.load(open(self.fns[id],'rb'))
            read_idxs = np.random.randint(0,len(x)-1,self.n_reads)
            read_idxs_dict[id] = read_idxs

            for r,idx in enumerate(read_idxs):
                batch_X[i,r*self.read_len + r*self.n_pad:(r+1)*self.read_len \
                        + r*self.n_pad] = x[idx]

        return batch_X, batch_y, read_idxs_dict

    def __1hot_generation(self, ids):
        'Generates data of batch_size samples'

        # Initialization
        batch_X = np.zeros((self.n_batch, self.max_len, self.n_vocab), dtype=np.float32)
        batch_y = np.zeros((self.n_batch, self.n_classes), dtype=np.float32)

        read_idxs_dict = dict()

        for i,id in enumerate(ids):

            batch_y[i][self.labels[id]] = 1.0

            x = six.moves.cPickle.load(open(self.fns[id],'rb'))
            read_idxs = np.random.randint(0,len(x)-1,self.n_reads)
            read_idxs_dict[id] = read_idxs

            position = 0
            for read in read_idxs[:-1]:
                for idx in x[read]:
                    batch_X[i,position,idx] = 1.0
                    position += 1
                for pad in range(self.n_pad):
                    batch_X[i,position,0] = 1.0
                    position += 1
            for idx in x[read_idxs[-1]]:
                batch_X[i,position,idx] = 1.0
                position += 1

        return batch_X, batch_y, read_idxs_dict

    def generate(self, dataset='train'):
        'Generates batches of samples'

        ids = self.ids[dataset]

        # Infinite loop
        while True:
            # Generate order of exploration of dataset
            ids_tmp = self.__get_exploration_order(ids)

            # Generate batches
            end = int(len(ids)/self.n_batch)
            for i in range(end):
                # Find list of IDs
                batch_ids = ids_tmp[i*self.n_batch:(i+1)*self.n_batch]
                # Generate data
                if self.one_hot:
                    batch_X, batch_y, read_idxs_dict = self.__1hot_generation(batch_ids)
                else:
                    batch_X, batch_y, read_idxs_dict = self.__data_generation(batch_ids)

                yield [batch_X,batch_X,batch_X], batch_y, batch_ids, read_idxs_dict

    def __data_test(self, ids, n_reads):

        max_len = n_reads * (self.read_len + self.n_pad)

        reads = list()

        read_idxs_dict = dict()

        for i,id in enumerate(ids):

            i_reads = six.moves.cPickle.load(open(self.fns[id],'rb'))
            read_idxs = np.random.randint(0,len(i_reads)-1,n_reads)
            read_idxs_dict[id] = read_idxs

            read = list()

            for idx in read_idxs:
              for r in i_reads[idx]:
                  read.extend(r)
                  read.extend([0]*self.n_pad)

            reads.append(read)

        reads = sequence.pad_sequences(reads, maxlen=max_len)

        return reads, read_idxs_dict

    def __1hot_test(self, ids, n_reads):

        max_len = n_reads * (self.read_len + self.n_pad)

        reads = np.zeros((len(ids), max_len, self.n_vocab), dtype=np.float32)

        read_idxs_dict = dict()

        for i,id in enumerate(ids):

            x = six.moves.cPickle.load(open(self.fns[id],'rb'))
            read_idxs = np.random.randint(0,len(i_reads)-1,n_reads)
            read_idxs_dict[id] = read_idxs

            position = 0
            for read in read_idxs[:-1]:
                for idx in x[read]:
                    reads[i,position,idx] = 1.0
                    position += 1
                for pad in range(self.n_pad):
                    reads[i,position,0] = 1.0
                    position += 1
            for idx in x[read_idxs[-1]]:
                reads[i,position,idx] = 1.0
                position += 1

        return reads, read_idxs_dict

    def test(self, n_reads=None):

        ids = self.ids('test')

        if n_reads is None:
            n_reads = self.n_reads

        labs = np.zeros((len(ids),self.n_classes), dtype=np.float32)

        for i,id in enumerate(ids):
            labs[i][self.labels[id]] = 1.0

        if self.one_hot:
            reads, read_idxs_dict = self.__1hot_test(ids, n_reads)
        else:
            reads, read_idxs_dict = self.__data_test(ids, n_reads)

        return [reads,reads,reads], labs, ids, read_idxs_dict
