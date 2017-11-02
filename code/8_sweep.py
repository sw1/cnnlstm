#!/usr/bin/env python

from __future__ import print_function

import os
import sys
import six.moves.cPickle
import csv
import numpy as np
import random
from sklearn.model_selection import train_test_split
from itertools import product
from collections import Counter
from keras.models import Model
from keras.layers import Dense, LSTM, Dropout, Input, concatenate
from keras.layers.embeddings import Embedding
from keras.layers.pooling import GlobalMaxPooling1D
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.preprocessing import sequence
from keras.optimizers import Adam
from keras.utils import np_utils
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform, conditional


def data():

    k = 4
    seed_data = 42

    nts = ['A','C','G','T']
    key = {''.join(kmer):i+1 for i,kmer in enumerate(product(nts,repeat=k))}
    key['PAD'] = 0

    kmer_dir = 'data/kmers_' + str(k)

    meta_fn = 'data/labels.csv'

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

    ids_dict = {'train':ids_train,'val':ids_val,'test':ids_test}
    labels = {r:int(l) for r,l in meta}

    return labels, fns, ids_dict, key


class GenerateBatch(object):

    def __init__(self, read_len, n_batch = 32, n_classes = 2, n_reads = 250, n_pad = 21, 
            shuffle = True):
        'Initialization'
        self.read_len = read_len
        self.n_classes = n_classes
        self.n_batch = n_batch
        self.n_reads = n_reads
        self.n_pad = n_pad
        self.max_len = n_reads * (read_len + n_pad)
        self.shuffle = shuffle

    def __get_exploration_order(self, ids):
        'Generates order of exploration'
        # Find exploration order
        if self.shuffle == True:
            random.shuffle(ids)

        return ids

    def __data_generation(self, fns, labels, ids):
        'Generates data of batch_size samples'

        # Initialization
        batch_X = np.zeros((self.n_batch, self.max_len), dtype=int)
        batch_y = np.zeros((self.n_batch,n_classes), dtype=float)

        for i,id in enumerate(ids):

            x = six.moves.cPickle.load(open(fns[id],'rb'))
            batch_y[i][labels[id]] = 1.0
            read_idxs = random.sample(range(len(x)),self.n_reads)

            for r,idx in enumerate(read_idxs):
                batch_X[i,r*self.read_len + r*self.n_pad:(r+1)*self.read_len + r*self.n_pad] = x[idx]

        return batch_X, batch_y

    def generate(self, fns, labels, ids):
        'Generates batches of samples'
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
                batch_X, batch_y = self.__data_generation(fns, labels, batch_ids)

                yield [batch_X,batch_X,batch_X], batch_y

    def test(self, fns, labels, ids,trim=True):

        reads = list()
        reads_len = list()
        labs = np.zeros((len(ids),n_classes), dtype=float)

        for i,id in enumerate(ids):

            labs[i][labels[id]] = 1.0

            i_reads = six.moves.cPickle.load(open(fns[id],'rb'))
            random.shuffle(i_reads)

            reads_len.append(len(i_reads))
            read = list()

            for r in i_reads:
                read.extend(r)
                read.extend([0]*self.n_pad)

            reads.append(read)

        reads = sequence.pad_sequences(reads, maxlen=self.max_len)

        return [reads,reads,reads], labs

def model(labels,fns,ids,key):

    n_epochs = 1 #25
    
    windows = {{choice([(2,2,2),(4,4,4),(2,4,8),(2,6,10)])}}

    read_len = len(six.moves.cPickle.load(open(next(iter(fns.values())),'rb'))[0])

    n_classes = max(labels.values()) + 1

    params = {'read_len': read_len,
              'n_classes': n_classes,
              'n_batch': {{choice([32,64])}},
              'n_reads': {{choice([250,600,999])}},
              'n_pad': max(windows) * k,
              'shuffle': True}

    gen = GenerateBatch(**params)
    train_generator = gen.generate(fns,labels,ids['train'])
    val_generator = gen.generate(fns,labels,ids['val'])


    inputs = list()
    submodels = list()

    for i,w in enumerate(windows):
        inputs.append(Input(shape=(gen.max_len,), dtype='int32'))
        layer_embed = Embedding(input_dim=len(key), 
                output_dim={{choice([32,64,128])}}, 
                input_length=gen.max_len,mask_zero=False)(inputs[i])
        layer_cnn = Conv1D({{choice([32,64,128])}}, 
                kernel_size=w, padding='same', activation='relu')(layer_embed)
        submodels.append(layer_cnn)

    layer_cnns = concatenate(submodels)
    layer_lstm_1 = LSTM({{choice([64,128,256])}}, 
            dropout={{uniform(0,.5)}}, 
            recurrent_dropout={{uniform(0,.5)}})(layer_cnns)
    output = Dense(n_classes, activation='softmax')(layer_lstm_1)
    model = Model(inputs=inputs, outputs=[output])

    model.compile(loss='categorical_crossentropy',
            optimizer={{choice(['rmsprop','adam'])}},
            metrics=['accuracy'])

    model.fit_generator(generator = train_generator,
                        steps_per_epoch = len(ids['train'])//gen.n_batch,
                        validation_data = val_generator,
                        validation_steps = len(ids['test'])//gen.n_batch,
                        epochs=n_epochs,
                        verbose=1)

    X_testset, y_testset = gen.test(fns,labels,ids['test'])

    scores,acc = model.evaluate(X_testset,y_testset,batch_size=params['n_batch'],verbose=1)
    print('Test accuracy:',acc)

    return {'loss':-acc, 'status': STATUS_OK, 'model': model}
    
if __name__ == '__main__':

    max_evals = 1 #5

    best_run, best_model = optim.minimize(model=model,data=data,algo=tpe.suggest,
            max_evals=max_evals,trials=Trials())
    
    six.moves.cPickle.dump({'best_run':best_run,'best_model':best_model},
            open('out/sweep.pkl','wb'))

    print('Best hyperparams: ')
    print(best_run)
