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
from hyperopt import Trials, STATUS_OK, tpe, rand
from hyperas import optim
from hyperas.distributions import choice, uniform, conditional



class GenerateBatch(object):
    def __init__(self, read_len, windows, n_vocab, 
            n_batch = 32, n_classes = 2, n_reads = 250, n_pad = 21, 
            shuffle = True):
        'Initialization'
        self.read_len = read_len
        self.windows = windows
        self.n_vocab = n_vocab
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
        batch_y = np.zeros((self.n_batch, self.n_classes), dtype=float)

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
        labs = np.zeros((len(ids),self.n_classes), dtype=float)

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


k = int(argv[1])
cl = argv[2]

windows = (2,4,8)
n_reads = 250
n_batch = 32

seed = 42

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

ids = {'train':ids_train,'val':ids_val,'test':ids_test}
labels = {r:int(l) for r,l in meta}

read_len = len(six.moves.cPickle.load(open(next(iter(fns.values())),'rb'))[0])

n_classes = max(labels.values()) + 1
class_weights = Counter(labels.values())

params = {'read_len': read_len,
          'windows': windows,
          'n_vocab': len(key),
          'n_classes': n_classes,
          'n_batch': n_batch,  
          'n_reads': n_reads,
          'n_pad': max(windows) * k,
          'shuffle': True}

gen = GenerateBatch(**params)

train_generator = gen.generate(fns,labels,ids['train'])
val_generator = gen.generate(fns,labels,ids['val'])





n_epochs = 100
d_emb = 64
d_cnn = 32
d_lstm = 32

inputs = list()
submodels = list()

layer_embed = Embedding(input_dim=gen.n_vocab, 
        output_dim=d_emb, 
        input_length=gen.max_len,mask_zero=False,
        name='embedding')

for i,w in enumerate(gen.windows):
    inputs.append(Input(shape=(gen.max_len,),
        dtype='int32',
        name='input_cp' + str(i)))
    embedding = layer_embed(inputs[i])
    layer_cnn = Conv1D(d_cnn, 
            kernel_size=w,
            padding='same',
            activation='relu',
            name='cnn_w' + str(w))(embedding)
    submodels.append(layer_cnn)

layer_cnns = concatenate(submodels,name='merge_cnn_features')
layer_dropout_1 = Dropout(.5)(layer_cnns) 
layer_lstm_1 = Bidirectional(LSTM(d_lstm,
        return_sequences=True,
        dropout=.25, 
        recurrent_dropout=.1),name='biLSTM')(layer_dropout_1)
layer_lstm_n = Lambda(lambda x: x[:,-1,:],output_shape=(2*d_lstm, ),
        name='biLSTM_last_layer')(layer_lstm_1)
output = Dense(gen.n_classes,activation='softmax',name='output')(layer_lstm_n)
model = Model(inputs=inputs,outputs=[output])

model_cp = ModelCheckpoint('out/model_k' + str(k) + '_{epoch:02d}_{val_loss:.2f}.hdf5',
        monitor='loss',verbose=1,save_best_only=True,period=1)
model_reduce_lr = ReduceLROnPlateau(monitor='loss',factor=0.65,patience=3,min_lr=.00005,
        verbose=1)
model_stop = EarlyStopping(monitor='loss',min_delta=0.01,patience=25,verbose=1)
model_tb = TensorBoard(log_dir='/scratch/sw424/train_' + cl + '/logs_k' + str(k),
        write_graph=True,write_grads=True,
        batch_size=32,write_images=True)
cbs = [model_cp,model_reduce_lr,model_stop,model_tb]

adam = Adam(lr=.01)

model.compile(loss='categorical_crossentropy',
        optimizer=adam,
        metrics=['accuracy'])

model.fit_generator(generator = train_generator,
                    steps_per_epoch = len(ids['train'])//gen.n_batch,
                    validation_data = val_generator,
                    validation_steps = len(ids['val'])//gen.n_batch,
                    epochs=n_epochs,
                    class_weight=class_weights,
                    callbacks=cbs,
                    verbose=1)

X_testset, y_testset = gen.test(fns,labels,ids['test'])
scores, acc = model.evaluate(X_testset,y_testset)


model.save('out/model_final_k' + str(k) + '.pkl')

six.moves.cPickle.dump({'ids':ids,'labels':labels},
                open('out/model_final_k' + str(k) + '_ids.pkl','wb'))

six.moves.cPickle.dump({'test':scores,'acc':acc},
                open('out/model_final_k' + str(k) + '_testscores.pkl','wb'))

print(scores)
