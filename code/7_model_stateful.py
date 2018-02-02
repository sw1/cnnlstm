#!/usr/bin/env python

from __future__ import print_function

import os
import sys
import math
import six.moves.cPickle
import csv
import numpy as np
import random
from sys import argv
import matplotlib.pyplot as plt
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
from keras import backend as K

class GenerateBatch(object):
    def __init__(self, read_len, windows, n_vocab,
            n_batch = 32, n_classes = 2, n_reads = 250, n_pad = 21,
            shuffle = True, one_hot = False):
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
        self.one_hot = one_hot
        self.out_type = 'float32'

        if one_hot:
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

    def __1hot_generation(self, fns, labels, ids):
        'Generates data of batch_size samples'

        # Initialization
        batch_y = np.zeros((self.n_batch, self.n_classes), dtype=np.float32)

        first_pass = True
        read_lens = list()
        read_idxs = dict()
        m = 0
        max_read_n = 99999
        while m < max_read_n:

            batch_X = np.zeros((self.n_batch, self.max_len, self.n_vocab), dtype=np.float32)

            for i,id in enumerate(ids):

                x = six.moves.cPickle.load(open(fns[id],'rb'))

                if first_pass:
                    batch_y[i][labels[id]] = 1.0
                    read_idxs[id] = random.sample(range(len(x)),len(x))
                    read_lens.append(len(x))

                read_idxs_batch = read_idxs[id][m:m + self.n_reads]
                for r,idx in enumerate(read_idxs_batch):
                    batch_X[i,r*self.read_len + r*self.n_pad:(r+1)*self.read_len \
                        + r*self.n_pad,x[idx]] = 1.0

            if first_pass:
                max_read_n = max(read_lens)
                first_pass = False

            m += self.n_reads

            yield [batch_X,batch_X,batch_X], batch_y, {'m':m,'max_read_n':max_read_n,'labels':labels,'read_idxs':read_idxs_batch}

    def generate(self, fns, labels, ids):
        'Generates batches of samples'

        # Generate order of exploration of dataset
        ids_tmp = self.__get_exploration_order(ids)

        # Generate batches
        end = int(len(ids)/self.n_batch)
        for i in range(end):
            # Find list of IDs
            batch_ids = ids_tmp[i*self.n_batch:(i+1)*self.n_batch]
            # Generate subsequence generators
            yield self.__1hot_generation(fns, labels, batch_ids)

def plot_performance(train,val,title='',path='show'):
    plt.plot(train)
    plt.plot(val)
    plt.title(title)
    plt.xlabel('epoch')
    plt.ylabel('')
    plt.legend(['training','validation'],loc='upper left')
    if path == 'show':
        plt.show()
    else:
        plt.savefig(path)

k = int(argv[1])
cl = argv[2]
one_hot = bool(int(argv[3]))

windows = (2,4,8)
n_reads = 150 # 250
n_batch = 128 # 64

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
          'shuffle': True,
          'one_hot': one_hot}

gen = GenerateBatch(**params)

n_epochs = 20
d_emb = 64
d_cnn = 32
d_lstm = 32

inputs = list()
submodels = list()

layer_embed = Embedding(input_dim=gen.n_vocab,
        output_dim=d_emb,
        #input_length=gen.max_len,
        name='embedding')

for i,w in enumerate(gen.windows):

    if gen.one_hot:

        inputs.append(Input(shape=gen.in_len,
            batch_shape=(gen.n_batch,gen.max_len,gen.n_vocab),
            dtype=gen.in_type,
            name='input_cp' + str(i)))
        layer_cnn = Conv1D(d_cnn,
            kernel_size=w,
            padding='same',
            activation='relu',
            name='cnn_' + str(i) + '_w' + str(w))(inputs[i])

    else:

        inputs.append(Input(shape=gen.in_len,
            dtype=gen.in_type,
            name='input_cp' + str(i)))
        embedding = layer_embed(inputs[i])
        layer_cnn = Conv1D(d_cnn,
            kernel_size=w,
            padding='same',
            activation='relu',
            name='cnn_' + str(i) + '_w' + str(w))(embedding)

    submodels.append(layer_cnn)

layer_cnns = concatenate(submodels,name='merge_cnn_features')
layer_dropout_1 = Dropout(.1,name='dropout_cnns')(layer_cnns)
layer_lstm_1 = Bidirectional(LSTM(d_lstm,
        return_sequences=True,
        stateful=True,
        dropout=.2,
        recurrent_dropout=.2),name='biLSTM')(layer_dropout_1)
layer_lstm_n = Lambda(lambda x: x[:,-1,:],output_shape=(2*d_lstm, ),
        name='biLSTM_last_layer')(layer_lstm_1)
output = Dense(gen.n_classes,activation='softmax',name='output')(layer_lstm_n)
model = Model(inputs=inputs,outputs=[output])

adam = Adam(lr=.01)
model.compile(loss='categorical_crossentropy',
        optimizer=adam,
        metrics=['accuracy'])


np.random.seed(114)

get_gradients = model.optimizer.get_gradients(model.total_loss,
                                              model.trainable_weights)

inputs = [model.layers[0].input,
          model.layers[1].input,
          model.layers[2].input,
          model.sample_weights[0],
          K.learning_phase(),
          model.targets[0]]

grad_fct = K.function(inputs=inputs,outputs=get_gradients)

train_generator = gen.generate(fns,labels,ids['train'])

sys.stdout = open('out/train_log.txt','w')

total_norm = 0
bi = 0
for b in train_generator:
    
    print('Estimating gradient  on batch %s/%s: ' % (bi+1,len(ids['train'])//gen.n_batch))
    sys.stdout.flush()

    sb = next(b) 
    
    gradients = grad_fct([sb[0][0],sb[0][1],sb[0][2],
                          np.ones(sb[0][0].shape[0]),
                          0,
                          sb[1]])

    total_norm += np.sqrt(np.sum([np.sum(np.square(g)) for g in gradients]))

    bi += 1

clip = total_norm / bi

print('Clipping gradient at %.3f.' % (clip))

adam_clip = Adam(lr=.01,clipnorm=clip)

model.compile(loss='categorical_crossentropy',
        optimizer=adam_clip,
        metrics=['accuracy'])


np.random.seed(434)

ei = 0
early_stop = 0
decrease_lr = 0
losses_train = list()
losses_val = list()
accs_train = list()
accs_val = list()
min_loss_train = 999999
min_loss_val = 999999

for e in range(n_epochs):

    train_generator = gen.generate(fns,labels,ids['train'])
    val_generator = gen.generate(fns,labels,ids['val'])

    print('Epoch %s:' % (ei))

    bi = 0
    for b in train_generator:

        print('Training on batch %s/%s: ' % (bi,len(ids['train'])//gen.n_batch),end='')

        sbi = 0
        for sb in b:

            if sbi == 0:
                print('%s steps over %s reads.' % (sb[2]['max_read_n']//gen.n_reads,sb[2]['max_read_n']))

            train_loss, train_acc = model.train_on_batch(x=sb[0],y=sb[1],class_weight=class_weights)
            
            if math.isnan(train_loss):
                print('\n Loss is NaN: epoch %s, batch %s, sub-batch %s; saving output.' % (ei,bi,sbi))
                six.moves.cPickle.dump({'b':b,'sb':sb,'ei':ei,'bi':bi,'sbi':sbi,'last':last},
                        open('out/nan_dump.pkl','wb'))
                model.save('out/model.nan.hd5f')
                sys.stdout.flush()
                sys.exit('Exit: NaN in loss.')
            else:
                last = {'b':b,'sb':sb,'ei':ei,'bi':bi,'sbi':sbi}
                model.save('out/model.last.hd5f')

            print('.',end='')
            if sbi % 25 == 0:
                print('\nSub-batch %s: Loss: %.5f' % (sbi,train_loss))
            sys.stdout.flush()

            if sbi % 100 == 0:
                model.save('out/model.tmp.hd5f')
                six.moves.cPickle.dump({'ei':ei,'bi':bi,'sbi':sbi},
                    open('out/indxes.tmp.pkl','wb'))

            sbi += 1

        model.reset_states()

        print('\tBatch %s Loss: %.5f; Accuracy:  %.3f' % (bi+1,train_loss,train_acc))

        model.save('out/model.tmp.hd5f')
        six.moves.cPickle.dump({'ei':ei,'bi':bi,'sbi':sbi},
                open('out/indexes.tmp.pkl','wb'))

        bi += 1

    val_loss = 0
    val_acc = 0
    bi = 0
    for b in val_generator:

        
        print('Validating on batch %s/%s: ' % (bi,len(ids['val'])//gen.n_batch),end='')

        sbi = 0
        for sb in b:

            if sbi == 0:
                print('Validating on %s samples: ' % (len(ids['val'])),end='')
                print('%s steps over %s reads.' % (sb[2]['max_read_n']//gen.n_reads,sb[2]['max_read_n']))

            val_loss_tmp, val_acc_tmp = model.test_on_batch(x=sb[0],y=sb[1])
            
            print('.',end='')
            sys.stdout.flush()

            sbi += 1

        val_loss += val_loss_tmp
        val_acc += val_acc_tmp

        model.reset_states()

        print('\nValidation Loss: %.5f; Accuracy:  %.3f' % (val_loss,val_acc))

        bi += 1

    val_loss /= bi
    val_acc /= bi

    losses_train.append(train_loss)
    losses_val.append(val_loss)
    accs_train.append(train_acc)
    accs_val.append(val_acc)

    if round(train_loss,3) < round(min_loss_train,3):
        early_stop = 0
        decrease_lr = 0
        min_loss_train = train_loss
        print('Saving temp. model.')
        model.save('out/model.epoch_' + str(ei) + 'end.tmp.hd5f')
    else:
        early_stop += 1
        decrease_lr += 1

    if early_stop == 10:
        print('Stopping early due to lack of improvement in loss.')
        break

    if ei > 3:
        if decrease_lr == 2:
            current_lr = K.eval(model.optimizer.lr)
            if current_lr > .0005:
                new_lr = current_lr * .65
                print('Decreasing learning rate from %s to %s.' % (current_lr, new_lr))
                model.optimizer.lr.assign(new_lr)
                decease_lr = 0

    plot_performance(losses_train,losses_val,title='Loss',path='out/loss.png')
    plot_performance(accs_train,accs_val,title='Accuracy',path='out/acc.png')

    ei += 1

    print('________________________________________________\n')

model.save('out/model.epoch_' + str(ei) + '.end.hd5f')

test_generator = gen.generate(fns,labels,ids['test'])

bi = 0
test_loss = 0
test_acc = 0
for b in test_generator:

    
    print('Testing on batch %s/%s: ' % (bi,len(ids['test'])//gen.n_batch),end='')

    sbi = 0
    for sb in b:

        if sbi == 0:
            print('Testing on %s samples: ' % (len(ids['test'])),end='')
            print('%s steps over %s reads.' % (sb[2]['max_read_n']//gen.n_reads,sb[2]['max_read_n']))

        test_loss_tmp, test_acc_tmp = model.test_on_batch(x=sb[0],y=sb[1])
        
        print('.',end='')
        sys.stdout.flush()

        sbi += 1

    test_loss += test_loss_tmp
    test_acc += test_acc_tmp

    model.reset_states()

    bi += 1

test_loss /= bi
test_acc /= bi

print('\nTesting Loss: %.5f; Accuracy:  %.3f' % (test_loss,test_acc))
sys.stdout.close()
