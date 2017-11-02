#!/usr/bin/env python

import os
import six.moves.cPickle
import csv
import numpy as np
import random
import gzip
from itertools import product
from collections import Counter
import time

def data():

    k = 4
    min_reads = 1000

    work_dir = os.path.expanduser('~/earth')
    out_dir = os.path.join(work_dir,'data/kmers_' + str(k))

    meta_fn = os.path.join(work_dir,'data/labels.csv')
    samples_fn = os.path.join(work_dir,'data/kmers_' + '_k_'  + str(k) + '.pkl')
    run_fns = os.listdir(os.path.join(work_dir,'data/filtered'))

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    meta = csv.reader(open(meta_fn,'r'), delimiter=',', quotechar='"')
    meta = {sample:site for sample,site in meta}

    v = 10

    nts = ['A','C','G','T']
    key = {''.join(kmer):i+1 for i,kmer in enumerate(product(nts,repeat=k))}
    key['PAD'] = 0

    sample_counter = 0
    reads = list()

    t1 = time.time()
    for fn in run_fns:

        run_id = fn.split('_')[0]

        if run_id not in meta.keys():
            continue

        sample_counter += 1

        in_file = gzip.open(os.path.join(work_dir,'data','filtered',fn))

        while True:
            line = in_file.readline().decode()
            if line[0] == '@':
                l = 0
                break

        reads = list()
        for line in in_file:

            if l % 4 == 0:

                read = line.decode().strip('\n')

    #             r = len(read) % k
    #             if r == 0:
    #                 end = len(read)
    #             else:
    #                 end = len(read) - r

    #             read_idxs = list()
    #             for i in range(0,end,k):
    #                 try:
    #                     idx = key[read[i:i+k]]
    #                 except:
    #                     idx = 0
    #                 read_idxs.append(idx)
    #             reads.append(read_idxs)

                read_idxs = list()
                M = len(read) - k + 1
                for n in range(M):
                    try:
                        idx = key[read[n:n+k]]
                    except:
                        idx = 0
                    read_idxs.append(idx)
                reads.append(read_idxs)

            l += 1

        if len(reads) > min_reads:
            out_fn = os.path.join(out_dir,run_id + '.pkl')
            six.moves.cPickle.dump(reads,open(out_fn,'wb'))

        if sample_counter % v == 0:
            t_diff = str(round((time.time() - t1)/60,1)) + ' min.'
            print('Processed ' + str(sample_counter) + ' samples in ' + t_diff)
            t1 = time.time()

data()
