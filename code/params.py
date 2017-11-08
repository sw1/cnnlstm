#!/usr/bin/env python

from itertools import product
from csv import writer

w = [(2,2,2),(2,4,8),(2,6,12)]
nr = [250,500]
nb = [32,64]

params = [w,nr,nb]
params_grid = list(product(*params))

with open('params.csv','w') as f:
    out = writer(f)
    out.writerows(params_grid)

