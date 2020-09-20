# -*- coding: utf-8 -*-

PATH = 'XX'

import os
os.chdir(PATH)
import numpy as np
import pandas as pd
from CMLE import BinLogitCMLE

df = pd.read_excel('synthetic_data.xlsx')
n, T = int(len(df)/3), 3
W = np.zeros((n, T, 1))
Y = np.zeros((n, T))
for i, elem in enumerate(np.unique(df['id'])):
    W[i,:,:] = df.loc[df['id']==elem, ['x']].values
    Y[i,:] = np.array(df.loc[df['id']==elem, 'y'])

CMLE = BinLogitCMLE(W, Y)
print(CMLE.separation_test())
