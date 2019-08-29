
# load data
from math import inf

import numpy as np
import os
import pickle
from os.path import join, isfile
from numpy import linalg as LA

import torch

from config import EMB_DIR
from utils import soft_indicator, vocab2vec

with open('data/scmag.pkl','rb') as f:
    Xs = pickle.load(f)

# TODO: check the data, whether the property p is satisfied

type = 'glove'
emb_fname = 'glove.6B.300d'
emb_xs_name = join(EMB_DIR,emb_fname+'_x')
emb_xps_name = join(EMB_DIR,emb_fname+'_xp')
emb_xms_name = join(EMB_DIR,emb_fname+'_xm')
base_emb = join(EMB_DIR,emb_fname+'.txt')

if isfile(emb_xs_name+'.npy') and isfile(emb_xps_name+'.npy') and isfile(emb_xms_name+'.npy'):

    P_x = np.load(emb_xs_name+'.npy')
    P_xp = np.load(emb_xps_name+'.npy')
    P_xm = np.load(emb_xms_name+'.npy')
else:

    xs,xps,xms = [],[],[]

    for X in Xs:
        x,xp,xm = X
        if x.is_integer():
            xs.append(str(int(x)))
        else:
            xs.append(str(x))
        if xp.is_integer():
            xps.append(str(int(xp)))
        else:
            xps.append(str(xp))
        if xm.is_integer():
            xms.append(str(int(xm)))
        else:
            xms.append(str(xm))
        # xs.append(str(x))
        # xps.append(str(xp))
        # xms.append(str(xm))


    _,P_x = vocab2vec(xs,EMB_DIR,emb_xs_name,base_emb,['npy'])
    _,P_xp = vocab2vec(xps,EMB_DIR,emb_xps_name,base_emb,['npy'])
    _,P_xm = vocab2vec(xms,EMB_DIR,emb_xms_name,base_emb,['npy'])

# calculate the original accuracy
Dp = LA.norm(P_x - P_xp,axis=0)
Dm = LA.norm(P_x - P_xm,axis=0)
acc = sum(Dp < Dm) / Dp.size
print('original acc: ',acc)

# model

d = 300
# w = torch.nn.Parameter(torch.randn(d))
w = torch.randn(d,requires_grad=True)
n_epochs = 1000

beta = 4
lr = 0.5
optimizer = torch.optim.Adam([w],lr)
n = len(Xs)
P_x = torch.tensor(P_x,dtype=torch.float)
P_xp = torch.tensor(P_xp,dtype=torch.float)
P_xm = torch.tensor(P_xm,dtype=torch.float)
pre_acc = -inf

# dp = torch.abs(torch.matmul(w/torch.norm(w),P_x-P_xp))
# dm = torch.abs(torch.matmul(w/torch.norm(w),P_x-P_xm))
#
# acc = float(torch.sum(dp < dm)) / n
# print('specialized init acc: ',acc)

for t in range(n_epochs):

    dp = torch.abs(torch.matmul(w/torch.norm(w),P_x-P_xp))
    dm = torch.abs(torch.matmul(w/torch.norm(w),P_x-P_xm))

    objs = soft_indicator(dm-dp,beta)
    loss = -torch.sum(objs)
    # print(loss)
    optimizer.zero_grad()

    loss.backward()

    acc = float(torch.sum(dp < dm)) / n
    if acc > pre_acc:
        best_w = w.clone()
        pre_acc = acc
        # print(best_w)
        print('specialized acc: ', acc)
        torch.save(w, 'w.pt')

    optimizer.step()

# todo: test the direction vector
# best_w = torch.load('w.pt')
dp = torch.abs(torch.matmul(best_w/torch.norm(best_w),P_x-P_xp))
dm = torch.abs(torch.matmul(best_w/torch.norm(best_w),P_x-P_xm))

best_acc = float(torch.sum(dp < dm)) / n
print('best acc: ',best_acc)