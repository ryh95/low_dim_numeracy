
# load data
import math
from math import inf

import numpy as np
import os
import pickle
from os.path import join, isfile
from numpy import linalg as LA

import torch

from config import EMB_DIR
from utils import soft_indicator, vocab2vec

with open('data/ovamag.pkl','rb') as f:
    Xss = pickle.load(f)

# TODO: check the data, whether the property p is satisfied

type = 'glove'
emb_fname = 'glove.6B.300d'
d = 300 # todo: auto derive the d
emb_xs_name = join(EMB_DIR,emb_fname+'_ova_x')
emb_xps_name = join(EMB_DIR,emb_fname+'_ova_xp')
emb_xms_name = join(EMB_DIR,emb_fname+'_ova_xm')
base_emb = join(EMB_DIR,emb_fname+'.txt')

if isfile(emb_xs_name+'.npy') and isfile(emb_xps_name+'.npy') and isfile(emb_xms_name+'.npy'):

    P_x = np.load(emb_xs_name+'.npy') # dxp
    P_xp = np.load(emb_xps_name+'.npy') # dxp
    P_xms = np.load(emb_xms_name+'.npy') # a dx(n-2)xp dimensional tensor

else:

    # xs,xps,xms = [],[],[]

    # collect all number embedding
    number_set = set(np.array(Xss).flat)
    number_array = np.zeros(len(number_set),dtype=str)
    number_array[:-1] = np.array(number_set - set([math.inf])).astype(int).astype(str)
    number_array[-1] = str(math.inf)
    number_emb_dict,_ = vocab2vec(number_array, EMB_DIR, emb_xms_name, base_emb, ['pickle','npy'])

    xs,xps = [],[]
    n_ova = len(Xss[0])
    n_tests = len(Xss)
    P_xms = np.zeros((n_tests, d, n_ova), dtype=np.float32)
    P_x = np.zeros((n_tests,d), dtype=np.float32)
    P_xp = np.zeros((n_tests,d), dtype=np.float32)
    for i,Xs in enumerate(Xss):
        xms = []
        for j,(x,xp,xm) in enumerate(Xs):
            if xm.is_integer():
                P_xms[i,:,j] = number_emb_dict[str(int(xm))]
            else:
                P_xms[i,:,j] = number_emb_dict[str(xm)]

        if Xs[0].is_integer():
            P_x[i,:] = number_emb_dict[str(int(Xs[0]))]
        else:
            P_x[i,:] = number_emb_dict[str(Xs[0])]
        if Xs[1].is_integer():
            P_xp[i,:] = number_emb_dict[str(int(Xs[1]))]
        else:
            P_xp[i,:] = number_emb_dict[str(Xs[1])]

        # xs.append(str(x))
        # xps.append(str(xp))
        # xms.append(str(xm))

# calculate the original accuracy
Dp = LA.norm(P_x - P_xp,axis=1)
Dm = LA.norm(P_x[:,:,None] - P_xms,axis=1).min(axis=1)
acc = sum(Dp <= Dm) / Dp.size
print('original acc: ',acc)

# model

# w = torch.nn.Parameter(torch.randn(d))
w = torch.randn(d,requires_grad=True)
n_epochs = 1000

beta = 4
lr = 0.5
optimizer = torch.optim.Adam([w],lr)
P_x = torch.tensor(P_x,dtype=torch.float)
P_xp = torch.tensor(P_xp,dtype=torch.float)
P_xms = torch.tensor(P_xms,dtype=torch.float)
pre_acc = -inf

# dp = torch.abs(torch.matmul(w/torch.norm(w),P_x-P_xp))
# dm = torch.abs(torch.matmul(w/torch.norm(w),P_x-P_xm))
#
# acc = float(torch.sum(dp < dm)) / n
# print('specialized init acc: ',acc)

for t in range(n_epochs):

    dp = torch.abs(torch.matmul(P_x-P_xp,w/torch.norm(w)))
    dm = torch.min(torch.abs(torch.matmul(w/torch.norm(w),P_x[:,:,None]-P_xms)),dim=1)

    objs = soft_indicator(dm-dp,beta)
    loss = -torch.sum(objs)
    # print(loss)
    optimizer.zero_grad()

    loss.backward()

    acc = float(torch.sum(dp < dm)) / n_tests
    if acc > pre_acc:
        best_w = w.clone()
        pre_acc = acc
        # print(best_w)
        print('specialized acc: ', acc)
        torch.save(w, 'w_ova.pt')

    optimizer.step()

# todo: test the direction vector
# best_w = torch.load('w.pt')
dp = torch.abs(torch.matmul(P_x-P_xp,best_w/torch.norm(best_w)))
dm = torch.min(torch.abs(torch.matmul(best_w/torch.norm(best_w),P_x[:,:,None]-P_xms)),dim=1)

best_acc = float(torch.sum(dp < dm)) / n_tests
print('best acc: ',best_acc)