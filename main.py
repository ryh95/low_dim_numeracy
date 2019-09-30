
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

with open('data/scmag.pkl','rb') as f:
    Xs = pickle.load(f)

# TODO: check the data, whether the property p is satisfied

type = 'glove'
emb_fname = 'random' # random
emb_xs_name = join(EMB_DIR,emb_fname+'_x')
emb_xps_name = join(EMB_DIR,emb_fname+'_xp')
emb_xms_name = join(EMB_DIR,emb_fname+'_xm')
emb_num_name = join(EMB_DIR,emb_fname+'_ova_all') # todo: change this name later
base_emb = join(EMB_DIR,emb_fname+'.txt')

if isfile(emb_xs_name+'.npy') and isfile(emb_xps_name+'.npy') and isfile(emb_xms_name+'.npy'):

    P_x = np.load(emb_xs_name+'.npy')
    P_xp = np.load(emb_xps_name+'.npy')
    P_xm = np.load(emb_xms_name+'.npy')
else:

    xs,xps,xms = [],[],[]

    number_set = set(np.array(Xs).flat)
    number_set.remove(math.inf)
    number_array = [str(int(n)) for n in number_set]
    number_array.append(str(math.inf))

    if os.path.isfile(emb_num_name+'.pickle'):
        with open(emb_num_name+'.pickle','rb') as f:
            number_emb_dict = pickle.load(f)
    else:
        if emb_fname == 'random':
            d = 300
            number_emb_dict = {n:np.random.randn(d) for n in number_array}
            with open(join(EMB_DIR,emb_fname+'.pickle'), 'wb') as handle:
                pickle.dump(number_emb_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            number_emb_dict,_ = vocab2vec(number_array, EMB_DIR, emb_num_name, base_emb, ['pickle','npy'])

    P_x,P_xp,P_xm = [],[],[]

    for X in Xs:
        x,xp,xm = X
        if x.is_integer():
            P_x.append(number_emb_dict[str(int(x))])
        else:
            P_x.append(number_emb_dict[str(x)])
        if xp.is_integer():
            P_xp.append(number_emb_dict[str(int(xp))])
        else:
            P_xp.append(number_emb_dict[str(xp)])
        if xm.is_integer():
            P_xm.append(number_emb_dict[str(int(xm))])
        else:
            P_xm.append(number_emb_dict[str(xm)])


    P_x = np.vstack(P_x).T
    P_xp = np.vstack(P_xp).T
    P_xm = np.vstack(P_xm).T
    np.save(emb_xs_name+'.npy',P_x)
    np.save(emb_xps_name+'.npy',P_xp)
    np.save(emb_xms_name+'.npy',P_xm)

# calculate the original accuracy
Dp = LA.norm(P_x - P_xp,axis=0)
Dm = LA.norm(P_x - P_xm,axis=0)
acc = sum(Dp < Dm) / Dp.size
print('original acc: ',acc)

# model

d = 300
dim = 10
# w = torch.nn.Parameter(torch.randn(d))
# w = torch.randn(d,requires_grad=True)
# w.data = w.data/torch.norm(w).data
W = torch.randn((dim,d),requires_grad=True)
# w = torch.randn(d,requires_grad=True)
# w.data = w.data/torch.norm(w).data
torch.nn.init.orthogonal_(W)
W.data = W.T.data # col orthognol

n_epochs = 1000

beta = 14
lr = 0.5
optimizer = torch.optim.Adam([W],lr)
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

    # dp = torch.abs(torch.matmul(w,P_x-P_xp))
    # dm = torch.abs(torch.matmul(w,P_x-P_xm))
    dp = torch.norm(torch.matmul(W.T, P_x - P_xp),dim=0)
    dm = torch.norm(torch.matmul(W.T, P_x - P_xm),dim=0)

    objs = soft_indicator(dm-dp,beta)
    loss = -torch.mean(objs)
    # print(loss)
    optimizer.zero_grad()

    loss.backward()

    acc = float(torch.sum(dp < dm)) / n
    if acc > pre_acc:
        best_W = W.clone()
        pre_acc = acc
        # print(best_w)
        print('specialized acc: ', acc)
        torch.save(W, 'W.pt')

    optimizer.step()
    # w.data = w.data / torch.norm(w).data
    u, s, v = torch.svd(W.data)
    W.data = u @ v.T
    assert not torch.isnan(W.data).any(), 'W has nan values'

# todo: test the direction vector
# best_w = torch.load('w.pt')
dp = torch.norm(torch.matmul(best_W.T, P_x - P_xp), dim=0)
dm = torch.norm(torch.matmul(best_W.T, P_x - P_xm), dim=0)

best_acc = float(torch.sum(dp < dm)) / n
print('best acc: ',best_acc)