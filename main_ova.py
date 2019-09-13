
# load data
import math
from math import inf

import numpy as np
import os
import pickle
from os.path import join, isfile
from numpy import linalg as LA

import torch
from torch import autograd
from torch.utils.data import TensorDataset, DataLoader

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
emb_num_name = join(EMB_DIR,emb_fname+'_ova_all')
base_emb = join(EMB_DIR,emb_fname+'.txt')

if isfile(emb_xs_name+'.npy') and isfile(emb_xps_name+'.npy') and isfile(emb_xms_name+'.npy'):

    P_x = np.load(emb_xs_name+'.npy') # dxp
    P_xp = np.load(emb_xps_name+'.npy') # dxp
    P_xms = np.load(emb_xms_name+'.npy') # a dx(n-2)xp dimensional tensor

else:

    # xs,xps,xms = [],[],[]

    # collect all number embedding
    number_set = set(np.array(Xss).flat)
    number_set.remove(math.inf)
    number_array = [str(int(n)) for n in number_set]
    number_array.append(str(math.inf))
    if os.path.isfile(emb_num_name+'.pickle'):
        with open(emb_num_name+'.pickle','rb') as f:
            number_emb_dict = pickle.load(f)
    else:
        number_emb_dict,_ = vocab2vec(number_array, EMB_DIR, emb_num_name, base_emb, ['pickle','npy'])

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

        if Xs[0][0].is_integer():
            P_x[i,:] = number_emb_dict[str(int(Xs[0][0]))]
        else:
            P_x[i,:] = number_emb_dict[str(Xs[0][0])]
        if Xs[0][1].is_integer():
            P_xp[i,:] = number_emb_dict[str(int(Xs[0][1]))]
        else:
            P_xp[i,:] = number_emb_dict[str(Xs[0][1])]

        # xs.append(str(x))
        # xps.append(str(xp))
        # xms.append(str(xm))
    np.save(emb_xs_name+'.npy',P_x)
    np.save(emb_xps_name+'.npy',P_xp)
    np.save(emb_xms_name+'.npy',P_xms)

# calculate the original accuracy
batch_size = P_x.shape[0]
beta = 80

Dp = LA.norm(P_x - P_xp,axis=1)
Dm = LA.norm(P_x[:,:,None] - P_xms,axis=1).min(axis=1)

I_hat = float(torch.sum(soft_indicator(torch.tensor(Dm - Dp, dtype=torch.float), beta=beta)))/batch_size
acc = sum(Dp <= Dm) / batch_size
print('original acc: ',acc)
print('original I_hat: ', -I_hat)

# model
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# w = torch.nn.Parameter(torch.randn(d))
w = torch.randn(d,requires_grad=True,device=device)
# w = torch.randn(d,requires_grad=True)
w.data = w.data/torch.norm(w).data

n_epochs = 20

lr = 0.002

P_x = torch.from_numpy(P_x).float()
P_xp = torch.from_numpy(P_xp).float()
P_xms = torch.from_numpy(P_xms).float()
train_data = TensorDataset(P_x,P_xp,P_xms)

optimizer = torch.optim.Adam([w],lr)

init_acc = -inf
init_loss = inf


def evaluate_w(P_x,P_xp,P_xms,mini_batch_size,w):
    train_data = TensorDataset(P_x, P_xp, P_xms)
    mini_batchs = DataLoader(train_data,batch_size=mini_batch_size,shuffle=True,num_workers=8)
    loss,acc = 0,0
    for mini_batch in mini_batchs:
        mini_P_x, mini_P_xp, mini_P_xms = mini_batch
        dp = torch.abs(torch.matmul(mini_P_x-mini_P_xp,w.cpu()))
        dm = torch.min(torch.abs(torch.matmul(w.cpu(),mini_P_x[:,:,None]-mini_P_xms)),dim=1)[0]
        #
        objs = soft_indicator(dm-dp,beta)
        loss += -torch.sum(objs).item()
        acc += torch.sum(dp<=dm).item()
    batch_size = P_x.shape[0]
    return acc/batch_size,loss/batch_size

mini_batch_size = 128

acc,loss = evaluate_w(P_x,P_xp,P_xms,mini_batch_size,w)
print('init specialized acc: ',acc)
print('init specialized I_hat: ',loss)

mini_batchs = DataLoader(train_data,batch_size=mini_batch_size,shuffle=True,num_workers=8)

for t in range(n_epochs):

    print('epoch number: ', t)

    for i,mini_batch in enumerate(mini_batchs):

        mini_P_x, mini_P_xp, mini_P_xms = mini_batch
        mini_P_x = mini_P_x.to(device)
        mini_P_xp = mini_P_xp.to(device)
        mini_P_xms = mini_P_xms.to(device)

        dp = torch.abs(torch.matmul(mini_P_x-mini_P_xp,w))
        dm = torch.min(torch.abs(torch.matmul(w,mini_P_x[:,:,None]-mini_P_xms)),dim=1)[0]

        objs = soft_indicator(dm-dp,beta)
        loss = -torch.mean(objs)
        # print(loss)
        optimizer.zero_grad()

        with autograd.detect_anomaly():
            # avoid nan gradient
            loss.backward()

        # evaluate
        if i % 10 == 0:
            # acc = float(torch.sum(dp <= dm)) / batch_size
            acc,_ = evaluate_w(P_x,P_xp,P_xms,mini_batch_size,w)
            print("epochs :{}, acc :{} , iteration: {},".format(t, acc, i))
            if acc > init_acc:
                best_w = w.clone()
                init_acc = acc
                # print(best_w)
                print('specialized acc: ', acc)
                torch.save(w, 'w_ova.pt')

        # if loss < init_loss:
        #     best_w = w.clone()
        #     init_loss = loss
        #     print('specialized I_hat: ', loss)
        #     torch.save(w, 'w_ova.pt')

        optimizer.step()

        # project the variables back to the feasible set
        w.data = w.data / torch.norm(w).data

# best_w = torch.load('w.pt')

best_acc,best_loss = evaluate_w(P_x,P_xp,P_xms,mini_batch_size,best_w)
print('best acc: ',best_acc)
print('best specialized I_hat: ',best_loss)