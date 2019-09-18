import math
import os
import pickle
import torch
from os.path import join

import numpy as np

from config import EMB_DIR
from utils import vocab2vec, soft_indicator

emb_fname = 'glove.6B.300d'
emb_num_name = join(EMB_DIR,emb_fname+'_sub_nums')
base_emb = join(EMB_DIR,emb_fname+'.txt')

with open('data/scmag.pkl','rb') as f:
    Xss = pickle.load(f)

number_set = set(np.array(Xss).flat)
number_set.remove(math.inf)
number_array = [str(int(n)) for n in sorted(number_set)]
number_array.append(str(math.inf))

if os.path.isfile(emb_num_name+'.pickle'):
    with open(emb_num_name+'.pickle','rb') as f:
        number_emb_dict = pickle.load(f)
else:
    number_emb_dict,_ = vocab2vec(number_array, EMB_DIR, emb_num_name, base_emb, ['pickle','npy'])

d = 300
n = len(number_array)

random_emb = True

if random_emb:
    number_emb = torch.randn((d,n))
    P_x = number_emb[:,:-1]
    P_xp = number_emb[:,1:]
else:
    P_x = torch.zeros((d, n - 1))
    P_xp = torch.zeros_like(P_x)
    for i, num_str in enumerate(number_array[:-1]):
        P_x[:,i] = torch.from_numpy(number_emb_dict[num_str])
        P_xp[:,i] = torch.from_numpy(number_emb_dict[number_array[i+1]])

device = 'cuda' if torch.cuda.is_available() else 'cpu'
w = torch.randn(d,requires_grad=True,device=device)
w.data = w.data/torch.norm(w).data

test_init_w = True
if test_init_w:
    aver_num = 20
    aver_acc = 0
    for i in range(aver_num):
        w_init = torch.randn(d, device=device)
        w_init.data = w_init.data / torch.norm(w_init).data
        S_x = torch.matmul(w_init, P_xp - P_x)
        acc = torch.mean((S_x >= 0).float())
        aver_acc += acc.item()
    print(aver_acc/aver_num)

beta = 16
n_epochs = 4000
lr = 0.1

optimizer = torch.optim.Adam([w],lr)
pre_acc = -math.inf

for t in range(n_epochs):

    S_x = torch.matmul(w, P_xp - P_x)
    loss = -torch.mean(soft_indicator(S_x,beta))
    # print(loss)
    optimizer.zero_grad()

    loss.backward()

    acc = torch.mean((S_x>=0).float())
    if acc.item() > pre_acc:
        best_w = w.clone()
        pre_acc = acc
        # print(best_w)
        print('specialized acc: ', acc)
        torch.save(w, 'w_ordering.pt')

    optimizer.step()

S_x = torch.matmul(best_w, P_xp - P_x)
acc = torch.mean((S_x>=0).float())
print('best acc: ',acc)