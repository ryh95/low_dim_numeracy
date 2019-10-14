
import time
from math import inf

import torch
from torch import autograd
from torch.utils.data import DataLoader

from dataset import OVADataset
from utils import soft_indicator

# TODO: check the data, whether the property p is satisfied

emb_fname = 'glove.6B.300d' # or 'random'
d = 300 # dimension of the word embedding

# calculate the original accuracy
beta = 6
dim = 64
n_epochs = 50
lr = 0.005
mini_batch_size = 512

# Dp = LA.norm(P_x - P_xp,axis=1)
# Dm = LA.norm(P_x[:,:,None] - P_xms,axis=1).min(axis=1)
#
# I_hat = float(torch.sum(soft_indicator(torch.tensor(Dm - Dp, dtype=torch.float), beta=beta)))/batch_size
# acc = sum(Dp <= Dm) / batch_size
# print('original acc: ',acc)
# print('original I_hat: ', -I_hat)

# model
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# w = torch.nn.Parameter(torch.randn(d))
W = torch.randn((dim,d),requires_grad=True,device=device)
# w = torch.randn(d,requires_grad=True)
# w.data = w.data/torch.norm(w).data
torch.nn.init.orthogonal_(W)
W.data = W.T.data # col orthognol


train_data = OVADataset('data/ovamag_str.pkl',{"emb_fname":emb_fname})

optimizer = torch.optim.Adam([W],lr)
# SGD performs poorly, the reason is unclear
# optimizer = torch.optim.SGD([W],lr,momentum=0.9)

init_acc = -inf
init_loss = inf


def evaluate_w(train_data,mini_batch_size,W):
    mini_batchs = DataLoader(train_data,batch_size=mini_batch_size,shuffle=True,num_workers=8,pin_memory=True)
    loss,acc = 0,0
    start = time.time()
    for mini_batch in mini_batchs:
        mini_P_x, mini_P_xp, mini_P_xms = mini_batch
        mini_P_x = mini_P_x.to(device)
        mini_P_xp = mini_P_xp.to(device)
        mini_P_xms = mini_P_xms.to(device)

        dp = torch.norm(torch.matmul(mini_P_x-mini_P_xp,W.data),dim=1)
        dm = torch.min(torch.norm(torch.matmul(W.data.T,mini_P_x[:,:,None]-mini_P_xms),dim=1),dim=1)[0]
        #
        objs = soft_indicator(dm-dp,beta)
        loss += -torch.sum(objs).item()
        acc += torch.sum(dp<=dm).item()
    print("evaluate: ",time.time()-start)
    batch_size = len(train_data)
    return acc/batch_size,loss/batch_size

acc,loss = evaluate_w(train_data,mini_batch_size,W)
print('init specialized acc: ',acc)
print('init specialized I_hat: ',loss)

mini_batchs = DataLoader(train_data,batch_size=mini_batch_size,shuffle=True,num_workers=8,pin_memory=True)

for t in range(n_epochs):

    print('epoch number: ', t)
    start = time.time()
    for i,mini_batch in enumerate(mini_batchs):

        mini_P_x, mini_P_xp, mini_P_xms = mini_batch
        mini_P_x = mini_P_x.to(device) # can set non_blocking=True
        mini_P_xp = mini_P_xp.to(device)
        mini_P_xms = mini_P_xms.to(device)

        dp = torch.norm(torch.matmul(mini_P_x-mini_P_xp,W),dim=1)
        dm = torch.min(torch.norm(torch.matmul(W.T,mini_P_x[:,:,None]-mini_P_xms),dim=1),dim=1)[0]

        objs = soft_indicator(dm-dp,beta)
        loss = -torch.mean(objs)
        acc = torch.mean((dp <= dm).float()).item() # mini-batch acc
        # print(loss)
        optimizer.zero_grad()

        with autograd.detect_anomaly():
            # avoid nan gradient
            loss.backward()

        if i % 5:
            if acc > init_acc:
                best_w = W.clone()
                init_acc = acc
                # print(best_w)
                print('specialized acc: ', acc)

        optimizer.step()

        # project the variables back to the feasible set
        # w.data = w.data / torch.norm(w).data

        # find the nearest col orthogonal matrix
        # ref: http://people.csail.mit.edu/bkph/articles/Nearest_Orthonormal_Matrix.pdf
        # ref: https://math.stackexchange.com/q/2500881
        # ref: https://math.stackexchange.com/a/2215371
        u,s,v = torch.svd(W.data)
        W.data = u @ v.T
        assert not torch.isnan(W.data).any(),'W has nan values'

    print("train: ", time.time() - start)
    torch.save(best_w, 'w_ova.pt')

# best_w = torch.load('w.pt')

best_acc,best_loss = evaluate_w(train_data,mini_batch_size,best_w)
print('best acc: ',best_acc)
print('best specialized I_hat: ',best_loss)