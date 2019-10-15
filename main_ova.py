
import time
from math import inf

import torch
from torch import autograd
from torch.utils.data import DataLoader

from dataset import OVADataset

# TODO: check the data, whether the property p is satisfied
from model import OVA_Subspace_Model

emb_fname = 'glove.6B.300d' # or 'random'

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

train_data = OVADataset('data/ovamag_str.pkl',{"emb_fname":emb_fname})

# model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
d = next(iter(train_data.number_emb_dict.values())).numel()
ova_model = OVA_Subspace_Model(dim,d,beta)

optimizer = torch.optim.Adam(ova_model.parameters(),lr)
# SGD performs poorly, the reason is unclear
# optimizer = torch.optim.SGD([W],lr,momentum=0.9)

best_acc = -inf
best_loss = inf

mini_batchs = DataLoader(train_data,batch_size=mini_batch_size,shuffle=True,num_workers=8,pin_memory=True)

acc,loss = ova_model.evaluate(mini_batchs)
print('init specialized acc: ',acc)
# print('init specialized I_hat: ',loss)

for t in range(n_epochs):

    print('epoch number: ', t)
    start = time.time()
    # num_mini_batches = 0
    for i,mini_batch in enumerate(mini_batchs):

        mini_P_x, mini_P_xp, mini_P_xms = mini_batch
        mini_P_x = mini_P_x.to(device) # can set non_blocking=True
        mini_P_xp = mini_P_xp.to(device)
        mini_P_xms = mini_P_xms.to(device)

        dp,dm = ova_model(mini_P_x, mini_P_xp, mini_P_xms)
        loss,acc = ova_model.criterion(dp,dm)

        # print(loss)
        optimizer.zero_grad()

        with autograd.detect_anomaly():
            # avoid nan gradient
            loss.backward()

        if i % 5:
            if acc.item() > best_acc:
                best_W = ova_model.W.data.clone()
                best_acc = acc.item()
                print('specialized acc: ', best_acc)

        optimizer.step()

        ova_model.project()
        # num_mini_batches += 1
    # print(num_mini_batches)
    print("train: ", time.time() - start)

print("Deviation from the constraint: ",torch.norm(best_W.T @ best_W - torch.eye(dim).to(device)).item())
ova_model.W = torch.nn.Parameter(best_W)
evaluate_acc, evaluate_loss = ova_model.evaluate(mini_batchs)
torch.save({
        'beta':beta,
        'dim':dim,
        'n_epochs':n_epochs,
        'mini_batch_size':mini_batch_size,
        'W':ova_model.state_dict(),
        'optimizer_state':optimizer.state_dict(),
        'acc': evaluate_acc
    }, '_'.join(['results',emb_fname,str(dim)])+'.pt')
print('best acc: ', evaluate_acc)
# print('best loss: ',evaluate_loss)