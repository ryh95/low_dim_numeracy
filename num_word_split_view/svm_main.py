from os.path import join

import torch

import numpy as np
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from skopt import gp_minimize
from skopt.callbacks import CheckpointSaver
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from torch.utils.data import DataLoader

from tqdm import tqdm

from config import EMB, EMB_DIR
from experiments.local_utils import load_dataset, init_evaluate
from utils import is_number

# collect data
# X: embedding, Y: label(number or not)
# emb_type = 'random'
emb = 'skipgram-5'
number_emb,word_emb = [],[]
with open(join(EMB_DIR,emb), 'r') as f:
    if 'skipgram' in emb:
        f.readline() # skipgram-5.txt
    for line in tqdm(f):
        word, *vec = line.rstrip().split(' ')
        vec = np.array(vec, dtype=float)
        if 'skipgram' in emb:
            word = word.split('_')[0] # skipgram-5.txt
        if is_number(word):
            number_emb.append(vec)
        else:
            word_emb.append(vec)
#
X_num = np.stack(number_emb)
y_num = np.ones(X_num.shape[0],dtype=int)
X_word = np.stack(word_emb)
y_word = -np.ones(X_word.shape[0],dtype=int)
X = np.concatenate([X_num,X_word])
y = np.concatenate([y_num,y_word])

print(X_num.shape)
print(X_word.shape)

# if emb_type == 'pre-train':
#
#
# elif emb_type == 'random':


# preprocess the data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# avoud data copy
if not X.flags['C_CONTIGUOUS'] or X.dtype != np.float64:
    print('X may be copied')

# svm classifier
# space = [Real(1e-6, 1e+6, prior='log-uniform',name='C'),
#          Categorical(['linear', 'poly', 'sigmoid'],name='kernel'),
#          Integer(1,8,name='degree'),
#          Real(1e-6, 1e+1, prior='log-uniform',name='gamma'),
#          Real(-10, 10,name='coef0')]

svc = SVC(cache_size=7000,class_weight='balanced',verbose=True)

# @use_named_args(space)
# def objective(**params):
#     svc.set_params(**params)
#     svc.fit(X,y)
#     y_pred = svc.predict(X)
#     return -f1_score(y,y_pred)

# checkpoint_callback = CheckpointSaver('svm_checkpoint.pkl', store_objective=False)
# x0 = [1.0,'poly',3,1/(300*X.var()),0.0]
# res_gp = gp_minimize(objective,space,n_calls=50,callback=[checkpoint_callback],verbose=True,x0=x0)

params = {'C':1.0,'kernel':'poly','degree':3,'gamma':1/(300*X.var()),'coef0':0.0}
svc.set_params(**params)
svc.fit(X,y)
y_pred = svc.predict(X)
np.save('svm_default_predict.npy',y_pred)
print(f1_score(y,y_pred))

dataset = load_dataset('scmag_str', {'emb_fname':'skipgram-5_num'})
# poly kernel

# def kernel_sim(x,y,gamma,r,d):
#     if len(y.size()) == 3:
#         # ova
#         (gamma * torch.bmm(x.unsqueeze(1), y) + r) ** d
#     else:

kernel_sim = lambda x,y,gamma,r,d: (gamma*torch.bmm(x.unsqueeze(1),y.unsqueeze(2))+r)**d
# distance function, cosine distance
distance_func = lambda x,y, \
                       gamma=params['gamma'],r=params['coef0'],d=params['degree']: \
    1 - kernel_sim(x,y,gamma,r,d) / ((kernel_sim(x,x,gamma,r,d) ** 0.5)*(kernel_sim(y,y,gamma,r,d) ** 0.5))
acc = init_evaluate(dataset,distance_func)
print(acc)
# print('best f1 score: %f' % res_gp.fun)
# print('best params: ', res_gp.x)