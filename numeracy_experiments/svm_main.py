import pickle
import time
from os.path import join

import torch

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils import shuffle
from skopt import gp_minimize
from skopt.callbacks import CheckpointSaver
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from torch.utils.data import DataLoader

from tqdm import tqdm

from config import EMB, EMB_DIR, DATA_DIR
from subspace_magnitude_experiments.local_utils import load_dataset, init_evaluate
from numeracy_experiments.local_utils import KernelDistance, cosine_distance, prepare_separation_data, Minimizer, \
    parallel_predict
from utils import is_number, is_valid_triple

# collect data
# X: embedding, Y: label(number or not)
# emb_type = 'random'
femb = 'skipgram-5.txt'
X,y = prepare_separation_data(femb)

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

# subspace_magnitude_experiments on number word separability
params = {
    'kernel': 'poly',
    'C':1.0,
    'degree': 3,
    'gamma': 1/(300*X.var()),
    'coef0': 0.0
}
svc = SVC(cache_size=8000,class_weight='balanced',verbose=True,max_iter=20000)
svc.set_params(**params)
start = time.time()
svc.fit(X,y)
print('fit time: ',time.time()-start)
y_pred = parallel_predict(X,svc.predict,10)
np.save('svm_poly_num_word_predict.npy',y_pred)
print('number word f1: ',f1_score(y,y_pred))

# random choose some token in emb as numbers and refit the model
# to test whether this kernel can seperate arbitray tokens
f1s = []
for _ in range(10):
    svc = SVC(cache_size=8000, class_weight='balanced', verbose=True, max_iter=20000)
    svc.set_params(**params)
    start = time.time()
    X = shuffle(X)
    svc.fit(X,y)
    print('fit time: ',time.time()-start)
    y_pred = parallel_predict(X, svc.predict, 10)
    f1 = f1_score(y,y_pred)
    print('random number word f1: ',f1)
    f1s.append(f1)
print('average random number word f1: ',np.mean(f1s))

# select best kernel that can seperate the number and word
# report the best classification results on that kernel
# space = [Real(1e-6, 1e+6, prior='log-uniform'),
#          Categorical(['poly', 'sigmoid']),
#          Integer(1,8),
#          Real(1e-6, 1e+1, prior='log-uniform'),
#          Real(-10, 10)]
# base_workspace={
#     'model': SVC(cache_size=7000,class_weight='balanced',verbose=True),
#     'fitting_X': X,
#     'fitting_y': y,
# }
# optimize_types = ['C','kernel','degree','gamma','coef0']
# checkpoint_callback = CheckpointSaver('svm_checkpoint.pkl', store_objective=False)
# x0 = [1.0,'sigmoid',3,1/(300*X.var()),0.0]
# minimizer = Minimizer(base_workspace,optimize_types,gp_minimize)
# res_gp = minimizer.mini_func(space,n_calls=11,callback=[checkpoint_callback],verbose=True,x0=x0)
# params = {'C':res_gp.x[0],'kernel':res_gp.x[1],'degree':res_gp.x[2],'gamma':res_gp.x[3],'coef0':res_gp.x[4]}
# svc = SVC(cache_size=7000,class_weight='balanced',verbose=True)
# svc.set_params(**params)
# svc.fit(X,y)
# y_pred = svc.predict(X)
# np.save('svm_best_predict.npy',y_pred)
# print('best f1: ',f1_score(y,y_pred))

test_dataset = 'scmag_str'
test_femb = 'skipgram-5_num'
dataset = load_dataset(test_dataset, {'emb_fname':test_femb})

# evaluate embs in the original space
print('acc in original space: ',init_evaluate(dataset,cosine_distance))

# evaluate embs in the best hilbert space
# kd_object = KernelDistance(res_gp.x)
# acc = init_evaluate(dataset,kd_object.distance_func)
# print('acc in phi space: ',acc)