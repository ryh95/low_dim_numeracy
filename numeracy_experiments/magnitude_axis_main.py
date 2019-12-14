from math import sqrt
from os.path import splitext, join

from keras import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import numpy as np
from skopt import gp_minimize, dump
from skopt.callbacks import CheckpointSaver
from skopt.space import Real, Categorical, Integer
from termcolor import colored

from config import VOCAB_DIR
from numeracy_experiments.local_utils import load_num_emb, MagnitudeExperiments2

fembs = ['word2vec-wiki','word2vec-giga','glove-wiki','glove-giga','fasttext-wiki','fasttext-giga','random']
test_models = ['ridge','kernel_ridge','kernel_ridge_separation']
sel_nums = np.load(join(VOCAB_DIR,'inter_nums.npy'))
n_trials = 10
results = np.zeros((n_trials,len(fembs),len(test_models)))

for i in range(n_trials):

    sel_nums_train,sel_nums_test = train_test_split(sel_nums,test_size=0.2)
    sel_nums_train,sel_nums_val = train_test_split(sel_nums_train,test_size=0.1875)

    for j,femb in enumerate(fembs):

        if femb == 'random':
            X_train = np.random.randn(len(sel_nums_train),300)
            y_train = np.array(sel_nums_train,dtype=float)
            X_val = np.random.randn(len(sel_nums_val), 300)
            y_val = np.array(sel_nums_val, dtype=float)
            X_test = np.random.randn(len(sel_nums_test), 300)
            y_test = np.array(sel_nums_test, dtype=float)
        else:
            X_ys = []
            for sel_nums_split in [sel_nums_train,sel_nums_val,sel_nums_test]:
                X_ys += list(load_num_emb(femb,sel_nums_split))
            X_train,y_train,X_val,y_val,X_test,y_test = X_ys

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

        print(X_train.shape, X_val.shape, X_test.shape)

        # train,val,test split
        # train:0.65, val:0.15 test:0.2

        base_workspace = {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test,
            'X': np.concatenate([X_train, X_val]),
            'y': np.concatenate([y_train, y_val])
        }

        for k,model in enumerate(test_models):

            exp_name = femb+'_'+model+'_'+str(i)
            exp = MagnitudeExperiments2(exp_name, False, {'model':model, 'base_workspace':base_workspace})
            error = exp.run()
            # save results
            results[i][j][k] = error

            print(colored(femb+' '+model+' test root mse: ','red'), error)

    print(results[i,:,:])

np.save('results',results)