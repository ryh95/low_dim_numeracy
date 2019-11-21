from math import sqrt
from os.path import splitext

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
from numeracy_experiments.local_utils import load_num_emb, MagnitudeExperiments2

# fembs = ['skipgram-2_num','skipgram-5_num','glove.6B.300d','glove.840B.300d','crawl-300d-2M-subword_num','wiki-news-300d-1M-subword_num']
fembs = ['skipgram-2_num']
for femb in fembs:

    X,y = load_num_emb(femb)

    # filter the data due to the skewed magnitude distribution
    filters = (y>0)&(y<10000)
    X = X[filters,:]
    y = y[filters]

    # train,val,test split
    # train:0.65, val:0.15 test:0.2
    run_random = True
    test_models = ['ridge','kernel_ridge']
    n_trials = 10
    if run_random:
        results = np.zeros((2,len(test_models),n_trials))
    else:
        results = np.zeros((1, len(test_models), n_trials))

    for k in range(n_trials):


        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1875)

        print(X_train.shape,X_val.shape,X_test.shape)

        # preprocess the data
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        base_workspace={
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': scaler.transform(X_test),
            'y_test': y_test,
            'X': np.concatenate([X_train,X_val]),
            'y': np.concatenate([y_train,y_val])
        }
        base_workspaces = [base_workspace]

        if run_random:
            X_train = np.random.rand(*X_train.shape)
            X_val = np.random.rand(*X_val.shape)
            X_test = np.random.rand(*X_test.shape)
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)
            random_workspace = {
                'X_train': X_train,
                'y_train': y_train,
                'X_val': X_val,
                'y_val': y_val,
                'X_test': scaler.transform(X_test),
                'y_test': y_test,
                'X': np.concatenate([X_train, X_val]),
                'y': np.concatenate([y_train, y_val])
            }
            base_workspaces.append(random_workspace)

        for i,base_workspace in enumerate(base_workspaces):

            for j,model in enumerate(test_models):
                emb_type = femb+'_'+'random' if i == 1 else femb
                exp_name = emb_type+'_'+model+'_'+str(k)
                exp = MagnitudeExperiments2(exp_name, True, {'model':model, 'base_workspace':base_workspace})
                error = exp.run()
                # save results
                results[i][j][k] = error

                if i == 1:
                    print(colored(femb + ' random ','blue'),colored(model+' test root mse: ','red'), error)
                else:
                    print(colored(femb+' '+model+' test root mse: ','red'), error)

    np.save(femb + '_results',results)