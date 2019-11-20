from math import sqrt

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
from numeracy_experiments.local_utils import prepare_magnitude_data, MagnitudeAxisMinimizer, build_nn, \
    fit_test_best_model

femb = 'skipgram-5.txt'

X,y = prepare_magnitude_data(femb)

# filter the data due to the skewed magnitude distribution
filters = y<10000
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
            if model == 'ridge':
                space = [Real(1e-3, 1e+3, prior='log-uniform')]
                optimize_types = ['alpha']
                minimizer = MagnitudeAxisMinimizer(base_workspace, optimize_types, gp_minimize)
                minimizer.model = Ridge
                x0 = [1.0]
            elif model == 'kernel_ridge':
                space = [Categorical(['poly','rbf','sigmoid']),
                         Real(1e-3, 1e+3, prior='log-uniform'),
                         Integer(1, 8),
                         Real(1e-6, 1e+1, prior='log-uniform'),
                         Real(-10, 10)
                         ]
                optimize_types = ['kernel', 'alpha', 'degree', 'gamma', 'coef0']
                minimizer = MagnitudeAxisMinimizer(base_workspace, optimize_types, gp_minimize)
                minimizer.model = KernelRidge
                x0 = ['poly', 1.0, 3, 1 / (300 * X_train.var()), 0]
            elif model == 'nn':
                space = [Integer(16, 256),
                         Real(1e-5, 1, prior='log-uniform'),
                         ]
                optimize_types = ['n_hidden_units', 'lr']
                minimizer = MagnitudeAxisMinimizer(base_workspace, optimize_types, gp_minimize)
                minimizer.model_fixed_params = {'build_fn':build_nn,'epochs': 20, 'batch_size': 256}
                minimizer.model = KerasRegressor
                x0 = [64, 0.001]
            elif model == 'kernel_svm':
                space = [Categorical(['poly','rbf','sigmoid']),
                         Real(1e-3, 1e+3, prior='log-uniform'),
                         Integer(1, 8),
                         Real(1e-6, 1e+1, prior='log-uniform'),
                         Real(-10, 10)
                         ]
                optimize_types = ['kernel', 'C', 'degree', 'gamma', 'coef0']
                minimizer = MagnitudeAxisMinimizer(base_workspace, optimize_types, gp_minimize)
                minimizer.model_fixed_params = {'cache_size': 8000, 'max_iter': 10000}
                minimizer.model = SVR
                x0 = ['poly', 1.0, 3, 1 / (300 * X_train.var()), 0]
            else:
                assert False

            if i == 1:
                fcheckpoint = 'random_'+model + '_checkpoint.pkl'
                fresults = 'random_'+model+'_results.pkl'
            else:
                fcheckpoint = model + '_checkpoint.pkl'
                fresults = model + '_results.pkl'

            checkpoint_callback = CheckpointSaver(fcheckpoint, store_objective=False)
            res = minimizer.minimize(space, n_calls=70, callback=[checkpoint_callback], verbose=True, x0=x0)
            dump(res, fresults, store_objective=False)

            params = {type: v for type, v in zip(minimizer.optimize_types, res.x)}
            if hasattr(minimizer,'model_fixed_params'):
                params = {**params,**minimizer.model_fixed_params}
            error = fit_test_best_model(minimizer.model,
                        base_workspace['X'], base_workspace['y'], base_workspace['X_test'], base_workspace['y_test'], **params)

            # save results
            results[i][j][k] = error

            if i == 1:
                print(colored('random ','blue'),colored(model+' test root mse: ','red'), error)
            else:
                print(colored(model+' test root mse: ','red'), error)

print(np.mean(results,axis=-1))
np.save('results',results)