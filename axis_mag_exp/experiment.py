from math import sqrt

import skopt
from keras import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from skopt import gp_minimize
from skopt.space import Real, Categorical, Integer

from .minimizer import MagnitudeAxisMinimizer


class MagnitudeAxisExperiments(object):

    def __init__(self, exp_name, save_results, exp_data):
        self.name = exp_name
        self.save_results = save_results
        self.exp_data = exp_data

    def run(self):

        model = self.exp_data['model']
        base_workspace = self.exp_data['base_workspace']

        if model == 'ridge':
            space = [Real(1e-3, 1e+3, prior='log-uniform')]
            optimize_types = ['alpha']
            minimizer = MagnitudeAxisMinimizer(base_workspace, optimize_types, gp_minimize)
            minimizer.model = Ridge
            x0 = [1.0]
        elif model == 'kernel_ridge':
            space = [Categorical(['poly', 'rbf', 'sigmoid']),
                     Real(1e-3, 1e+3, prior='log-uniform'),
                     Integer(1, 8),
                     Real(1e-6, 1e+1, prior='log-uniform'),
                     Real(-10, 10)
                     ]
            optimize_types = ['kernel', 'alpha', 'degree', 'gamma', 'coef0']
            minimizer = MagnitudeAxisMinimizer(base_workspace, optimize_types, gp_minimize)
            minimizer.model = KernelRidge
            x0 = ['poly', 1.0, 3, 1 / 300, 0]
        elif model == 'kernel_ridge_separation':
            space = [Real(1e-3, 1e+3, prior='log-uniform')]
            optimize_types = ['alpha']
            minimizer = MagnitudeAxisMinimizer(base_workspace, optimize_types, gp_minimize)
            minimizer.model_fixed_params = {'kernel': 'poly', 'degree': 3, 'gamma': 1 / 300, 'coef0': 0}
            minimizer.model = KernelRidge
            x0 = [1.0]
        elif model == 'nn':
            space = [Integer(16, 256),
                     Real(1e-5, 1, prior='log-uniform'),
                     ]
            optimize_types = ['n_hidden_units', 'lr']
            minimizer = MagnitudeAxisMinimizer(base_workspace, optimize_types, gp_minimize)
            minimizer.model_fixed_params = {'build_fn': self.build_nn, 'epochs': 20, 'batch_size': 256}
            minimizer.model = KerasRegressor
            x0 = [64, 0.001]
        elif model == 'kernel_svm':
            space = [Categorical(['poly', 'rbf', 'sigmoid']),
                     Real(1e-3, 1e+3, prior='log-uniform'),
                     Integer(1, 8),
                     Real(1e-6, 1e+1, prior='log-uniform'),
                     Real(-10, 10)
                     ]
            optimize_types = ['kernel', 'C', 'degree', 'gamma', 'coef0']
            minimizer = MagnitudeAxisMinimizer(base_workspace, optimize_types, gp_minimize)
            minimizer.model_fixed_params = {'cache_size': 8000, 'max_iter': 10000}
            minimizer.model = SVR
            x0 = ['poly', 1.0, 3, 1 / 300, 0]
        else:
            assert False

        res = minimizer.minimize(space, n_calls=40, verbose=True, x0=x0)
        if self.save_results:
            skopt.dump(res, self.name+'.pkl', store_objective=False)

        params = {type: v for type, v in zip(minimizer.optimize_types, res.x)}
        if hasattr(minimizer, 'model_fixed_params'):
            params = {**params, **minimizer.model_fixed_params}
        error = self.fit_test_best_model(minimizer.model,
                                    base_workspace['X'], base_workspace['y'], base_workspace['X_test'],
                                    base_workspace['y_test'], **params)
        return error

    def fit_test_best_model(self,model, X, y, X_test, y_test, **best_params):
        model = model(**best_params)
        if isinstance(model, KerasRegressor):
            model.fit(X, y, verbose=0)
        else:
            model.fit(X, y)
        y_test_pred = model.predict(X_test)
        error = sqrt(mean_squared_error(y_test, y_test_pred))
        return error

    def build_nn(self,n_hidden_units=64,lr=0.001):
        model = Sequential()
        model.add(Dense(n_hidden_units, activation='relu', input_dim=300))
        model.add(Dropout(0.1))
        model.add(Dense(1))
        adam = Adam(learning_rate=lr)
        model.compile(optimizer=adam,
                      loss='mse')
        return model