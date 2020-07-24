from math import sqrt
import numpy as np
import skopt
from keras import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.decomposition import PCA
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from skopt import gp_minimize
from skopt.space import Real, Categorical, Integer
from thundersvm import SVC

from separability_exp.helper.utils import prepare_separation_data
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
        elif model == 'pca':
            if not hasattr(self,'pca_component'):
                pca = PCA(n_components=1)
                _, X_nums,*_ = prepare_separation_data(self.name.split('_')[0]+'.txt')
                pca.fit(X_nums)
                # pca.fit(np.concatenate([base_workspace['X'],base_workspace['X_test']]))
                self.pca_component = pca.components_  # 1xd
            error = self.evaluate_w(base_workspace['X_test'], self.pca_component, base_workspace['y_test'])
            return error
        elif model == 'proj_pca':
            if not hasattr(self,'proj_pca_component'):
                X,X_nums,y_label,_ = prepare_separation_data(self.name.split('_')[0]+'.txt')
                svc = SVC(kernel='linear', degree=3, gamma=1 / 300, coef0=0, C=1,
                          cache_size=4000, class_weight='balanced', verbose=True)

                svc.fit(X, y_label)
                beta = svc.coef_  # 1xd
                # X_pred = svc.decision_function(X) nx1
                # X_nums = np.concatenate([base_workspace['X'],base_workspace['X_test']])
                X_proj = X_nums - ((svc.decision_function(X_nums) / beta @ beta.T) @ beta)  # nxd

                pca = PCA(n_components=1)
                pca.fit(X_proj)
                self.proj_pca_component = pca.components_  # 1xd
            error = self.evaluate_w(base_workspace['X_test'], self.proj_pca_component, base_workspace['y_test'])
            return error
        elif model == 'kernel_proj_pca':
            pass
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

    def evaluate_w(self, X_nums, u, nums):
        '''

        :param X_nums: number embeddings
        :param w: the direction vector
        :param nums: gold numbers
        :return:
        '''
        # todo: check whether u is unit vector
        pred_nums = (X_nums @ u.T).squeeze()  # s
        w = np.ones(X_nums.shape[0])  # s
        t = ((nums - pred_nums).T @ w) / (w.T @ w)
        res = np.sqrt(
            (nums - t * w - pred_nums).T @ (nums - t * w - pred_nums) / nums.shape[0])  # RMSE of the best shift
        return res