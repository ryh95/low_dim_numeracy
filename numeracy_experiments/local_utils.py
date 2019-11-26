import pickle
import time
from math import ceil, sqrt
from os.path import join
import numpy as np
import skopt
from joblib import Parallel, delayed
from keras import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.metrics import f1_score, mean_squared_error
from sklearn.svm import SVC, SVR
from skopt import gp_minimize
from skopt.space import Categorical, Real, Integer
from tqdm import tqdm
from keras import backend as K

from config import EMB_DIR, DATA_DIR
from utils import is_valid_triple, is_number

def test_on_svm_w(svc,test_dataset,test_femb):
    num_emb_fname = join(EMB_DIR, test_femb + '_' + test_dataset + '_num_emb')
    with open(num_emb_fname + '.pickle', 'rb') as f:
        number_emb_dict = pickle.load(f)
    test_num, test_num_emb = zip(*number_emb_dict.items())
    test_num_emb = np.stack(test_num_emb)
    test_magnitudes = svc.decision_function(test_num_emb)
    test_num_magnitude = {i: j for i, j in zip(test_num, test_magnitudes)}
    with open(join(DATA_DIR, test_dataset + '.pkl'), 'rb') as f:
        X = pickle.load(f)
    print(len(X))
    test_results = []
    for test in X:
        triple = [test_num_magnitude[e] for e in test]
        test_results.append(1 if is_valid_triple(triple) else 0)
    print('acc on proj w: ', np.mean(test_results))

def prepare_separation_data(femb):
    '''

    :param femb: 'skipgram-5.txt'
    :return:
    '''
    number_emb, word_emb = [], []
    print('prepare fitting data...')
    with open(join(EMB_DIR, femb), 'r') as f:
        f.readline()  # skipgram or fasttext
        for line in tqdm(f):
            word, *vec = line.rstrip().split(' ')
            vec = np.array(vec, dtype=float)
            word = word.split('_')[0]  # skipgram
            if is_number(word):
                number_emb.append(vec)
            else:
                word_emb.append(vec)
    #
    X_num = np.stack(number_emb)
    y_num = np.ones(X_num.shape[0], dtype=int)
    X_word = np.stack(word_emb)
    y_word = -np.ones(X_word.shape[0], dtype=int)
    X = np.concatenate([X_num, X_word])
    y = np.concatenate([y_num, y_word])
    print('fitting data has prepared')
    print('number embedding: ',X_num.shape)
    print('word embedding: ',X_word.shape)
    return X,y

def load_num_emb(femb, sel_nums_train, sel_nums_val, sel_nums_test):
    '''
    load number embedding from femb
    embedding in femb should be all number embedding
    :param femb:
    :return:
    '''
    # todo: possible duplicate with utils/preprocess_google_news_skip
    number_emb_train,number_target_train = [],[]
    number_emb_val, number_target_val = [], []
    number_emb_test, number_target_test = [], []
    print('prepare data...')
    sel_nums_train_set = set(sel_nums_train)
    sel_nums_val_set = set(sel_nums_val)
    sel_nums_test_set = set(sel_nums_test)
    with open(join(EMB_DIR, femb+'.txt'), 'r') as f:
        f.readline()
        for line in tqdm(f):
            word, *vec = line.rstrip().split(' ')
            vec = np.array(vec, dtype=float)
            word = word.split('_')[0]
            if word in sel_nums_train_set:
                number_emb_train.append(vec)
                number_target_train.append(float(word))
                sel_nums_train_set.remove(word)
            elif word in sel_nums_val_set:
                number_emb_val.append(vec)
                number_target_val.append(float(word))
                sel_nums_val_set.remove(word)
            elif word in sel_nums_test_set:
                number_emb_test.append(vec)
                number_target_test.append(float(word))
                sel_nums_test_set.remove(word)

    X_train = np.stack(number_emb_train)
    y_train = np.array(number_target_train)
    X_val = np.stack(number_emb_val)
    y_val = np.array(number_target_val)
    X_test = np.stack(number_emb_test)
    y_test = np.array(number_target_test)
    return X_train,y_train,X_val,y_val,X_test,y_test

def parallel_predict(X, predict_func, n_cores):
    n_samples = X.shape[0]
    slices = [(ceil(n_samples * i / n_cores), ceil(n_samples * (i + 1) / n_cores)) for i in range(n_cores)]
    start = time.time()
    y_pred = np.concatenate(Parallel(n_jobs=n_cores)(
        delayed(predict_func)(X[slices[i_core][0]:slices[i_core][1], :]) for i_core in range(n_cores)))
    print('predict time: ', time.time() - start)
    return y_pred

class Minimizer(object):

    def __init__(self,base_workspace, optimize_types, mini_func):
        self.base_workspace = base_workspace
        self.mini_func = mini_func
        self.optimize_types = optimize_types

    def objective(self,feasible_point):
        optimize_workspace = {type: type_values for type, type_values in zip(self.optimize_types, feasible_point)}

        # combine two workspace
        workspace = {**self.base_workspace, **optimize_workspace}
        model = workspace['model']
        X = workspace['fitting_X']
        y = workspace['fitting_y']

        model.set_params(**optimize_workspace)
        start = time.time()
        model.fit(X, y)
        print('fit time: ',time.time()-start)
        y_pred = parallel_predict(X, model.predict, 10)
        return -f1_score(y, y_pred)

    def minimize(self,space,**min_args):

        return self.mini_func(self.objective, space, **min_args)

class MagnitudeAxisMinimizer(Minimizer):

    def objective(self,feasible_point):
        optimize_workspace = {type: type_values for type, type_values in zip(self.optimize_types, feasible_point)}

        X_train = self.base_workspace['X_train']
        y_train = self.base_workspace['y_train']
        X_val = self.base_workspace['X_val']
        y_val = self.base_workspace['y_val']

        if hasattr(self,'model_fixed_params'):
            model = self.model(**optimize_workspace,**self.model_fixed_params)
        else:
            model = self.model(**optimize_workspace)

        try:
            if isinstance(model,KerasRegressor):
                K.clear_session()
                model.fit(X_train,y_train,verbose=0)
            else:
                model.fit(X_train,y_train)
            y_pred_val = model.predict(X_val)
            error = mean_squared_error(y_val, y_pred_val)
        except np.linalg.LinAlgError:
            error = 1e+30
        return error

class SeparableExperiments(object):

    def __init__(self,exp_name,save_results,exp_data):
        self.name = exp_name
        self.save_results = save_results
        self.exp_data = exp_data

    def run(self):
        # space = [Real(1e-6, 1e+6, prior='log-uniform')]
        # optimize_types = ['C']
        # x0=[1.0]
        # model = SVC(kernel='poly',degree=3,gamma=1/(300*exp_data['X'].var()),coef0=0,
        #   cache_size=8000,class_weight='balanced',verbose=True,max_iter=15000)
        # fitting_X = exp_data['X']
        # fitting_y = exp_data['y']
        # base_workspace = {'model':model,'fitting_X':fitting_X,'fitting_y':fitting_y}
        # minimizer = Minimizer(base_workspace,optimize_types,gp_minimize)
        # res_gp = minimizer.mini_func(space,n_calls=11,verbose=True,x0=x0,n_jobs=-1)
        # if self.save_results:
        #     skopt.dump(res_gp,self.name+'.pkl',store_objective=False)
        # return -res_gp.fun

        # word2vec-wiki iter: 20000/ word2vec-giga iter: 20000
        # glove-wiki iter: 20000/ glove-giga iter: 30000
        # fasttext-wiki iter: 30000/ fasttext-giga iter: 30000

        # cache size: 4000
        iteration = 30000
        name = self.name.split('_')[0]
        if name == 'glove-wiki' or name == 'word2vec-wiki' or name == 'word2vec-giga':
            iteration = 20000

        svc = SVC(kernel='poly', degree=3, gamma=1 / 300, coef0=0, C=1,
                  cache_size=4000, class_weight='balanced', verbose=True, max_iter=iteration)
        start = time.time()
        svc.fit(self.exp_data['X'], self.exp_data['y'])
        print('fit time: ', time.time() - start)
        y_pred = parallel_predict(self.exp_data['X'], svc.predict, 10)
        f1 = f1_score(self.exp_data['y'], y_pred)
        print(self.name,f1)
        return f1

class MagnitudeExperiments2(object):

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