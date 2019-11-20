import pickle
import time
from math import ceil, sqrt
from os.path import join
import numpy as np
import torch
from joblib import Parallel, delayed
from keras import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.metrics import f1_score, mean_squared_error
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
        if 'skipgram' in femb:
            f.readline()  # skipgram-5.txt
        for line in tqdm(f):
            word, *vec = line.rstrip().split(' ')
            vec = np.array(vec, dtype=float)
            if 'skipgram' in femb:
                word = word.split('_')[0]  # skipgram-5.txt
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

def prepare_magnitude_data(femb):
    '''
    prepare data for regression, X, number embedding, y, number value(represents magnitude)
    :param femb:
    :return:
    '''
    # todo: possible duplicate with utils/preprocess_google_news_skip
    number_emb,number_target = [],[]
    print('prepare fitting data...')
    # n_integer = 0
    with open(join(EMB_DIR, femb), 'r') as f:
        if 'skipgram' in femb:
            f.readline()  # skipgram-5.txt
        for line in tqdm(f):
            word, *vec = line.rstrip().split(' ')
            vec = np.array(vec, dtype=float)
            if 'skipgram' in femb:
                word = word.split('_')[0]  # skipgram-5.txt
            if is_number(word):
                if np.isinf(float(word)): continue
                # if float(word).is_integer(): n_integer += 1
                number_emb.append(vec)
                number_target.append(float(word))

    X = np.stack(number_emb)
    y = np.array(number_target)
    print('number embedding: ',X.shape)
    return X,y

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
        model.fit(X, y)
        y_pred = model.predict(X)
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

def build_nn(n_hidden_units=64,lr=0.001):
    model = Sequential()
    model.add(Dense(n_hidden_units, activation='relu', input_dim=300))
    model.add(Dropout(0.1))
    model.add(Dense(1))
    adam = Adam(learning_rate=lr)
    model.compile(optimizer=adam,
                  loss='mse')
    return model

def fit_test_best_model(model,X,y,X_test,y_test,**best_params):
    model = model(**best_params)
    if isinstance(model,KerasRegressor):
        model.fit(X, y, verbose=0)
    else:
        model.fit(X,y)
    y_test_pred = model.predict(X_test)
    error = sqrt(mean_squared_error(y_test, y_test_pred))
    return error

def parallel_predict(X,predict_func,n_cores):
    n_samples = X.shape[0]
    slices = [(ceil(n_samples * i / n_cores), ceil(n_samples * (i + 1) / n_cores)) for i in range(n_cores)]
    start = time.time()
    y_pred = np.concatenate(Parallel(n_jobs=n_cores)(
        delayed(predict_func)(X[slices[i_core][0]:slices[i_core][1], :]) for i_core in range(n_cores)))
    print('predict time: ', time.time() - start)
    return y_pred