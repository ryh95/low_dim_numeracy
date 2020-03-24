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
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import train_test_split
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



class SepMagExp(object):

    def __init__(self, exp_name, save_results, exp_data):
        self.name = exp_name
        self.save_results = save_results
        self.exp_data = exp_data

    def run(self):

        iteration = 30000
        name = self.name.split('_')[0]
        if name == 'glove-wiki' or name == 'word2vec-wiki' or name == 'word2vec-giga':
            iteration = 20000

        svc = SVC(kernel='poly', degree=3, gamma=1 / 300, coef0=0, C=1,
                  cache_size=4000, class_weight='balanced', verbose=True, max_iter=iteration)
        start = time.time()
        svc.fit(self.exp_data['X'], self.exp_data['y'])
        print('fit time: ', time.time() - start)

        # use the perpendicular direction that separates the numbers and words
        # to predict magnitude
        sel_pred_mag = svc.decision_function(self.exp_data['sel_X'])
        # evaluate the predicted magnitude with the test in wallace et.al
        error = sqrt(mean_squared_error(self.exp_data['sel_mag'], sel_pred_mag))
        if self.save_results:
            np.save(self.name,error)
        return error


