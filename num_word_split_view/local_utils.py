import pickle
from os.path import join
import numpy as np
import torch
from sklearn.metrics import f1_score
from tqdm import tqdm

from config import EMB_DIR, DATA_DIR
from utils import is_valid_triple, is_number
import torch.nn.functional as F

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

class KernelDistance(object):

    def __int__(self,*kernel_setting):

        self.kernel_setting = kernel_setting

    def kernel_sim(self,x,y,kernel_type,gamma,r,d):
        if len(x.size()) == 2 and len(y.size()) == 2:
            # x: B,d y: B,d
            inner_product = torch.sum(x * y,dim=1)
        elif len(x.size()) == 3 and len(y.size()) == 3:
            # x: B,d,n-2 y: B,d,n-2
            inner_product = torch.sum(x * y,dim=1)
        elif len(x.size()) == 2 and len(y.size()) == 3:
            # x: B,d y: B,d,n-2
            inner_product = (x[:,None,:] @ y).squeeze()
        else:
            assert False
        if kernel_type == 'poly':
            return (gamma * inner_product + r) ** d
        elif kernel_type == 'sigmoid':
            return torch.tanh(gamma * inner_product + r)

    def distance_func(self,x,y):
        _,kernel_type,d,gamma,r = self.kernel_setting
        k_xy = self.kernel_sim(x, y, kernel_type, gamma, r, d)
        k_xx = self.kernel_sim(x, x, kernel_type, gamma, r, d)
        if len(y.size()) == 3:
            k_xx = k_xx[:,None]
        k_yy = self.kernel_sim(y, y, kernel_type, gamma, r, d)
        return 1 - k_xy / ((k_xx ** 0.5) * (k_yy ** 0.5))

def cosine_distance(x,y):
    if len(y.size()) == 3:
        x = x[:,:,None]
    return 1 - F.cosine_similarity(x, y)

def prepare_fitting_data(femb):
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
