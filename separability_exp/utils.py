import time
from math import ceil
from os.path import join

import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

from ..config import EMB_DIR
from ..utils import is_number


def parallel_predict(X, predict_func, n_cores):
    n_samples = X.shape[0]
    slices = [(ceil(n_samples * i / n_cores), ceil(n_samples * (i + 1) / n_cores)) for i in range(n_cores)]
    start = time.time()
    y_pred = np.concatenate(Parallel(n_jobs=n_cores)(
        delayed(predict_func)(X[slices[i_core][0]:slices[i_core][1], :]) for i_core in range(n_cores)))
    print('predict time: ', time.time() - start)
    return y_pred

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