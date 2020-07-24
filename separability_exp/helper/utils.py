import time
from math import ceil
from os.path import join

import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

from config import EMB_DIR
from utils import is_number


def parallel_predict(X, predict_func, n_cores):
    n_samples = X.shape[0]
    slices = [(ceil(n_samples * i / n_cores), ceil(n_samples * (i + 1) / n_cores)) for i in range(n_cores)]
    start = time.time()
    y_pred = np.concatenate(Parallel(n_jobs=n_cores)(
        delayed(predict_func)(X[slices[i_core][0]:slices[i_core][1], :]) for i_core in range(n_cores)))
    print('predict time: ', time.time() - start)
    return y_pred

def prepare_separation_data(femb,sample_ratio=1):
    '''

    :param femb: 'skipgram-5.txt'
    :return:
    '''
    number_emb, word_emb, nums = [], [], []
    print('prepare fitting data...')
    with open(join(EMB_DIR, femb), 'r') as f:
        f.readline()  # skipgram or fasttext
        for line in tqdm(f):
            word, *vec = line.rstrip().split(' ')
            vec = np.array(vec, dtype=float)
            word = word.split('_')[0]  # skipgram
            if is_number(word):
                number_emb.append(vec)
                nums.append(float(word))
            else:
                word_emb.append(vec)
    #
    X_num = np.stack(number_emb)
    nums = np.stack(nums)
    y_num = np.ones(X_num.shape[0], dtype=int)
    X_word = np.stack(word_emb)
    # y_word = -np.ones(X_word.shape[0], dtype=int)
    y_word = np.zeros(X_word.shape[0], dtype=int)
    if sample_ratio<=1:
        n_num = X_num.shape[0]
        ind = np.random.choice(n_num,ceil(sample_ratio*n_num),replace=False)
        X_num = X_num[ind,:]
        y_num = y_num[ind]
        nums = nums[ind]
        n_word = X_word.shape[0]
        ind = np.random.choice(n_word,ceil(sample_ratio*n_word),replace=False)
        # ind = np.random.choice(n_word,n_num,replace=False)
        X_word = X_word[ind,:]
        y_word = y_word[ind]

    X = np.concatenate([X_num, X_word])
    # ind = np.random.choice(X.shape[0],2*X_num.shape[0],replace=False)
    # X_sample = X[ind,:]
    y = np.concatenate([y_num, y_word])
    # y_sample = np.concatenate([y_num,-y_num])
    print('fitting data has prepared')
    print('number embedding: ',X_num.shape)
    print('word embedding: ',X_word.shape)
    # print('number embedding: ', X_num.shape)
    # print('word embedding: ', X_sample.shape)
    # return X_sample,y_sample
    return X,X_num,y,nums