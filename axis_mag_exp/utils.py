from os.path import join
import numpy as np
from tqdm import tqdm

from ..config import EMB_DIR


def load_num_emb(femb, sel_nums):
    '''
    load number embedding from femb
    embedding in femb should be all number embedding
    :param femb:
    :return:
    '''
    # todo: possible duplicate with utils/preprocess_google_news_skip
    # todo: might duplicate with vocab2vec
    number_emb,number_target = [],[]
    number_emb_val, number_target_val = [], []
    number_emb_test, number_target_test = [], []
    print('prepare data...')
    sel_nums_set = set(sel_nums)
    with open(join(EMB_DIR, femb+'.txt'), 'r') as f:
        f.readline()
        for line in tqdm(f):
            word, *vec = line.rstrip().split(' ')
            vec = np.array(vec, dtype=float)
            word = word.split('_')[0]
            if word in sel_nums_set:
                number_emb.append(vec)
                number_target.append(float(word))
                sel_nums_set.remove(word)

    X = np.stack(number_emb)
    y = np.array(number_target)
    return X,y