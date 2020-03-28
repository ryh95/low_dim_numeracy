from os.path import join

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from config import VOCAB_DIR, ORD_EXP_DIR
from dataset import OrdDataset


def prepare_ord_k(fnums,k=100):
    # sort numbers
    nums = np.load(join(VOCAB_DIR, fnums + '.npy'))
    s_numbers = sorted(nums, key=lambda x: float(x))

    X_ord_k = np.empty((nums.size-k, k, 2), dtype=np.object) # since the window is slicing, numeber of sample is not n
    for i,n in enumerate(s_numbers[:-k]):
        X_ord_k[i,:,0] = n
        X_ord_k[i,:,1] = s_numbers[i+1:i+k+1]

    print('number of ord_k tests: %d' % (nums.size-k))
    fX_ord_k = fnums + '_ord-k'
    np.save(join(ORD_EXP_DIR, 'data', fX_ord_k), X_ord_k)
    return X_ord_k

def load_batched_samples(X, num_emb, pre_emb=True):
    batched_samples = OrdDataset(X,num_emb)
    if pre_emb:
        # preload all train_data into memory to save time
        mini_batchs = DataLoader(batched_samples, batch_size=128, num_workers=6)
        P_x, P_xms = [], []
        for i, mini_batch in enumerate(mini_batchs):
            mini_P_x, mini_P_xp, mini_P_xms = mini_batch
            P_x.append(mini_P_x)
            P_xms.append(mini_P_xms)
        P_x = torch.cat(P_x)
        P_xms = torch.cat(P_xms)
        batched_samples = TensorDataset(P_x, P_xms)
    return batched_samples

def init_evaluate(dataset):
    pass