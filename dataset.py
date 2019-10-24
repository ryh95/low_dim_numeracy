import math
import os
import pickle
from os.path import join, isfile

import torch
from torch.utils.data import Dataset
import numpy as np

from config import EMB_DIR, DATA_DIR
from utils import vocab2vec

"""
custom dataset
see ref:
https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
"""

class BaseDataset(Dataset):

    def __init__(self, fname, emb_config):
        """

        :param fname: file name of the OVA test, e.g. 'data/ovamag.pkl'
        :param emb_config: {"emb_fname": 'random'/'glove.6B.300d',"dim":300(random)}
        """
        with open(join(DATA_DIR, fname + '.pkl'), 'rb') as f:
            X = pickle.load(f)

        emb_fname = emb_config['emb_fname'] # embedding file name used to create number embedding
        num_emb_fname = join(EMB_DIR, emb_fname + '_' + fname + '_num_emb') # embedding file name for numbers in OVA
        base_emb = join(EMB_DIR, emb_fname + '.txt') # add suffix

        if isfile(num_emb_fname + '.pickle'):
            with open(num_emb_fname + '.pickle', 'rb') as f:
                number_emb_dict = pickle.load(f)
        else:

            number_array = list(set(np.array(X).flat))

            if emb_fname == 'random':
                print('generate random embedding...')
                d = emb_config['dim']
                number_emb_dict = {n: np.random.randn(d) for n in number_array}
                with open(num_emb_fname+'.pickle', 'wb') as handle:
                    pickle.dump(number_emb_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
                print('embedding saved')
            else:
                number_emb_dict, _ = vocab2vec(number_array, EMB_DIR, num_emb_fname, base_emb, ['pickle'])

        number_emb_dict = {k: torch.from_numpy(v).float() for k,v in number_emb_dict.items()}

        self.number_emb_dict = number_emb_dict
        self.number_emb_source = emb_fname
        self.data = X

    def __len__(self):
        return len(self.data)

class OVADataset(BaseDataset):

    def __getitem__(self, idx):
        """

        :param idx: id of the sample
        :return: the sample corresponding with the idx
        """
        # generally not recommended to return CUDA tensors in multi - process loading
        # see ref
        # https://pytorch.org/docs/stable/data.html
        test_sample = self.data[idx]
        n_ova = len(test_sample)
        d = next(iter(self.number_emb_dict.values())).numel()
        P_xms = torch.zeros(d, n_ova, dtype=torch.float32)
        for i,(x,xp,xm) in enumerate(test_sample):
            P_xms[:,i] = self.number_emb_dict[xm]

        P_x = self.number_emb_dict[test_sample[0][0]]
        P_xp = self.number_emb_dict[test_sample[0][1]]


        return P_x,P_xp,P_xms

class SCDataset(BaseDataset):

    def __getitem__(self, idx):
        test_sample = self.data[idx]
        P_x = self.number_emb_dict[test_sample[0]]
        P_xp = self.number_emb_dict[test_sample[1]]
        P_xm = self.number_emb_dict[test_sample[2]]
        return P_x,P_xp,P_xm