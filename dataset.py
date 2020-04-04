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

    def __init__(self, X, num_emb):
        """

        :param X: sc/sc-k/ova datasets
        :param num_emb: emb dict that covers all numbers in X
        """

        self.number_emb = {k: torch.from_numpy(v).float() for k,v in num_emb.items()}
        self.X = X

    def __len__(self):
        return self.X.shape[0]

class OVADataset(BaseDataset):

    def __getitem__(self, idx):
        """

        :param idx: id of the sample
        :return: the sample corresponding with the idx
        """
        # generally not recommended to return CUDA tensors in multi - process loading
        # see ref
        # https://pytorch.org/docs/stable/data.html
        test_sample = self.X[idx]
        n_ova = test_sample.shape[0]
        d = next(iter(self.number_emb.values())).numel()
        P_xms = torch.zeros(n_ova, d, dtype=torch.float32)
        for i,(x,xp,xm) in enumerate(test_sample):
            P_xms[i,:] = self.number_emb[xm]

        P_x = self.number_emb[test_sample[0][0]]
        P_xp = self.number_emb[test_sample[0][1]]


        return P_x,P_xp,P_xms

class SCDataset(BaseDataset):

    def __getitem__(self, idx):
        test_sample = self.X[idx]
        P_x = self.number_emb[test_sample[0]]
        P_xp = self.number_emb[test_sample[1]]
        P_xm = self.number_emb[test_sample[2]]
        return P_x,P_xp,P_xm

class OrdDataset(BaseDataset):

    def __getitem__(self, idx):
        """

        :param idx: id of the sample
        :return: the embeded sample corresponding with the idx
        """

        sample = self.X[idx]
        k = sample.shape[0]
        d = next(iter(self.number_emb.values())).numel()
        P_xms = torch.zeros(k, d, dtype=torch.float32) # kxd
        for i, (x, xm) in enumerate(sample):
            P_xms[i, :] = self.number_emb[xm]

        P_x = self.number_emb[sample[0][0]] # d

        return P_x, P_xms