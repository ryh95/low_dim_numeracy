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

    def __init__(self, fdata, emb_config):
        """

        :param fdata: file name of the OVA test, e.g. 'data/ovamag.pkl'
        :param emb_config: {"emb_fname": 'random'/'glove.6B.300d',"dim":300(random)}
        """
        X = np.load(join(DATA_DIR, fdata + '.npy'), allow_pickle=True)
        # with open(join(DATA_DIR, fname + '.pkl'), 'rb') as f:
        #     X = pickle.load(f)

        femb = emb_config['emb_fname'] # embedding file name used to create number embedding
        if femb == 'random':
            # fdata_emb = join(EMB_DIR, fdata + '_' + femb + '_emb')
            fdata_emb = fdata + '_' + femb + '_emb'
        else:
            # fdata_emb = join(EMB_DIR, fdata + '_emb')
            fdata_emb = fdata + '_emb'

        if isfile(join(EMB_DIR,fdata_emb + '.pickle')):
            with open(join(EMB_DIR,fdata_emb + '.pickle'), 'rb') as f:
                data_emb = pickle.load(f)
        else:
            number_array = list(set(np.array(X).flat))
            if 'train' in fdata_emb or 'dev' in fdata_emb or 'test' in fdata_emb:
                frootdata_emb = fdata_emb.replace('_train','').replace('_dev','').replace('_test','')
                print('prepare %s from %s' % (fdata_emb,frootdata_emb))
                if not os.path.isfile(join(EMB_DIR,frootdata_emb+'.pickle')):
                    print('%s not found, prepare...'%(frootdata_emb))
                    frootdata = fdata.replace('_train', '').replace('_dev', '').replace('_test', '')
                    X_root = np.load(join(DATA_DIR, frootdata + '.npy'), allow_pickle=True)
                    rootnumber_array = list(set(np.array(X_root).flat))
                    if 'random' in frootdata_emb:
                        d = emb_config['dim']
                        rootdata_emb = {n: np.random.randn(d) for n in rootnumber_array}
                        with open(join(EMB_DIR, frootdata_emb + '.pickle'), 'wb') as handle:
                            pickle.dump(rootdata_emb, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    else:
                        rootdata_emb, _ = vocab2vec(rootnumber_array, EMB_DIR, frootdata_emb,
                                                    join(EMB_DIR, femb + '.txt'), ['pickle'])
                    print('%s has been prepared'%(frootdata_emb))
                else:
                    with open(join(EMB_DIR,frootdata_emb + '.pickle'), 'rb') as f:
                        rootdata_emb = pickle.load(f)
                data_emb = {n:rootdata_emb[n] for n in number_array}
                print('%s has been prepared'%(fdata_emb))
            else:
                # todo: fix following
                if 'random' in femb:
                    print('generate random embedding...')
                    d = emb_config['dim']
                    data_emb = {n: np.random.randn(d) for n in number_array}
                    with open(fdata_emb+'.pickle', 'wb') as handle:
                        pickle.dump(data_emb, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    print('embedding saved')
                else:
                    data_emb, _ = vocab2vec(number_array, EMB_DIR, fdata_emb, base_emb, ['pickle'])

        data_emb = {k: torch.from_numpy(v).float() for k,v in data_emb.items()}

        self.number_emb_dict = data_emb
        self.number_emb_source = femb
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