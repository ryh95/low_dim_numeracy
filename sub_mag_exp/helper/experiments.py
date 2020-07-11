import os
import pickle
from os.path import join, isfile

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from skopt import dump
from skopt.space import Integer, Real
from torch.utils.data import ConcatDataset

from config import DATA_DIR, EMB_DIR, SUB_MAG_EXP_DIR
from sub_mag_exp.helper.utils import load_batched_samples, init_evaluate
from utils import vocab2vec, cosine_distance


class SubspaceMagExp(object):

    def __init__(self, exp_name, exp_data):
        self.name = exp_name
        self.exp_data = exp_data

    def prepare_train_dev_test_split(self):

        if os.path.exists(self.fX_splits[0] + '.npy'):
            # if train dev test split exists
            print('load train dev and test')
            X_train = np.load(self.fX_splits[0] + '.npy', allow_pickle=True)
            X_val = np.load(self.fX_splits[1] + '.npy', allow_pickle=True)
            X_test = np.load(self.fX_splits[2] + '.npy', allow_pickle=True)
            X_splits = [X_train, X_val, X_test]
        else:
            print('prepare train dev and test')
            X = np.load(join(SUB_MAG_EXP_DIR, 'data', self.fX + '.npy'), allow_pickle=True)
            X_train, X_test = train_test_split(X, test_size=0.2)
            X_train, X_val = train_test_split(X_train, test_size=0.1875)
            X_splits = [X_train, X_val, X_test]
            for name, data in zip(self.fX_splits, X_splits):
                np.save(name, data)

        return X_splits

    def standardize_train_dev_test_emb(self,X_splits):

        embs = []
        nums = []
        with open(join(EMB_DIR, self.num_src + '_' + self.emb_type + '.pkl'), 'rb') as f:  # nums1-3_word2vec-wiki.pkl
            num_emb = pickle.load(f)
        # todo: all d should be automatically set
        d = 300
        for X_split in X_splits:
            # X_split = np.load(fX_split, allow_pickle=True)
            split_num = list(set(np.array(X_split).flat))
            nums.append(split_num)
            split_num_emb = np.zeros((len(split_num), d))
            for i, num in enumerate(split_num):
                split_num_emb[i, :] = num_emb[num]
            embs.append(split_num_emb)

        emb_train, emb_dev, emb_test = embs
        scaler = StandardScaler()
        emb_train = scaler.fit_transform(emb_train)
        emb_dev = scaler.transform(emb_dev)
        emb_test = scaler.transform(emb_test)
        embs = [emb_train, emb_dev, emb_test]
        num_emb_st = {n: e for num, emb in zip(nums, embs) for n, e in zip(num, emb)}
        with open(self.num_src+'_'+self.emb_type+'_st.pkl', 'wb') as handle: # nums1-3_word2vec-wiki_st.pkl
            pickle.dump(num_emb_st, handle,protocol=pickle.HIGHEST_PROTOCOL)

    def show_original_acc(self,num_emb,X_test):

        # emb_conf = {}self.
        # if 'random' in femb:
        #     emb_conf['dim'] = 300
        # emb_conf['emb_fname'] = femb
        testset = load_batched_samples(X_test, num_emb)
        orig_test_acc = init_evaluate(testset, cosine_distance)
        print('test acc in original space of %s: %.4f' % (self.emb_type, orig_test_acc))
        return orig_test_acc

    def show_subspace_acc(self,num_emb,X_splits,minimizer):

        datasets = []
        for X_split in X_splits:
            # load train val test dataset
            datasets.append(load_batched_samples(X_split, num_emb))

        minimizer.base_workspace['train_data'] = datasets[0]
        minimizer.base_workspace['val_data'] = datasets[1]
        if 'test_data' in minimizer.base_workspace:
            del minimizer.base_workspace['test_data']

        minimizer.base_workspace['working_status'] = 'optimize'
        minimizer.base_workspace['eval_data'] = ['val']

        # optimize
        hpy_tune_space = minimizer.base_workspace['hyp_tune_space']
        x0 = minimizer.base_workspace['hyp_tune_x0']
        n_calls = minimizer.base_workspace['hyp_tune_calls']
        res = minimizer.minimize(hpy_tune_space, x0=x0, n_calls=n_calls, verbose=True)

        # train on the train and dev sets and then test on test sets
        minimizer.base_workspace['train_data'] = ConcatDataset([datasets[0], datasets[1]])
        del minimizer.base_workspace['val_data']
        minimizer.base_workspace['test_data'] = datasets[2]

        minimizer.base_workspace['working_status'] = 'infer'
        minimizer.base_workspace['eval_data'] = ['test']
        minimizer.base_workspace['save_model'] = True

        test_acc = minimizer.objective(res.x)
        print('test acc: %f' % (test_acc))
        return test_acc,res

    def save_res(self,minimizer_res,orig_test_acc, test_acc):

        results_fname = '_'.join(['res-hyp', self.name])
        dump(minimizer_res, results_fname + '.pkl', store_objective=False)
        np.save('_'.join(['res-acc', self.name]), np.array([orig_test_acc, test_acc]))

    def run(self):
        self.num_src = self.exp_data['num_src'] # nums1-3
        self.emb_type = self.exp_data['emb_type'] # word2vec-wiki
        self.exp_type = self.exp_data['exp_type'] # sc-k
        self.minimizer = self.exp_data['minimizer']

        self.fX = self.num_src + '_' + self.exp_type # nums1-3_sc-k
        self.fX_splits = [self.fX + '_' + type for type in ['train', 'dev', 'test']] # nums1-3_sc-k_train

        # prepare train dev test split
        X_splits = self.prepare_train_dev_test_split()

        # todo: should we standardize number embeddings? i.e. call standardize_train_dev_test_emb?
        with open(join(EMB_DIR, self.num_src + '_' + self.emb_type + '.pkl'), 'rb') as f:  # nums1-3_word2vec-wiki.pkl
            num_emb = pickle.load(f)

        # show accuracy of test set in original space
        X_test = X_splits[-1]
        orig_test_acc = self.show_original_acc(num_emb,X_test)

        # show accuracy of test set in subspace
        test_acc,minimizer_res = self.show_subspace_acc(num_emb,X_splits,self.minimizer)

        # save results
        self.save_res(minimizer_res,orig_test_acc, test_acc)

        return orig_test_acc,test_acc

class RegularizedSubspaceMagExp(SubspaceMagExp):

    pass