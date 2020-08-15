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
from experiments import BaseExperiments
from sub_mag_exp.helper.utils import load_batched_samples, init_evaluate
from utils import vocab2vec, cosine_distance


class SubspaceMagExp(BaseExperiments):

    def prepare_datasets(self):
        self.num_src = self.exp_data['num_src']  # nums1-3
        self.emb_type = self.exp_data['emb_type']  # word2vec-wiki
        self.exp_type = self.exp_data['exp_type']  # sc-k
        self.minimizer = self.exp_data['minimizer']

        self.fX = self.num_src + '_' + self.exp_type  # nums1-3_sc-k
        self.fX_splits = [self.fX + '_' + type for type in ['train', 'dev', 'test']]  # nums1-3_sc-k_train

        # prepare train dev test split
        # todo: should we standardize number embeddings? i.e. call standardize_train_dev_test_emb?
        with open(join(EMB_DIR, self.num_src + '_' + self.emb_type + '.pkl'), 'rb') as f:  # nums1-3_word2vec-wiki.pkl
            self.num_emb = pickle.load(f)
        self.X_splits = self.train_dev_test_split()
        self.datasets = {}
        for X_split, split_type in zip(self.X_splits, ['train', 'val', 'test']):
            # load train val test dataset
            self.datasets[split_type] = load_batched_samples(X_split, self.num_emb)

    def train_dev_test_split(self):

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

        return num_emb_st

    def show_benchmark_res(self):

        # emb_conf = {}self.
        # if 'random' in femb:
        #     emb_conf['dim'] = 300
        # emb_conf['emb_fname'] = femb
        self.res['orig_test_res'] = init_evaluate(self.datasets['test'], cosine_distance)
        print('test acc in original space of %s: %.4f' % (self.emb_type, self.res['orig_test_res']))

    def show_model_res(self):

        self.minimizer.base_workspace['train_data'] = self.datasets['train']
        self.minimizer.evaluator.eval_data = self.datasets['val']

        # optimize
        hpy_tune_space = self.minimizer.base_workspace['hyp_tune_space']
        x0 = self.minimizer.base_workspace['hyp_tune_x0']
        n_calls = self.minimizer.base_workspace['hyp_tune_calls']
        res = self.minimizer.minimize(hpy_tune_space, x0=x0, n_calls=n_calls, verbose=True)

        # train on the train and dev sets and then test on test sets
        self.minimizer.base_workspace['train_data'] = ConcatDataset([self.datasets['train'], self.datasets['val']])
        self.minimizer.evaluator.eval_data = self.datasets['test']

        # self.minimizer.base_workspace['save_model'] = True

        test_acc = -self.minimizer.objective(res.x)
        print('test acc: %f' % (test_acc))

        self.res['test_res'] = test_acc
        self.res['minimizer_res'] = res

    def save_res(self):
        if 'minimizer_res' in self.res:
            results_fname = '_'.join(['res-hyp', self.name])
            dump(self.res['minimizer_res'], results_fname + '.pkl', store_objective=False)
            self.res.pop('minimizer_res')
        with open('_'.join(['res-exp', self.name])+'.pkl','wb') as f:
            pickle.dump(self.res,f,pickle.HIGHEST_PROTOCOL)


    def run(self):
        self.prepare_datasets()

        # show accuracy of test set in original space
        self.show_benchmark_res()

        # show accuracy of test set in subspace
        self.show_model_res()

        # save results
        self.save_res()

        return self.res
