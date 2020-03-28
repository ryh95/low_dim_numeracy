import pickle
from os.path import join

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from skopt import dump
from skopt.space import Integer, Real
from torch.utils.data import ConcatDataset

from config import DATA_DIR, EMB_DIR, ORD_EXP_DIR
from ordering_exp.helper.utils import load_batched_samples, init_evaluate


class OrderdingExp(object):

    def __init__(self, exp_name, exp_data):
        self.name = exp_name
        self.exp_data = exp_data

    def run(self):
        num_src = self.exp_data['num_src'] # nums1-3
        emb_type = self.exp_data['emb_type'] # word2vec-wiki
        exp_type = self.exp_data['exp_type'] # ord-k
        minimizer = self.exp_data['minimizer']

        fX = num_src + '_' + exp_type # nums1-3_ord-k
        fX_splits = [fX + '_' + type for type in ['train', 'dev', 'test']] # nums1-3_ord-k_train

        # prepare train dev test split
        print('prepare train dev and test')
        X = np.load(join(ORD_EXP_DIR, 'data', fX + '.npy'), allow_pickle=True)
        X_train, X_test = train_test_split(X, test_size=0.2)
        X_train, X_val = train_test_split(X_train, test_size=0.1875)
        X_splits = [X_train, X_val, X_test]
        for name, data in zip(fX_splits, X_splits):
            np.save(name, data)

        # standardize number embeddings and prepare train/dev/test embedding
        embs = []
        nums = []
        with open(join(EMB_DIR,num_src+'_'+emb_type+'.pkl'), 'rb') as f: # nums1-3_word2vec-wiki.pkl
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
        # standardize the embeddings
        scaler = StandardScaler()
        emb_train = scaler.fit_transform(emb_train)
        emb_dev = scaler.transform(emb_dev)
        emb_test = scaler.transform(emb_test)
        embs = [emb_train, emb_dev, emb_test]
        num_emb_st = {n:e for num,emb in zip(nums,embs) for n,e in zip(num,emb)}
        with open(num_src+'_'+emb_type+'_st.pkl', 'wb') as handle: # nums1-3_word2vec-wiki_st.pkl
            pickle.dump(num_emb_st, handle,protocol=pickle.HIGHEST_PROTOCOL)

        # todo: show accuracy of test set in original space
        # testset = load_batched_samples(X_test, num_emb_st, pre_emb=False)
        # orig_test_acc = init_evaluate(testset, cosine_distance)
        # print('test acc in original space of %s: %.4f' % (emb_type, orig_test_acc))

        # show accuracy of test set in subspace
        datasets = []
        for X_split in X_splits:
            # load train val test dataset
            datasets.append(load_batched_samples(X_split, num_emb_st, pre_emb=False))

        minimizer.base_workspace['train_data'] = datasets[0]
        minimizer.base_workspace['val_data'] = datasets[1]
        if 'test_data' in minimizer.base_workspace:
            del minimizer.base_workspace['test_data']

        minimizer.base_workspace['working_status'] = 'optimize'
        minimizer.base_workspace['eval_data'] = ['val']

        # optimize space
        # learning rate
        space = [Real(10 ** -5, 10 ** 0, "log-uniform")]

        # x0 = [128,64,6,0.001]
        x0 = [0.0025]
        res = minimizer.minimize(space, x0=x0, n_calls=50, verbose=True)


        # train on the train and dev sets and then test on test sets
        minimizer.base_workspace['train_data'] = ConcatDataset([datasets[0], datasets[1]])
        del minimizer.base_workspace['val_data']
        minimizer.base_workspace['test_data'] = datasets[2]

        minimizer.base_workspace['working_status'] = 'infer'
        minimizer.base_workspace['eval_data'] = ['test']

        test_acc = minimizer.objective(res.x)
        print('test acc: %f' % (test_acc))

        # save results
        results_fname = '_'.join(['res-hyp', self.name])
        dump(res, results_fname + '.pkl', store_objective=False)
        np.save('_'.join(['res-acc',self.name]),np.array([test_acc]))

        return test_acc