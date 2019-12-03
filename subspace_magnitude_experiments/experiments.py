import pickle
from os.path import join, isfile

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from skopt import dump
from skopt.space import Integer, Real
from torch.utils.data import ConcatDataset

from config import DATA_DIR, EMB_DIR
from subspace_magnitude_experiments.local_utils import load_dataset, init_evaluate
from utils import vocab2vec, cosine_distance


class SubspaceMagExp(object):

    def __init__(self, exp_name, save_results, exp_data):
        self.name = exp_name
        self.save_results = save_results
        self.exp_data = exp_data

    def run(self):
        src = self.exp_data['num_src']
        exp_type = self.exp_data['exp_type']
        femb = self.exp_data['femb']
        minimizer = self.exp_data['minimizer']

        fdatas = [src + '_' + exp_type + '_' + type for type in ['train', 'dev', 'test']]
        fdata_embs = [fdata + '_' + femb for fdata in fdatas]

        # if no train dev test split
        if not isfile(join(DATA_DIR, fdatas[0] + '.npy')):
            print('prepare train dev and test')
            X_root = np.load(join(DATA_DIR, src + '_' + exp_type + '.npy'), allow_pickle=True)
            X_train, X_test = train_test_split(X_root, test_size=0.2)
            X_train, X_val = train_test_split(X_train, test_size=0.1875)
            for name, data in zip(['train', 'dev', 'test'], [X_train, X_val, X_test]):
                np.save(join(DATA_DIR, src + '_' + exp_type + '_' + name), data)

        # if no embedding of training data, prepare and standardize
        # random embedding will be handled in dataset module
        if femb != 'random' and not isfile(join(EMB_DIR, fdata_embs[0] + '.pickle')):
            # prepare train/dev/test embedding
            embs = []
            nums = []
            X_root = np.load(join(DATA_DIR, src + '_' + exp_type + '.npy'), allow_pickle=True)
            X_root_num = list(set(np.array(X_root).flat))
            X_root_num_demb, _ = vocab2vec(X_root_num, output_dir=EMB_DIR, output_name=fdata_embs[0],
                                           word_emb=join(EMB_DIR, femb + '.txt'), savefmt=['None'])
            for fdata in fdatas:
                X = np.load(join(DATA_DIR, fdata + '.npy'), allow_pickle=True)
                X_num = list(set(np.array(X).flat))
                nums.append(X_num)
                X_num_emb = np.zeros((len(X_num), 300))
                for i, nu in enumerate(X_num):
                    X_num_emb[i, :] = X_root_num_demb[nu]
                embs.append(X_num_emb)

            emb_train, emb_dev, emb_test = embs
            # standardize the embeddings
            scaler = StandardScaler()
            emb_train = scaler.fit_transform(emb_train)
            emb_dev = scaler.transform(emb_dev)
            emb_test = scaler.transform(emb_test)
            embs = [emb_train, emb_dev, emb_test]
            for i in range(3):
                with open(join(EMB_DIR, fdata_embs[i] + '.pickle'), 'wb') as handle:
                    pickle.dump({num: embs[i][j, :] for j, num in enumerate(nums[i])}, handle,
                                protocol=pickle.HIGHEST_PROTOCOL)

        # show accuracy of test set in original space
        emb_conf = {}
        if 'random' in femb:
            emb_conf['dim'] = 300
        emb_conf['emb_fname'] = femb
        test_dataset = load_dataset(fdatas[-1], emb_conf)
        orig_test_acc = init_evaluate(test_dataset, cosine_distance)
        print('test acc in original space of %s: %.4f' % (femb, orig_test_acc))

        # show accuracy of test set in subspace
        datas = []
        for fdata in fdatas:
            # load train val test dataset
            datas.append(load_dataset(fdata, emb_conf))

        minimizer.base_workspace['train_data'] = datas[0]
        minimizer.base_workspace['val_data'] = datas[1]
        if 'test_data' in minimizer.base_workspace:
            del minimizer.base_workspace['test_data']

        minimizer.base_workspace['working_status'] = 'optimize'
        minimizer.base_workspace['eval_data'] = ['val']

        # optimize space
        space = [Integer(2, 256),
                 Real(10 ** -5, 10 ** 0, "log-uniform"),
                 ]

        # x0 = [128,64,6,0.001]
        x0 = [160, 0.0025]
        res = minimizer.minimize(space, x0=x0, n_calls=50, verbose=True)


        # train on the train and dev sets and then test on test sets
        minimizer.base_workspace['train_data'] = ConcatDataset([datas[0], datas[1]])
        del minimizer.base_workspace['val_data']
        minimizer.base_workspace['test_data'] = datas[2]

        minimizer.base_workspace['working_status'] = 'infer'
        minimizer.base_workspace['eval_data'] = ['test']

        test_acc = minimizer.objective(res.x)
        print('test acc: %f' % (test_acc))

        if self.save_results:
            results_fname = '_'.join(['results', src, exp_type, femb])
            dump(res, results_fname + '.pkl', store_objective=False)
            np.save(self.name,np.array([orig_test_acc,test_acc]))

        return orig_test_acc,test_acc