import itertools
import pickle
from os.path import join
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from config import EMB_DIR, SUB_MAG_EXP_DIR, EMB
from numeral_prediction_exp.helper.evaluate import SubspacePredictionEvaluator, BasePredictionEvaluator
from sub_mag_exp.helper.experiments import SubspaceMagExp
from sub_mag_exp.helper.utils import load_batched_samples
from utils import vocab2vec


class RegualarizedSubspacePrediction(SubspaceMagExp):

    def prepare_datasets(self):

        self.num_src = self.exp_data['num_src']  # nums1-3
        self.emb_type = self.exp_data['emb_type']  # word2vec-wiki
        self.exp_type = self.exp_data['exp_type']  # sc-k
        self.fX = self.num_src + '_' + self.exp_type  # nums1-3_sc-k
        self.minimizer = self.exp_data['minimizer']
        self.window_size = self.exp_data['window_size']

        # load number embedding
        with open(join(EMB_DIR, self.num_src + '_' + self.emb_type + '.pkl'), 'rb') as f:  # nums1-3_word2vec-wiki.pkl
            self.num_emb_dict = pickle.load(f)
        # load numbers
        self.nums = np.load('../data/vocab/nums1-3.npy')
        # load nums_sc-k
        X = np.load(join(SUB_MAG_EXP_DIR, 'data', self.fX + '.npy'), allow_pickle=True)
        # load embedded nums_sc-k
        self.train_data = load_batched_samples(X, self.num_emb_dict, False)

        # load numeral prediction data (sentences)
        with open('data/processed_sentences.pickle', 'rb') as f:
            sens = pickle.load(f)
        # split numeral prediction sentences
        self.val_data,self.test_data = train_test_split(sens, test_size=0.5)

        # load vocab embedding (vocab of numeral prediction)
        if not Path('data/vocab_embedding.pkl').exists():
            vocab = list(set(itertools.chain.from_iterable((sens))))
            self.emb_dict, emb = vocab2vec(vocab, 'data', 'vocab_embedding', EMB, ['pickle', 'npy'], oov_handle='none')
        else:
            with open('data/vocab_embedding.pkl', 'rb') as f:
                self.emb_dict = pickle.load(f)
            emb = np.load('data/vocab_embedding.npy')


    def show_benchmark_res(self):
        evaluator = BasePredictionEvaluator(self.emb_dict, self.num_emb_dict, self.nums, self.test_data, self.window_size)
        self.res['orig_test_res'] = evaluator.evaluate()
        print(self.res['orig_test_res'])

    def show_model_res(self):

        self.minimizer.base_workspace['train_data'] = self.train_data

        evaluator = SubspacePredictionEvaluator(self.emb_dict, self.num_emb_dict, self.nums, self.val_data,
                                                self.window_size)
        self.minimizer.evaluator = evaluator

        # optimize
        hpy_tune_space = self.minimizer.base_workspace['hyp_tune_space']
        x0 = self.minimizer.base_workspace['hyp_tune_x0']
        n_calls = self.minimizer.base_workspace['hyp_tune_calls']
        res = self.minimizer.minimize(hpy_tune_space, x0=x0, n_calls=n_calls, verbose=True)

        # test on test sets
        self.minimizer.evaluator.sens = self.test_data
        best_model = self.minimizer.evaluator.best_model
        self.res['test_res'] = self.minimizer.evaluator.evaluate(best_model)
        # self.minimizer.base_workspace['save_model'] = True

        self.res['minimizer_res'] = res
        print(self.res['test_res'])

