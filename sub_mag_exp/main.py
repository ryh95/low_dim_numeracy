from os.path import join

import torch
from skopt import gp_minimize
from skopt.space import Integer, Real

from config import SUB_MAG_EXP_DIR, EMB_DIR
from model import OVAModel, SCModel, LogisticLoss
from sub_mag_exp.helper.evaluate import SubspaceMagEvaluator
from sub_mag_exp.helper.experiments import SubspaceMagExp
from sub_mag_exp.helper.minimizer import Minimizer

exp_type = 'sc-k'
num_src = 'nums1-3'

if exp_type == 'ova' or exp_type == 'sc-k':
    model = OVAModel
elif exp_type == 'sc':
    model = SCModel
else:
    assert False

emb_types = ['word2vec-wiki', 'word2vec-giga', 'glove-wiki', 'glove-giga', 'fasttext-wiki', 'fasttext-giga','random']
# emb_types = ['word2vec-wiki']
exps = []

base_workspace = {
    'train_verbose':True,
    'n_epochs':50,
    'mini_batch_size':256,
    'emb_dim':300,
    'model':model,
    'model_type':'normal', # regularized
    'mapping_type':'subspace',
    # 'subspace_dim':160,
    'loss': LogisticLoss,
    # 'loss_params':{'beta':18},
    'save_model': False,
    'select_inter_model':False,
    # 'eval_data': ['val'],
    # 'working_status': 'optimize', # 'optimize'/ 'infer' / 'eval'
    'optimizer':torch.optim.Adam,
    'distance_metric':'cosine',
    'hyp_tune_space' : [Integer(2,20),
             Integer(2, 256),
             Real(10 ** -5, 10 ** 0, "log-uniform"),
             ],
    'hyp_tune_x0' : [18, 160, 0.0025],
    'hyp_tune_calls': 50
}
mini_func = gp_minimize
optimize_types = ['loss__beta','subspace_dim','lr']
# optimize_types = ['n_hidden1','n_out','lr']
minimizer = Minimizer(base_workspace, optimize_types, mini_func)
evaluator = SubspaceMagEvaluator()
minimizer.evaluator = evaluator

for emb_type in emb_types:
    exp_data = {
        'num_src': num_src,
        'emb_type': emb_type,
        'exp_type': exp_type,
        'minimizer': minimizer
    }
    exp_name = '_'.join([num_src, exp_type, emb_type])
    exp = SubspaceMagExp(exp_name, exp_data)
    exps.append(exp)

for exp in exps:
    exp.run()