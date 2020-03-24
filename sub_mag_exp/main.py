import torch
from skopt import gp_minimize

from ..model import OVAModel, SCModel, LogisticLoss
from .experiments import SubspaceMagExp
from .minimizer import Minimizer

exp_type = 'sc_k'

if exp_type == 'ova' or exp_type == 'sc_k':
    model = OVAModel
elif exp_type == 'sc':
    model = SCModel
else:
    assert False
num_sources = ['sel_orig_nums']
fembs = ['word2vec-giga','glove-wiki','glove-giga','fasttext-wiki','fasttext-giga']
exps = []

base_workspace = {
    'train_verbose':True,
    'n_epochs':50,
    'mini_batch_size':256,
    'emb_dim':300,
    'model':model,
    'mapping_type':'subspace',
    # 'subspace_dim':160,
    'loss': LogisticLoss,
    'loss_params':{'beta':18},
    'save_model': False,
    'select_inter_model':False,
    # 'eval_data': ['val'],
    # 'working_status': 'optimize', # 'optimize'/ 'infer' / 'eval'
    'optimizer':torch.optim.Adam,
    'distance_metric':'cosine'
}
mini_func = gp_minimize
optimize_types = ['subspace_dim','lr']
# optimize_types = ['n_hidden1','n_out','lr']
minimizer = Minimizer(base_workspace, optimize_types, mini_func)


for src in num_sources:
    for femb in fembs:
        exp_data = {
            'num_src':src,
            'exp_type':exp_type,
            'femb':femb,
            'minimizer':minimizer
        }
        exp_name = '_'.join(['test','results', src, exp_type, femb])
        exp = SubspaceMagExp(exp_name, True, exp_data)
        exps.append(exp)

for exp in exps:
    exp.run()