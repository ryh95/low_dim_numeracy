import torch
from skopt import gp_minimize

from model import LogisticLoss, AxisOrdering
from ordering_exp.helper.experiments import OrderdingExp
from ordering_exp.helper.minimizer import Minimizer

exp_type = 'ord-k'
num_src = 'nums1-3'

model = AxisOrdering

emb_types = ['word2vec-wiki','word2vec-giga', 'glove-wiki', 'glove-giga', 'fasttext-wiki', 'fasttext-giga']
exps = []

base_workspace = {
    'train_verbose':True,
    'n_epochs':50,
    'mini_batch_size':256,
    'emb_dim':300,
    'model':model,
    'mapping_type':'subspace',
    'loss': LogisticLoss,
    'loss_params':{'beta':16},
    'save_model': False,
    'select_inter_model':False,
    # 'train_data':
    # 'val_data':
    # 'test_data':
    # 'eval_data': ['val'],
    # 'working_status': 'optimize', # 'optimize'/ 'infer' / 'eval'
    'optimizer':torch.optim.Adam,
}
mini_func = gp_minimize
optimize_types = ['lr']
minimizer = Minimizer(base_workspace, optimize_types, mini_func)

for emb_type in emb_types:
    exp_data = {
        'num_src': num_src,
        'emb_type': emb_type,
        'exp_type': exp_type,
        'minimizer': minimizer
    }
    exp_name = '_'.join([num_src, exp_type, emb_type])
    exp = OrderdingExp(exp_name, exp_data)
    exps.append(exp)

for exp in exps:
    exp.run()