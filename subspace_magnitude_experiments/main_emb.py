import os
import pickle
from os.path import join, isfile

import skopt
import torch
from skopt import gp_minimize
from skopt.callbacks import CheckpointSaver
from skopt.space import Integer, Real, Categorical
from skopt.utils import dump
from torch.utils.data import ConcatDataset, DataLoader
import torch.nn.functional as F

from config import DATA_DIR
from model import OVA_Subspace_Model, SC_Subspace_Model
from subspace_magnitude_experiments.local_utils import load_dataset, Minimizer, init_evaluate, train_dev_test_split

experiments = 'sc'
if experiments == 'ova':
    model = OVA_Subspace_Model
elif experiments == 'sc':
    model = SC_Subspace_Model
else:
    assert False
ftrain = experiments+'mag_str_train'
fdev = experiments+'mag_str_dev'
ftest = experiments+'mag_str_test'
if not isfile(join(DATA_DIR,ftrain+'.pkl')):
    print('prepare train dev and test')
    with open(join(DATA_DIR,experiments+'mag_str.pkl'),'rb') as f:
        X = pickle.load(f)
    train_dev_test_split(X,[0.7,0.1,0.2],experiments+'mag_str')


base_workspace = {
    'train_verbose':False,
    'n_epochs':50,
    'mini_batch_size':256,
    'emb_dim':300,
    'model':model,
    'save_model': True,
    'select_inter_model':True,
    'eval_data': ['train'],
    'working_status': 'eval', # 'optimize'/ 'infer' / 'eval'
    'optimizer':torch.optim.Adam,
    'distance_metric':'cosine'
}
mini_func = gp_minimize
optimize_types = ['subspace_dim','beta','lr']
minimizer = Minimizer(base_workspace, optimize_types, mini_func)

# embs = ['skipgram-2_num','skipgram-5_num','wiki-news-300d-1M-subword_num','crawl-300d-2M-subword_num', 'glove.840B.300d','glove.6B.300d']
# embs = ['random-1','random-2','random-3','random-4','random-5']
embs = ['skipgram-5_num']
# embs = ['random']

# dataset = load_dataset(subspace_magnitude_experiments,{'emb_fname':'wiki-news-300d-1M-subword_num'})
# cosine_distance = lambda x,y: 1 - F.cosine_similarity(x,y)
# print(init_evaluate(dataset,cosine_distance))

# for fname in embs:
#     dataset = load_dataset('scmag_str',{'emb_fname':fname})
#
#     # evaluate the embedding at the original space
#     print(init_evaluate(dataset,cosine_distance))

params = torch.load('128_300_1_cosine_0.0255_50.pt')
model = model(128,300,1,base_workspace['distance_metric'])
model.W = torch.nn.Parameter(params['W'])
train_data = load_dataset('scmag_str', {'emb_fname':'random-1'},pre_load=False)

num_emb = torch.stack([v for _,v in train_data.number_emb_dict.items()])
s_num_emb = num_emb @ model.W.cpu()

mini_batchs = DataLoader(train_data, batch_size=base_workspace['mini_batch_size'], shuffle=True, num_workers=0, pin_memory=True)
acc,_ = model.evaluate(mini_batchs)
print(acc)
exit()

for fname in embs:
    emb_conf = {}
    if 'random' in fname:
        emb_conf['dim'] = 300
    emb_conf['emb_fname'] = fname
    train_data = load_dataset('scmag_str', emb_conf,pre_load=False)
    print(len(train_data))
    # val_data = load_dataset(fdev, emb_conf)
    minimizer.base_workspace['train_data'] = train_data
    # base_workspace['val_data'] = val_data
    results_fname = '_'.join(['results', fname])
    res = skopt.load(results_fname+'.pkl')
    print(res.x)
    eval_accs = minimizer.objective(res.x)
    print('train acc: %f'%(eval_accs['train']))
    # print('val acc: %f'%(eval_accs['val']))

# for fname in embs:
#     emb_conf = {}
#     if 'random' in fname:
#         emb_conf['dim'] = 300
#     emb_conf['emb_fname'] = fname
#     train_data = load_dataset(ftrain, emb_conf)
#     dev_data = load_dataset(fdev, emb_conf)
#     test_data = load_dataset(ftest, emb_conf)
#
#     minimizer.base_workspace['train_data'] = train_data
#     minimizer.base_workspace['val_data'] = dev_data
#
#     # order should be the same as the "optimize_types"
#     space = [Integer(2,128),
#              Integer(1, 30),
#              Real(10 ** -5, 10 ** 0, "log-uniform"),
#              ]
#
#     checkpoint_fname = fname + '_checkpoint.pkl'
#     checkpoint_callback = CheckpointSaver(checkpoint_fname, store_objective=False)
#
#     # load checkpoint if there exists checkpoint, otherwise from scratch
#     if os.path.isfile(checkpoint_fname):
#         print('load checkpoint and continue optimization')
#         int_res = skopt.load(checkpoint_fname)
#         x0,y0 = int_res.x_iters,int_res.func_vals
#         res = minimizer.minimize(space, x0=x0, y0=y0, n_calls=50-len(x0), callback=[checkpoint_callback], verbose=True)
#     else:
#         x0 = [64,6,0.005]
#         res = minimizer.minimize(space, x0=x0, n_calls=50, callback=[checkpoint_callback], verbose=True)
#
#     results_fname = '_'.join(['results',fname])
#     dump(res, results_fname+'.pkl',store_objective=False)
#
#     # train on the train and dev sets and then test on test sets
#     minimizer.base_workspace['train_data'] = ConcatDataset([train_data,dev_data])
#     del minimizer.base_workspace['val_data']
#     minimizer.base_workspace['test_data'] = test_data
#
#     minimizer.base_workspace['working_status'] = 'infer'
#     minimizer.base_workspace['eval_data'] = ['test']
#
#     test_acc = minimizer.objective(res.x)
#     print('test acc: %f'% (test_acc))