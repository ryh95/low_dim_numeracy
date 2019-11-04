import os
import pickle
from os.path import join, isfile

import skopt
import torch
from skopt import gp_minimize
from skopt.callbacks import CheckpointSaver
from skopt.space import Integer, Real, Categorical
from skopt.utils import dump

from config import DATA_DIR
from model import OVA_Subspace_Model, SC_Subspace_Model
from experiments.local_utils import load_dataset, Minimizer, init_evaluate, train_dev_test_split

experiments = 'ova'
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
    'optimizer':torch.optim.Adam,
    'distance_metric':'cosine'
}
mini_func = gp_minimize
optimize_types = ['subspace_dim','beta','lr']
minimizer = Minimizer(base_workspace, optimize_types, mini_func)

# embs = ['skipgram-2_num','skipgram-5_num','wiki-news-300d-1M-subword_num','crawl-300d-2M-subword_num', 'glove.840B.300d','glove.6B.300d']
embs = ['random-1','random-2','random-3','random-4','random-5']
# embs = ['random']

# dataset = load_dataset(experiments,{'emb_fname':'wiki-news-300d-1M-subword_num'})
# cosine_distance = lambda x,y: 1 - F.cosine_similarity(x,y)
# print(init_evaluate(dataset,cosine_distance))

# for fname in embs:
#     dataset = load_dataset(fname)
#
#     # evaluate the embedding at the original space
#     init_evaluate(dataset)

for fname in embs:
    emb_conf = {}
    if 'random' in fname:
        emb_conf['dim'] = 300
    emb_conf['emb_fname'] = fname
    train_data = load_dataset(ftrain, emb_conf)
    dev_data = load_dataset(fdev, emb_conf)
    test_data = load_dataset(ftest, emb_conf)

    base_workspace['train_data'] = train_data
    base_workspace['val_data'] = dev_data
    base_workspace['test_data'] = test_data

    # order should be the same as the "optimize_types"
    space = [Integer(2,128),
             Integer(1, 30),
             Real(10 ** -5, 10 ** 0, "log-uniform"),
             ]

    checkpoint_fname = fname + '_checkpoint.pkl'
    checkpoint_callback = CheckpointSaver(checkpoint_fname, store_objective=False)

    # load checkpoint if there exists checkpoint, otherwise from scratch
    if os.path.isfile(checkpoint_fname):
        print('load checkpoint and continue optimization')
        int_res = skopt.load(checkpoint_fname)
        x0,y0 = int_res.x_iters,int_res.func_vals
        res = minimizer.minimize(space, x0=x0, y0=y0, n_calls=50-len(x0), callback=[checkpoint_callback], verbose=True)
    else:
        x0 = [64,6,0.005]
        res = minimizer.minimize(space, x0=x0, n_calls=50, callback=[checkpoint_callback], verbose=True)

    results_fname = '_'.join(['results',fname])
    dump(res, results_fname+'.pkl',store_objective=False)
