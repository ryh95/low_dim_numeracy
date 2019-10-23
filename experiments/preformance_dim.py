import torch
from skopt import gp_minimize
from skopt.space import Integer, Real, Categorical
from skopt.utils import dump

from model import OVA_Subspace_Model, SC_Subspace_Model
from experiments.local_utils import load_dataset, Minimizer, MyCheckpointSaver

emb_fname = 'glove.6B.300d' # or 'random'
experiments = 'sc'
if experiments == 'ova':
    model = OVA_Subspace_Model
elif experiments == 'sc':
    model = SC_Subspace_Model
else:
    assert False
base_workspace = {
    'train_verbose':False,
    'n_epochs':50,
    'emb_dim':300,
    'train_data':load_dataset(experiments,emb_fname),
    'model':model,
    'optimizer':torch.optim.Adam
}
mini_func = gp_minimize
optimize_types = ['subspace_dim','beta','lr','mini_batch_size']
minimizer = Minimizer(base_workspace, optimize_types, mini_func)

dims = [4,8,16,32,64,128]
for dim in dims:

    # order should be the same as the "optimize_types"
    space = [Categorical([dim]),
             Integer(1, 30),
             Real(10 ** -5, 10 ** 0, "log-uniform"),
             Categorical([32, 64, 128, 256, 512])]

    x0 = [dim,6,0.005,512]
    checkpoint_fname = '_'.join([emb_fname,str(dim)]) + '_checkpoint.pkl'
    checkpoint_callback = MyCheckpointSaver(checkpoint_fname, remove_func=True)

    res = minimizer.minimize(space,n_calls=30,verbose=True,x0=x0,callback=[checkpoint_callback])
    results_fname = '_'.join(['results',emb_fname,str(dim)])
    dump(res, results_fname+'.pkl',store_objective=False)
