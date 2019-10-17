import torch
from skopt import gp_minimize
from skopt.space import Integer, Real, Categorical
from skopt.utils import dump

from model import OVA_Subspace_Model
from ova_experiments.local_utils import load_dataset, Minimizer

emb_fname = 'glove.6B.300d' # or 'random'

# calculate the original accuracy

# Dp = LA.norm(P_x - P_xp,axis=1)
# Dm = LA.norm(P_x[:,:,None] - P_xms,axis=1).min(axis=1)
#
# I_hat = float(torch.sum(soft_indicator(torch.tensor(Dm - Dp, dtype=torch.float), beta=beta)))/batch_size
# acc = sum(Dp <= Dm) / batch_size
# print('original acc: ',acc)
# print('original I_hat: ', -I_hat)

base_workspace = {
    'train_verbose':False,
    'n_epochs':50,
    'emb_dim':300,
    'train_data':load_dataset(emb_fname),
    'model':OVA_Subspace_Model,
    'optimizer':torch.optim.Adam
}
mini_func = gp_minimize
optimize_types = ['subspace_dim','beta','lr','mini_batch_size']
minimizer = Minimizer(base_workspace, optimize_types, mini_func)

dims = [2,8,32,64,128]
for dim in dims:

    # order should be the same as the "optimize_types"
    space = [Categorical([dim]),
             Integer(1, 30),
             Real(10 ** -5, 10 ** 0, "log-uniform"),
             Categorical([32, 64, 128, 256, 512])]

    x0 = [dim,6,0.005,512]

    res = minimizer.minimize(space,n_calls=30,verbose=True,x0=x0)
    results_fname = '_'.join(['results',emb_fname,str(dim)])
    dump(res, results_fname+'.pkl',store_objective=False)

# torch.save({
#         'beta':beta,
#         'dim':dim,
#         'n_epochs':n_epochs,
#         'mini_batch_size':mini_batch_size,
#         'W':ova_model.state_dict(),
#         'optimizer_state':optimizer.state_dict(),
#         'acc': evaluate_acc
#     }, results_fname+'.pt')
# print('best acc: ', evaluate_acc)
# # print('best loss: ',evaluate_loss)
#
# with open(join(RESULTS_DIR,results_fname+'.txt'),'w') as f:
#     f.write('best acc: %f' % (evaluate_acc))