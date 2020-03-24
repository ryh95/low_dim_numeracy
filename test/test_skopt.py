
import numpy as np

from sklearn.datasets import load_boston
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from skopt.callbacks import CheckpointSaver

boston = load_boston()
X, y = boston.data, boston.target
n_features = X.shape[1]

# gradient boosted trees tend to do well on problems like this
reg = GradientBoostingRegressor(n_estimators=50, random_state=0)
from skopt.space import Real, Integer
from skopt.utils import use_named_args, dump, load

# The list of hyper-parameters we want to optimize. For each one we define the bounds,
# the corresponding scikit-learn parameter name, as well as how to sample values
# from that dimension (`'log-uniform'` for the learning rate)
space  = [Integer(1, 5, name='max_depth'),
          Real(10**-5, 10**0, "log-uniform", name='learning_rate'),
          Integer(1, n_features, name='max_features'),
          Integer(2, 100, name='min_samples_split'),
          Integer(1, 100, name='min_samples_leaf')]

# this decorator allows your objective function to receive a the parameters as
# keyword arguments. This is particularly convenient when you want to set scikit-learn
# estimator parameters
@use_named_args(space)
def objective(**params):
    reg.set_params(**params)

    return -np.mean(cross_val_score(reg, X, y, cv=5, n_jobs=-1,
                                    scoring="neg_mean_absolute_error"))

from skopt import gp_minimize

# checkpoint_saver = CheckpointSaver("test_checkpoint.pkl", store_objective=False)

# res_gp = gp_minimize(objective, space, n_calls=50, callback=[checkpoint_saver], random_state=0)
# res_gp.specs['args']['func'] = None
# dump(res_gp,'test_skopt.pkl')

res = load('test_checkpoint.pkl')
res_gp = gp_minimize(objective, space, x0=res.x_iters, y0=res.func_vals, n_calls=10, random_state=0, verbose=True)
res_gp.specs['args']['func'] = None
dump(res_gp,'test_skopt2.pkl')