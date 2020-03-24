import os
from os.path import join

import skopt

results = ['ova_acc_dim','sc_acc_dim','sc_acc_emb','ova_acc_emb']
for e in results:
    files = os.listdir(e)
    for f in files:
        res = skopt.load(join(e,f))
        res.specs['args']['callback'][0] = None
        skopt.dump(res,join(e,f))