import os
from os.path import join

from skopt import load
import numpy as np
import torch.nn.functional as F

# results_type = 'cosine_results'
# dir_name = join(results_type,'sc_acc_emb')
# files = os.listdir(dir_name)
# aver_res = []
# for f in files:
#     res = load(join(dir_name,f))
#     print(f,res.x,res.fun)
#     if 'random' in f:
#         aver_res.append(res.fun)
# print(np.mean(aver_res))

from sub_mag_exp.local_utils import load_dataset, init_evaluate

# embs = ['glove.6B.300d','glove.840B.300d','wiki-news-300d-1M-subword_num','crawl-300d-2M-subword_num','skipgram-2_num','skipgram-5_num']
embs = ['random-1','random-2','random-3','random-4','random-5']
test_type = 'sc'
distance_metric = lambda x,y: 1 - F.cosine_similarity(x,y) # cosine distance
accs = []
for fname in embs:
    emb_conf = {}
    if fname == 'random':
        emb_conf['dim'] = 300
    emb_conf['emb_fname'] = fname
    dataset = load_dataset(test_type,emb_conf)

    # evaluate the embedding at the original space
    accs.append(init_evaluate(dataset,distance_metric))

# results
print(accs)
print(np.mean(accs))