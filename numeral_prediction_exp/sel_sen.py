import pickle
from collections import deque

import numpy as np

# load nums and sentences
from tqdm import tqdm

nums = np.load('../data/vocab/nums1-3.npy')
set_nums = set(nums)
with open('number_sens.pickle','rb') as f:
    sens = pickle.load(f)

# select a subset of sentences
covered_num,sel_sens = set(),deque()
for l in tqdm(sens):
    inter_num = set(l) & set_nums
    if not inter_num.issubset(covered_num):
        sel_sens.append(l)
        covered_num |= inter_num

print(len(covered_num),len(set_nums))

with open('sel_number_sens.pickle','wb') as f:
    pickle.dump(sel_sens,f,pickle.HIGHEST_PROTOCOL)