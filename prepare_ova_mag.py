from os.path import join

import numpy as np
import pickle

from config import DATA_DIR
from utils import is_valid_relax_triple

with open(join(DATA_DIR,'scmag.pkl'),'rb') as f:
    Xs = pickle.load(f)

number_set = set(np.array(Xs).flat)
number_array = sorted(number_set)
l_number_array = len(number_array)
ova_tests = []
for i,n in enumerate(number_array):

    if i == 0 or i == l_number_array-1:
        continue
    x = n
    n_l1 = number_array[i - 1]
    n_r1 = number_array[i + 1]
    ld1 = abs(x-n_l1)
    rd1 = abs(x-n_r1)
    if ld1 <= rd1:
        xp = n_l1
    else:
        xp = n_r1
    remain_numbers = number_set - set([x,xp])

    one_test = [[x,xp,m] for m in remain_numbers]

    ova_tests.append(one_test)

# boundary cases

x = number_array[0]
xp = number_array[1]
remain_numbers = number_set - set([x, xp])

one_test = [[x, xp, m] for m in remain_numbers]
ova_tests.append(one_test)

x = number_array[-1]
xp = number_array[-2]
remain_numbers = number_set - set([x, xp])

one_test = [[x, xp, m] for m in remain_numbers]
ova_tests.append(one_test)

# check whether the test is valid
for test in ova_tests:
    for triple in test:
        assert is_valid_relax_triple(triple)

with open(join(DATA_DIR,'ovamag.pkl'),'wb') as f:
    pickle.dump(ova_tests,f,pickle.HIGHEST_PROTOCOL)