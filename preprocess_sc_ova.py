import math
from os.path import join

import numpy as np
import pickle

from config import DATA_DIR
from utils import is_valid_triple

with open(join(DATA_DIR,'scmag.pkl'),'rb') as f:
    Xs = pickle.load(f)

str_Xs = []
for triple in Xs:
    # skip the [Inf Inf Inf] sample
    if np.array([math.isinf(e) for e in triple]).all(): continue
    assert is_valid_triple(triple)
    str_triple = []
    for e in triple:
        if e.is_integer():
            str_e = str(int(e))
        else:
            str_e = str(e)
        str_triple.append(str_e)
    str_Xs.append(str_triple)

with open(join(DATA_DIR,'scmag_str.pkl'),'wb') as f:
    pickle.dump(str_Xs,f,pickle.HIGHEST_PROTOCOL)

number_set = set(np.array(Xs).flat)
number_array = sorted(number_set)
max_num = max(number_array)
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

    if ld1 < rd1:
        xp = n_l1
        xm = n_r1
    elif rd1 < ld1:
        xp = n_r1
        xm = n_l1
    else:
        # remove the boundary case
        xp = n_l1
        xm = max_num

    remain_numbers = number_set - set([x,n_l1,n_r1])

    if x.is_integer():
        str_x = str(int(x))
    else:
        str_x = str(x)
    if xp.is_integer():
        str_xp = str(int(xp))
    else:
        str_xp = str(xp)

    one_test = []
    for m in list(remain_numbers)+[xm]:
        if m.is_integer():
            str_m = str(int(m))
        else:
            str_m = str(m)
        one_test.append([str_x,str_xp,str_m])
    # one_test = [[x,xp,m] for m in remain_numbers]

    ova_tests.append(one_test)

# boundary cases

x = number_array[0]
xp = number_array[1]
remain_numbers = number_set - set([x, xp])

one_test = []
for m in remain_numbers:
    if m.is_integer():
        str_m = str(int(m))
    else:
        str_m = str(m)
    one_test.append([str(int(x)),str(int(xp)),str_m])

# one_test = [[x, xp, m] for m in remain_numbers]
ova_tests.append(one_test)

x = number_array[-1]
xp = number_array[-2]
remain_numbers = number_set - set([x, xp])

one_test = []
for m in remain_numbers:
    if m.is_integer():
        str_m = str(int(m))
    else:
        str_m = str(m)
    one_test.append([str(x),str(int(xp)),str_m])

# one_test = [[x, xp, m] for m in remain_numbers]
ova_tests.append(one_test)

# check whether the test is valid
# we believe that the last sample [Inf 100000 x](x<100000) is a valid test, since the
# magnitude of 100000 is closer to Inf's compared with x
# As the distances are both Inf, so just let this sample pass the test
for test in ova_tests[:-1]:
    for triple in test:
        assert is_valid_triple([float(n) for n in triple])

with open(join(DATA_DIR,'ovamag_str.pkl'),'wb') as f:
    pickle.dump(ova_tests,f,pickle.HIGHEST_PROTOCOL)