import math
from os.path import join

import numpy as np
import pickle

from config import DATA_DIR
from utils import is_valid_triple, obtain_OVA_from_SC

with open(join(DATA_DIR,'scmag.pkl'),'rb') as f:
    Xs = pickle.load(f)

# transform all numbers in sc into str

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

# prepare ova according to sc
obtain_OVA_from_SC(str_Xs)