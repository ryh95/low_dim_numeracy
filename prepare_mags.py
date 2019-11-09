import math
import pickle
import random
from collections import defaultdict

import numpy as np

from tqdm import tqdm

from config import EMB
from utils import is_number, is_valid_triple, obtain_OVA_from_SC

# collect numbers from word embedding
numbers = defaultdict(list)
with open(EMB,'r') as f:
    for line in tqdm(f):
        word, *vec = line.rstrip().split(' ')
        if is_number(word):
           numbers[float(word)].append(word)

# remove redudent numbers
# e.g. ['-3', '-3.0', '-03']
rem_numbers = [v[0] for k,v in numbers.items()]

# sort numbers
s_numbers = sorted(rem_numbers,key=lambda x: float(x))

# prepare (x,x_+,x_-) triples(sc)
x_triples = []
len_s = len(s_numbers)
for i,n in enumerate(s_numbers):
    if i == 0 or i == 1 or i == len_s -1 or i == len_s - 2:
        continue
    x_triple = []
    n_l1 = s_numbers[i - 1]
    n_l2 = s_numbers[i - 2]
    n_r1 = s_numbers[i + 1]
    n_r2 = s_numbers[i + 2]
    ld1 = abs(float(n) - float(n_l1))
    rd1 = abs(float(n) - float(n_r1))
    ld2 = abs(float(n) - float(n_l2))
    rd2 = abs(float(n) - float(n_r2))

    x_triple.append(n)

    if math.isclose(ld1, rd1):
        x_triple.append(random.choice([n_l1,n_r1]))
        if math.isclose(ld2, rd2):
            x_triple.append(random.choice([n_l2, n_r2]))
        elif ld2 < rd2:
            x_triple.append(n_l2)
        else:
            x_triple.append(n_r2)

    elif ld1 < rd1:
        x_triple.append(n_l1)

        if math.isclose(ld2, rd1):
            x_triple.append(random.choice([n_l2,n_r1]))
        elif ld2 < rd1:
            x_triple.append(n_l2)
        else:
            x_triple.append(n_r1)

    else:
        x_triple.append(n_r1)

        if math.isclose(rd2, ld1):
            x_triple.append(random.choice([n_r2, n_l1]))
        elif rd2 < ld1:
            x_triple.append(n_r2)
        else:
            x_triple.append(n_l1)

    # debug: check the triple
    if not is_valid_triple([float(n) for n in x_triple]):
        print(x_triple)
    assert is_valid_triple([float(n) for n in x_triple]),print(x_triple)

    x_triples.append(x_triple)

## deal with the boundary case
## i = 0,1,n,n-1

# i = 0
x_triple = [s_numbers[0],s_numbers[1],s_numbers[2]]
if is_valid_triple([float(n) for n in x_triple]):
    x_triples.append(x_triple)

# i = 1
n_l1 = s_numbers[0]
n_r1 = s_numbers[2]
n_r2 = s_numbers[3]
ld1 = abs(float(s_numbers[1])-float(n_l1))
rd1 = abs(float(s_numbers[1])-float(n_r1))
rd2 = abs(float(s_numbers[1])-float(n_r2))
x_triple = []
x_triple.append(s_numbers[1])
if math.isclose(ld1,rd1):
    x_triple.append(random.choice([n_l1,n_r1]))
    x_triple.append(n_r2)
elif ld1 < rd1:
    x_triple.append(n_l1)
    x_triple.append(n_r1)
else:
    x_triple.append(n_r1)
    if math.isclose(ld1,rd2):
        x_triple.append(random.choice([n_l1,n_r2]))
    elif ld1 < rd2:
        x_triple.append(n_l1)
    else:
        x_triple.append(n_r2)
if is_valid_triple([float(n) for n in x_triple]):
    x_triples.append(x_triple)

# i = n
x_triple = [s_numbers[-1],s_numbers[-2],s_numbers[-3]]
if is_valid_triple([float(n) for n in x_triple]):
    x_triples.append(x_triple)

# i = n-1
x_triple = []
x_triple.append(s_numbers[-2])
n_r1 = s_numbers[-1]
n_l1 = s_numbers[-3]
n_l2 = s_numbers[-4]
ld1 = abs(float(s_numbers[-2])-float(n_l1))
ld2 = abs(float(s_numbers[-2])-float(n_l2))
rd1 = abs(float(s_numbers[-2])-float(n_r1))
if math.isclose(ld1,rd1):
    x_triple.append(random.choice([n_l1,n_r1]))
    x_triple.append(n_l2)
elif ld1 > rd1:
    x_triple.append(n_r1)
    x_triple.append(n_l1)
else:
    x_triple.append(n_l1)
    if math.isclose(rd1,ld2):
        x_triple.append(random.choice([n_r1,n_l2]))
    elif rd1 < ld2:
        x_triple.append(n_r1)
    else:
        x_triple.append(n_l2)
if is_valid_triple([float(n) for n in x_triple]):
    x_triples.append(x_triple)

print('number of sc tests: %d' %(len(x_triples)))
with open('scmag_str.pickle','wb') as f:
    pickle.dump(x_triples,f,pickle.HIGHEST_PROTOCOL)

# prepare ova according to sc
obtain_OVA_from_SC(x_triples)