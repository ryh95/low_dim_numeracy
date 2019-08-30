import math
import pickle
import random

import numpy as np

from tqdm import tqdm

from config import EMB
from utils import is_number, is_valid_triple

# collect numbers from word embedding
numbers = set()
with open(EMB,'r') as f:
    for line in tqdm(f):
        word, *vec = line.rstrip().split(' ')
        if is_number(word):
           numbers.add(float(word))

# sort numbers
s_numbers = np.sort(list(numbers))

# prepare (x,x_+,x_-) triples(sc)
x_triples = []
for i,n in enumerate(s_numbers):
    if i == 0 or i == 1 or i == s_numbers.size -1 or i == s_numbers.size - 2:
        continue
    x_triple = []
    n_l1 = s_numbers[i - 1]
    n_l2 = s_numbers[i - 2]
    n_r1 = s_numbers[i + 1]
    n_r2 = s_numbers[i + 2]
    ld1 = abs(n - n_l1)
    rd1 = abs(n - n_r1)
    ld2 = abs(n - n_l2)
    rd2 = abs(n - n_r2)

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

    # check the triple
    if not is_valid_triple(x_triple):
        print(x_triple)
    assert is_valid_triple(x_triple),print(x_triple)

    x_triples.append(x_triple)

with open('scmag.pickle','wb') as f:
    pickle.dump(x_triples,f,pickle.HIGHEST_PROTOCOL)

## deal with the boundary case
## i = 0,1,n,n-1
x_triple = [s_numbers[0],s_numbers[1],s_numbers[2]]
assert is_valid_triple(x_triple)
x_triples.append(x_triple) # i = 0

# i = 1
n_l1 = s_numbers[0]
n_r1 = s_numbers[2]
n_r2 = s_numbers[3]
ld1 = abs(s_numbers[1]-n_l1)
rd1 = abs(s_numbers[1]-n_r1)
rd2 = abs(s_numbers[1]-n_r2)
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
assert is_valid_triple(x_triple)
x_triples.append(x_triple)
# i = n
x_triple = [s_numbers[-1],s_numbers[-2],s_numbers[-3]]
assert is_valid_triple(x_triple)
x_triples.append(x_triple)
# i = n-1
x_triple = []
x_triple.append(s_numbers[-2])
n_r1 = s_numbers[-1]
n_l1 = s_numbers[-3]
n_l2 = s_numbers[-4]
ld1 = abs(s_numbers[-2]-n_l1)
ld2 = abs(s_numbers[-2]-n_l2)
rd1 = abs(s_numbers[-2]-n_r1)
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
assert is_valid_triple(x_triple)
x_triples.append(x_triple)