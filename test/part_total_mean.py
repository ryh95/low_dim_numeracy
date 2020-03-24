import numpy as np

from itertools import islice
from random import randint

def random_chunk(li, min_chunk=10, max_chunk=30):
    it = iter(li)
    while True:
        nxt = list(islice(it,randint(min_chunk,max_chunk)))
        if nxt:
            yield nxt
        else:
            break

a = np.random.rand(100).tolist()
b = list(random_chunk(a))
total_l = 0
for e in b:
    total_l += len(e)
assert total_l == 100

part_mean = []
for e in b:
    part_mean.append(np.mean(e))

print(np.mean(part_mean))
print(np.mean(a))