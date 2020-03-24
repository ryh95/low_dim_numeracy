from collections import defaultdict
from os.path import join

from tqdm import tqdm
import numpy as np
from config import EMB_DIR
from utils import is_number

fembs = ['word2vec-wiki','word2vec-giga','glove-wiki','glove-giga','fasttext-wiki','fasttext-giga']
numbers_list = []
for femb in fembs:
    # collect numbers from word embedding
    numbers = defaultdict(list)
    with open(join(EMB_DIR,femb+'.txt'),'r') as f:
        f.readline()
        for line in tqdm(f):
            word, *vec = line.rstrip().split(' ')
            word = word.split('_')[0]
            if is_number(word):
               numbers[float(word)].append(word)

    # remove redudent numbers
    # e.g. ['-3', '-3.0', '-03']
    rem_numbers = [v[0] for k,v in numbers.items()]

    rem_numbers = [num for num in rem_numbers if float(num) > 0 and float(num) < 10000]

    numbers_list.append(rem_numbers)

saved_nums = set(numbers_list[0])
for nums in numbers_list[1:]:
    saved_nums &= set(nums)

print(len(saved_nums))
np.save('nums2',list(saved_nums))