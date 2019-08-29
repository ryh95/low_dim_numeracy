import os
from os.path import join

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = join(BASE_DIR,'data')
EMB_DIR = join(DATA_DIR,'embs')

EMB = join(EMB_DIR,'glove.6B.300d.txt')