import os
from os.path import join

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = join(BASE_DIR,'data')
VOCAB_DIR = join(DATA_DIR,'vocab')
EMB_DIR = join(DATA_DIR,'embs')
RESULTS_DIR = join(DATA_DIR,'archive_results')

SUB_MAG_EXP_DIR = join(BASE_DIR,'sub_mag_exp')
ORD_EXP_DIR = join(BASE_DIR,'ordering_exp')
EMB = join(EMB_DIR,'glove.6B.300d.txt')