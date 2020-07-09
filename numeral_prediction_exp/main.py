
# prepare vocab embedding and number embedding

## extract vocab embedding from texts
import pickle
import itertools
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

from config import EMB_DIR, EMB
from numeral_prediction_exp.evaluator import Evaluator
from utils import vocab2vec

with open('data/processed_sentences.pickle','rb') as f:
    sens = pickle.load(f)

if not Path('data/vocab_embedding.pkl').exists():
    vocab = list(set(itertools.chain.from_iterable((sens))))
    emb_dict,emb = vocab2vec(vocab,'data','vocab_embedding',EMB,['pickle','npy'],oov_handle='none')
else:
    with open('data/vocab_embedding.pkl','rb') as f:
        emb_dict = pickle.load(f)
    emb = np.load('data/vocab_embedding.npy')

## load number embedding
with open('../data/embs/nums1-3_word2vec-wiki.pkl','rb') as f:
    num_emb_dict = pickle.load(f)
num_emb_dict = OrderedDict(num_emb_dict)
nums = np.load('../data/vocab/nums1-3.npy')

# load projected number embedding
state_dict = torch.load('data/160_300_beta_18_cosine_0.0025_50.pt')
Q = state_dict['mapping.W']
del state_dict

num_emb = torch.tensor(np.stack([v for v in num_emb_dict.values()])).float().to('cuda')
projected_num_emb = num_emb @ Q @ Q.T
projected_num_emb_dict = OrderedDict((k,projected_num_emb[i,:].cpu().numpy()) for i,k in enumerate(num_emb_dict.keys()))

# compare original embedding with our projected embedding using evaluator
evaluator = Evaluator(emb_dict, num_emb_dict)
MdAE, MdAPE, AVGR = evaluator.evaluate_SA(nums, sens, 5)
print(MdAE,MdAPE,AVGR)

evaluator = Evaluator(emb_dict, projected_num_emb_dict)
MdAE, MdAPE, AVGR = evaluator.evaluate_SA(nums, sens, 5)
print(MdAE,MdAPE,AVGR)