# %%
# count the number of documents
# n_doc = 0
# with open('enwiki_20170220.txt', 'r') as f:
#     for l in f:
#         if l.startswith('<doc'):
#             n_doc += 1
# print(n_doc)
# %%
# sample 5 document
import pickle
from collections import deque

# sample_doc, counter = [], 0
# with open('enwiki_20170220.txt', 'r') as f:
#     for l in f:
#         sample_doc.append(l)
#         if l.startswith('</doc>'): counter += 1
#         if counter == 5: break
# %%
# sample_doc
# %%
import math
# %%
# 1. split doc into 50 pieces
# 2. each piece use stanford stanze tokenize and split sentences
# 3. extract sentences that contain numbers and save
# %%
import stanza
from tqdm import tqdm

nlp = stanza.Pipeline(lang='en', processors='tokenize')
# %%
# load numbers
import numpy as np

nums = np.load('../data/vocab/nums1-3.npy')
set_nums = set(nums)
# %%
n_doc_p = math.ceil(5339723/50000)
res_sens, doc_counter = deque(), 0
with open('enwiki_20170220.txt', 'r') as f:
    doc_p, is_end = '', False
    for l in tqdm(f):
        if l.startswith('<doc'):
            is_head = True
            continue
        if is_head:
            is_head = False
            continue
        if l.startswith('</doc>'):
            is_end = True
            doc_counter += 1

        if (doc_counter % n_doc_p == 0 and is_end == True) or doc_counter == 5339723:
            print(doc_counter,len(res_sens))
            doc = nlp(doc_p)
            for sen in doc.sentences:
                # if a target number appeared in sentence, the sentence will be collected
                sen_text = [t.text for t in sen.tokens]
                if set_nums & set(sen_text):
                    res_sens.append(sen_text)
            doc_p = ''
            is_end = False

        if not l.startswith('</doc>'): doc_p += l

        if len(res_sens) >= 100000: break

with open('number_sens.pickle','wb') as f:
    pickle.dump(res_sens,f,pickle.HIGHEST_PROTOCOL)
