import numpy as np

inter_nums_ova = np.load('sel_orig_nums_ova.npy',allow_pickle=True) # B x n-2 x 3

k = 100
inter_nums_ova_fl = inter_nums_ova.astype(np.float)
diff = np.abs(inter_nums_ova_fl[:,:,0] - inter_nums_ova_fl[:,:,2]) # B x n-2
# get the id of the first k smallest
# ref: https://stackoverflow.com/a/34226816/6609622
idx = np.argpartition(diff,k)[:,:k] # B x k
# index the first k elements
# ref: https://stackoverflow.com/a/48997870
sc_k_ova = inter_nums_ova[np.arange(len(idx))[:,None], idx] # B x k x 3
np.save('sel_orig_nums_sc_k',sc_k_ova)