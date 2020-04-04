import math
import os
import random
from os.path import join
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm

from utils import is_valid_triple
from config import DATA_DIR, SUB_MAG_EXP_DIR, VOCAB_DIR
from dataset import OVADataset, SCDataset


# own checkpoint can not be load if the package name is changed, so my checkpoint class is disabled
# class MyCheckpointSaver(CheckpointSaver):
#
#     def __init__(self,checkpoint_path, remove_func, **dump_options):
#         super(MyCheckpointSaver,self).__init__(checkpoint_path, **dump_options)
#         self.remove_func = remove_func
#
#     def __call__(self,res):
#         if self.remove_func:
#             res.specs['args']['func'] = None
#         dump(res, self.checkpoint_path, **self.dump_options)

def load_batched_samples(X, num_emb, pre_emb=True):
    if len(X.shape) == 3:
        batched_samples = OVADataset(X, num_emb)
    elif len(X.shape) == 2:
        batched_samples = SCDataset(X, num_emb)
    else:
        assert False
    if pre_emb:
        # preload all train_data into memory to save time
        mini_batchs = DataLoader(batched_samples, batch_size=128, num_workers=6)
        P_x, P_xp, P_xms = [], [], []
        for i, mini_batch in enumerate(mini_batchs):
            mini_P_x, mini_P_xp, mini_P_xms = mini_batch
            P_x.append(mini_P_x)
            P_xp.append(mini_P_xp)
            P_xms.append(mini_P_xms)
        P_x = torch.cat(P_x)
        P_xp = torch.cat(P_xp)
        P_xms = torch.cat(P_xms)
        batched_samples = TensorDataset(P_x, P_xp, P_xms)
    return batched_samples

def init_evaluate(dataset,distance_metric):
    losses, accs = [], []
    data_batches = DataLoader(dataset, batch_size=128,num_workers=0,pin_memory=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    for mini_batch in data_batches:
        mini_P_x, mini_P_xp, mini_P_xms = mini_batch
        mini_P_x = mini_P_x.to(device)  # can set non_blocking=True
        mini_P_xp = mini_P_xp.to(device)
        mini_P_xms = mini_P_xms.to(device)

        # Dp = torch.norm(mini_P_x - mini_P_xp,dim=1)
        Dp = distance_metric(mini_P_x,mini_P_xp)

        if len(mini_P_xms.size()) == 3:
            Dm = distance_metric(mini_P_x[:,:,None],mini_P_xms.transpose(1,2)).min(dim=1)[0]
            # Dm = torch.norm(mini_P_x[:,:,None] - mini_P_xms,dim=1).min(dim=1)[0]
        else:
            Dm = distance_metric(mini_P_x,mini_P_xms)
            # Dm = torch.norm(mini_P_x - mini_P_xms, dim=1)

        acc = torch.sum((Dp < Dm).float())

        accs.append(acc)
    return (torch.sum(torch.stack(accs))/len(dataset)).item()

def train_dev_test_split(data,ratios,fdata):
    """

    :param data: list, original sc/ova loaded data
    :param ratios: list, ratio, sum up to 1
    :return:
    """
    np.random.shuffle(data)
    begin,end = ratios[0],ratios[0]+ratios[1]
    begin = int(begin*len(data))
    end = int(end*len(data))
    train = data[:begin,:]
    dev = data[begin:end,:]
    test = data[end:,:]
    splited_data = [train,dev,test]
    for i,name in enumerate(['train','dev','test']):
        np.save(join(DATA_DIR, fdata+'_'+name),splited_data[i])
        # with open(join(DATA_DIR, fdata+'_'+name+'.pkl'), 'wb') as f:
        #     pickle.dump(splited_data[i], f, pickle.HIGHEST_PROTOCOL)


def prepare_sc(fnums):

    # sort numbers
    nums = np.load(join(VOCAB_DIR,fnums+'.npy'))
    s_numbers = sorted(nums,key=lambda x: float(x))

    # prepare (x,x_+,x_-) triples(sc)
    len_s = nums.size
    X_sc = np.empty((len_s,3),dtype=np.object)
    valid_id = np.ones(len_s)
    for i,n in enumerate(s_numbers):
        if i == 0 or i == 1 or i == len_s -1 or i == len_s - 2:
            continue
        x_sc = []
        n_l1 = s_numbers[i - 1]
        n_l2 = s_numbers[i - 2]
        n_r1 = s_numbers[i + 1]
        n_r2 = s_numbers[i + 2]
        ld1 = abs(float(n) - float(n_l1))
        rd1 = abs(float(n) - float(n_r1))
        ld2 = abs(float(n) - float(n_l2))
        rd2 = abs(float(n) - float(n_r2))

        x_sc.append(n)

        if math.isclose(ld1, rd1):
            x_sc.append(random.choice([n_l1,n_r1]))
            if math.isclose(ld2, rd2):
                x_sc.append(random.choice([n_l2, n_r2]))
            elif ld2 < rd2:
                x_sc.append(n_l2)
            else:
                x_sc.append(n_r2)

        elif ld1 < rd1:
            x_sc.append(n_l1)

            if math.isclose(ld2, rd1):
                x_sc.append(random.choice([n_l2,n_r1]))
            elif ld2 < rd1:
                x_sc.append(n_l2)
            else:
                x_sc.append(n_r1)

        else:
            x_sc.append(n_r1)

            if math.isclose(rd2, ld1):
                x_sc.append(random.choice([n_r2, n_l1]))
            elif rd2 < ld1:
                x_sc.append(n_r2)
            else:
                x_sc.append(n_l1)

        # debug: check the triple
        if is_valid_triple([float(n) for n in x_sc]):
            X_sc[i,:] = x_sc
        else:
            print(x_sc)
            valid_id[i] = 0

    ## deal with the boundary case
    ## i = 0,1,n,n-1

    # i = 0
    x_sc = [s_numbers[0],s_numbers[1],s_numbers[2]]
    if is_valid_triple([float(n) for n in x_sc]):
        X_sc[0,:] = x_sc
    else:
        print(x_sc)
        valid_id[0] = 0

    # i = 1
    n_l1 = s_numbers[0]
    n_r1 = s_numbers[2]
    n_r2 = s_numbers[3]
    ld1 = abs(float(s_numbers[1])-float(n_l1))
    rd1 = abs(float(s_numbers[1])-float(n_r1))
    rd2 = abs(float(s_numbers[1])-float(n_r2))
    x_sc = []
    x_sc.append(s_numbers[1])
    if math.isclose(ld1,rd1):
        x_sc.append(random.choice([n_l1,n_r1]))
        x_sc.append(n_r2)
    elif ld1 < rd1:
        x_sc.append(n_l1)
        x_sc.append(n_r1)
    else:
        x_sc.append(n_r1)
        if math.isclose(ld1,rd2):
            x_sc.append(random.choice([n_l1,n_r2]))
        elif ld1 < rd2:
            x_sc.append(n_l1)
        else:
            x_sc.append(n_r2)
    if is_valid_triple([float(n) for n in x_sc]):
        X_sc[1,:] = x_sc
    else:
        print(x_sc)
        valid_id[1] = 0

    # i = n
    x_sc = [s_numbers[-1],s_numbers[-2],s_numbers[-3]]
    if is_valid_triple([float(n) for n in x_sc]):
        X_sc[-1,:] = x_sc
    else:
        print(x_sc)
        valid_id[-1] = 0

    # i = n-1
    x_sc = []
    x_sc.append(s_numbers[-2])
    n_r1 = s_numbers[-1]
    n_l1 = s_numbers[-3]
    n_l2 = s_numbers[-4]
    ld1 = abs(float(s_numbers[-2])-float(n_l1))
    ld2 = abs(float(s_numbers[-2])-float(n_l2))
    rd1 = abs(float(s_numbers[-2])-float(n_r1))
    if math.isclose(ld1,rd1):
        x_sc.append(random.choice([n_l1,n_r1]))
        x_sc.append(n_l2)
    elif ld1 > rd1:
        x_sc.append(n_r1)
        x_sc.append(n_l1)
    else:
        x_sc.append(n_l1)
        if math.isclose(rd1,ld2):
            x_sc.append(random.choice([n_r1,n_l2]))
        elif rd1 < ld2:
            x_sc.append(n_r1)
        else:
            x_sc.append(n_l2)
    if is_valid_triple([float(n) for n in x_sc]):
        X_sc[-2,:] = x_sc
    else:
        print(x_sc)
        valid_id[-2] = 0

    valid_id = valid_id.astype(bool)
    X_sc = X_sc[valid_id,:]

    print('number of sc tests: %d' %(len(X_sc)))
    fX_sc = fnums+'_sc'
    np.save(join(SUB_MAG_EXP_DIR,'data',fX_sc),X_sc)
    return X_sc


def prepare_ova_with_sc(fX_sc):
    """

    :param X_sc: list of list of strs
    :return: none, save ova tests in DATA_DIR with ovamag_str.pkl
    """
    X_sc = np.load(join(SUB_MAG_EXP_DIR,'data',fX_sc+'.npy'),allow_pickle=True)
    number_set = set(np.array(X_sc).flat)
    number_array = sorted(number_set, key=lambda x: float(x))
    # max_num = number_array[-1]
    l_number_array = len(number_array)
    X_ova = np.empty((l_number_array,l_number_array-2,3),dtype=np.object)
    valid_id = np.ones(l_number_array)
    for i, x in tqdm(enumerate(number_array)):

        if i == 0 or i == l_number_array - 1:
            continue
        n_l1 = number_array[i - 1]
        n_r1 = number_array[i + 1]
        ld1 = abs(float(x) - float(n_l1))
        rd1 = abs(float(x) - float(n_r1))

        remain_numbers = number_set - set([x, n_l1, n_r1])

        if ld1 < rd1:
            xp = n_l1
            xm = n_r1
        elif rd1 < ld1:
            xp = n_r1
            xm = n_l1
        else:
            # resolve the boundary case
            # xp = random.choice([n_l1,n_r1])
            xp = n_l1
            xm = random.sample(remain_numbers,1)[0]
            # xm = max_num

        valid_test = 1
        for j,m in enumerate(list(remain_numbers) + [xm]):
            if is_valid_triple([float(n) for n in [x, xp, m]]):
                X_ova[i][j] = [x, xp, m]
            else:
                print([x, xp, m])
                valid_test = 0
                break

        if not valid_test:
            valid_id[i] = 0
            # ova_tests.append(one_test)

    # boundary cases

    x = number_array[0]
    xp = number_array[1]
    remain_numbers = number_set - set([x, xp])

    valid_test = 1
    for j,m in enumerate(remain_numbers):
        if is_valid_triple([float(n) for n in [x, xp, m]]):
            X_ova[0][j] = [x,xp,m]
        else:
            print([x,xp,m])
            valid_test = 0
            break

    if not valid_test:
        valid_id[0] = 0

    x = number_array[-1]
    xp = number_array[-2]
    remain_numbers = number_set - set([x, xp])

    valid_test = 1
    for j,m in enumerate(remain_numbers):
        if is_valid_triple([float(n) for n in [x, xp, m]]):
            X_ova[-1][j] = [x, xp, m]
        else:
            print([x,xp,m])
            valid_test = 0
            break

    if not valid_test:
        valid_id[-1] = 0

    valid_id = valid_id.astype(bool)
    X_ova = X_ova[valid_id,:,:]

    fX_ova = fX_sc[:-3]+'_ova'
    print('number of ova tests: %d' %(len(X_ova)))
    np.save(join(SUB_MAG_EXP_DIR, 'data', fX_ova), X_ova)
    return X_ova
    # with open('ovamag_str.pkl', 'wb') as f:
    #     pickle.dump(ova_tests, f, pickle.HIGHEST_PROTOCOL)

def prepare_sc_k_with_ova(fX_ova, k=100):

    X_ova = np.load(join(SUB_MAG_EXP_DIR, 'data', fX_ova+'.npy'), allow_pickle=True)  # B x n-2 x 3
    number_array = sorted(set(X_ova.flat), key=lambda x: float(x))
    # in order to remove the duplicate in the sc-k,
    X_sc_k = np.empty((X_ova.shape[0],k,3),dtype=np.object)

    for i in range(1,X_ova.shape[0]-1): # skip the first and last
        X_ova_i = X_ova[i,:,:] # n-2 x 3
        n_l1 = number_array[i-1]
        n_r1 = number_array[i+1]
        X_ova_i_fl = X_ova_i.astype(np.float)
        diff = np.abs(X_ova_i_fl[:, 0] - X_ova_i_fl[:, 2])  # n-2
        if X_ova_i[-1][2] != n_l1 and X_ova_i[-1][2] != n_r1:
            idx = np.argpartition(diff[:-1],k)[:k] # k
        else:
            idx = np.argpartition(diff, k)[:k]  # k
        X_sc_k[i, :, :] = X_ova_i[idx, :]

    # resolve the first and last
    X_ova_0_fl = X_ova[0].astype(np.float)
    diff = np.abs(X_ova_0_fl[:,0]-X_ova_0_fl[:,2]) # n-2
    idx = np.argpartition(diff, k)[:k]  # k
    X_sc_k[0] = X_ova[0][idx,:]

    X_ova_n_fl = X_ova[-1].astype(np.float)
    diff = np.abs(X_ova_n_fl[:, 0] - X_ova_n_fl[:, 2])  # n-2
    idx = np.argpartition(diff, k)[:k]  # k
    X_sc_k[-1] = X_ova[-1][idx, :]

    # X_ova_fl = X_ova.astype(np.float)
    # diff = np.abs(X_ova_fl[:, :, 0] - X_ova_fl[:, :, 2])  # B x n-2
    # # get the id of the first k smallest
    # # ref: https://stackoverflow.com/a/34226816/6609622
    # idx = np.argpartition(diff, k)[:, :k]  # B x k
    # # index the first k elements
    # # ref: https://stackoverflow.com/a/48997870
    # X_sc_k = X_ova[np.arange(len(idx))[:, None], idx]  # B x k x 3
    np.save(join(SUB_MAG_EXP_DIR,'data','_'.join(fX_ova.split('_')[:-1] + ['sc-k'])), X_sc_k)
    return X_sc_k