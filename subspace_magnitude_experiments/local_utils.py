import pickle
import random
import time
from math import inf
from os.path import join

import torch
from skopt.utils import use_named_args, dump
from torch import autograd
from torch.utils.data import DataLoader, TensorDataset

from config import DATA_DIR
from dataset import OVADataset, SCDataset
from model import OVA_Subspace_Model
from skopt.callbacks import CheckpointSaver
import scipy.linalg as LA
import numpy as np

class Minimizer(object):

    def __init__(self, base_workspace, optimize_types, mini_func):
        self.base_workspace = base_workspace
        self.model = base_workspace['model']
        self.optimizer = base_workspace['optimizer']
        self.mini_func = mini_func
        self.optimize_types = optimize_types

    def objective(self, feasible_point):

        optimize_workspace = {type:type_values for type,type_values in zip(self.optimize_types,feasible_point)}

        # combine two workspace
        workspace = {**self.base_workspace,**optimize_workspace}

        best_acc = -inf

        mini_batchs = DataLoader(workspace['train_data'], batch_size=workspace['mini_batch_size'], shuffle=True, num_workers=0, pin_memory=True)
        if 'val_data' in workspace:
            mini_batchs_val = DataLoader(workspace['val_data'], batch_size=workspace['mini_batch_size'], shuffle=True, num_workers=0, pin_memory=True)
        if 'test_data' in workspace:
            mini_batchs_test = DataLoader(workspace['test_data'], batch_size=workspace['mini_batch_size'], shuffle=True, num_workers=0, pin_memory=True)

        model = self.model(workspace['subspace_dim'],workspace['emb_dim'],workspace['beta'],workspace['distance_metric'])

        # acc, loss = ova_model.evaluate(mini_batchs)
        # print('init specialized acc: ', acc)

        # SGD performs poorly, the reason is not clear
        # optimizer = torch.optim.SGD([W],lr,momentum=0.9)
        optimizer = self.optimizer(model.parameters(), workspace['lr'])

        for t in range(workspace['n_epochs']):

            if workspace['train_verbose']:
                print('epoch number: ', t)
                start = time.time()

            for i,mini_batch in enumerate(mini_batchs):

                mini_P_x, mini_P_xp, mini_P_xms = mini_batch

                mini_P_x = mini_P_x.to(model.device)  # can set non_blocking=True
                mini_P_xp = mini_P_xp.to(model.device)
                mini_P_xms = mini_P_xms.to(model.device)

                dp,dm = model(mini_P_x, mini_P_xp, mini_P_xms)
                loss,acc = model.criterion(dp,dm)

                # print(loss)
                optimizer.zero_grad()

                with autograd.detect_anomaly():
                    # avoid nan gradient
                    loss.backward()

                # if 'val_data' not in workspace:
                if workspace['select_inter_model']:
                    if i % 5 == 0:
                        if acc.item() > best_acc:
                            best_W = model.W.data.clone()
                            best_acc = acc.item()
                            if workspace['train_verbose']:
                                print('specialized acc: ', best_acc)

                optimizer.step()

                model.project()

            if workspace['train_verbose']:
                print("train: ", time.time() - start)

        # print("Deviation from the constraint: ",torch.norm(best_W.T @ best_W - torch.eye(dim).to(device)).item())
        if workspace['select_inter_model']:
            model.W = torch.nn.Parameter(best_W)
        if workspace['save_model']:
            fname = '_'.join([str(i) for i in [workspace['subspace_dim'], workspace['emb_dim'], workspace['beta'],
                                       workspace['distance_metric'], f"{workspace['lr']:.4f}",workspace['n_epochs']]])+'.pt'
            torch.save(model.state_dict(),fname)
        evaluate_accs = {}
        for data in workspace['eval_data']:
            if data == 'val':
                print('evaluate on validation set')
                evaluate_acc, _ = model.evaluate(mini_batchs_val)
            elif data == 'train':
                print('evaluate on training set')
                evaluate_acc, evaluate_loss = model.evaluate(mini_batchs)
            elif data == 'test':
                print('evaluate on test set')
                evaluate_acc, evaluate_loss = model.evaluate(mini_batchs_test)
            evaluate_accs[data] = evaluate_acc

        # print(workspace['working_status'])
        if workspace['working_status'] == 'optimize':
            return -evaluate_accs['val']
        elif workspace['working_status'] == 'infer':
            return evaluate_accs['test']
        elif workspace['working_status'] == 'eval':
            return evaluate_accs

    def minimize(self,space,**min_args):

        return self.mini_func(self.objective, space, **min_args)

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

def load_dataset(fdata,emb_conf,pre_load=True):
    if 'ova' in fdata:
        data = OVADataset(fdata,emb_conf)
    elif 'sc' in fdata:
        data = SCDataset(fdata,emb_conf)
    else:
        assert False
    if pre_load:
        # preload all train_data into memory to save time
        mini_batchs = DataLoader(data, batch_size=128, num_workers=8)
        P_x, P_xp, P_xms = [], [], []
        for i, mini_batch in enumerate(mini_batchs):
            mini_P_x, mini_P_xp, mini_P_xms = mini_batch
            P_x.append(mini_P_x)
            P_xp.append(mini_P_xp)
            P_xms.append(mini_P_xms)
        P_x = torch.cat(P_x)
        P_xp = torch.cat(P_xp)
        P_xms = torch.cat(P_xms)
        data = TensorDataset(P_x, P_xp, P_xms)
        data.number_emb_source = emb_conf['emb_fname']
    return data

def init_evaluate(dataset,distance_metric):
    losses, accs = [], []
    data_batches = DataLoader(dataset, batch_size=128)
    for mini_batch in data_batches:
        mini_P_x, mini_P_xp, mini_P_xms = mini_batch

        # Dp = torch.norm(mini_P_x - mini_P_xp,dim=1)
        Dp = distance_metric(mini_P_x,mini_P_xp)

        if len(mini_P_xms.size()) == 3:
            Dm = distance_metric(mini_P_x,mini_P_xms).min(dim=1)[0]
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