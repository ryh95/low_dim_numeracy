import time
from math import inf
from os.path import join

import torch
from skopt.utils import use_named_args, dump
from torch import autograd
from torch.utils.data import DataLoader, TensorDataset

from config import DATA_DIR
from dataset import OVADataset
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

        model = self.model(workspace['subspace_dim'],workspace['emb_dim'],workspace['beta'])

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

                dp,dm = model(mini_P_x, mini_P_xp, mini_P_xms)
                loss,acc = model.criterion(dp,dm)

                # print(loss)
                optimizer.zero_grad()

                with autograd.detect_anomaly():
                    # avoid nan gradient
                    loss.backward()

                if i % 5:
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
        model.W = torch.nn.Parameter(best_W)
        evaluate_acc, evaluate_loss = model.evaluate(mini_batchs)
        return -evaluate_acc

    def minimize(self,space,**min_args):

        return self.mini_func(self.objective, space, **min_args)

class MyCheckpointSaver(CheckpointSaver):

    def __init__(self,checkpoint_path, remove_func, **dump_options):
        super(MyCheckpointSaver,self).__init__(checkpoint_path, **dump_options)
        self.remove_func = remove_func

    def __call__(self,res):
        if self.remove_func:
            res.specs['args']['func'] = None
        dump(res, self.checkpoint_path, **self.dump_options)

def load_dataset(emb_fname,pre_load=True):

    train_data = OVADataset(join(DATA_DIR,'ovamag_str.pkl'), {"emb_fname": emb_fname})
    if pre_load:
        # preload all train_data into memory to save time
        mini_batchs = DataLoader(train_data, batch_size=128, num_workers=8)
        P_x, P_xp, P_xms = [], [], []
        for i, mini_batch in enumerate(mini_batchs):
            mini_P_x, mini_P_xp, mini_P_xms = mini_batch
            P_x.append(mini_P_x)
            P_xp.append(mini_P_xp)
            P_xms.append(mini_P_xms)
        P_x = torch.cat(P_x)
        P_xp = torch.cat(P_xp)
        P_xms = torch.cat(P_xms)
        train_data = TensorDataset(P_x, P_xp, P_xms)
        train_data.number_emb_source = emb_fname
    return train_data

def init_evaluate(dataset):
    losses, accs = [], []
    data_batches = DataLoader(dataset, batch_size=128)
    for mini_batch in data_batches:
        mini_P_x, mini_P_xp, mini_P_xms = mini_batch

        Dp = LA.norm(mini_P_x - mini_P_xp,axis=1)
        Dm = LA.norm(mini_P_x[:,:,None] - mini_P_xms,axis=1).min(axis=1)
        acc = np.mean(Dp<Dm)

        accs.append(acc)
    print('original acc: ',np.mean(accs))