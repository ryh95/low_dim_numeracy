from math import inf
import time
import torch
from torch import autograd
from torch.utils.data import DataLoader, TensorDataset

from model import SubspaceMapping, NeuralNetworkMapping

class Minimizer(object):

    def __init__(self, base_workspace, optimize_types, mini_func):
        self.base_workspace = base_workspace
        self.model = base_workspace['model']
        self.loss = base_workspace['loss']
        self.optimizer = base_workspace['optimizer']
        self.mini_func = mini_func
        self.optimize_types = optimize_types

    def prepare_workspace(self,feasible_point):

        # optimize_workspace = {type:type_values for type,type_values in zip(self.optimize_types,feasible_point)}
        loss_params, optimize_workspace = {}, {}
        for type, type_values in zip(self.optimize_types, feasible_point):
            if type.startswith('loss__'):
                loss_params[type[len('loss__'):]] = type_values
                continue
            optimize_workspace[type] = type_values
        optimize_workspace['loss_params'] = loss_params

        # combine two workspace
        workspace = {**self.base_workspace, **optimize_workspace}

        return workspace

    def prepare_models(self,workspace):

        if workspace['mapping_type'] == 'subspace':
            mapping_model = SubspaceMapping(workspace['subspace_dim'], workspace['emb_dim'])
        elif workspace['mapping_type'] == 'nn':
            mapping_model = NeuralNetworkMapping(workspace['n_hidden1'], workspace['n_out'])
            # todo: solve the following
            mapping_model = mapping_model.cuda()
        else:
            assert False

        if 'loss_params' in workspace:
            loss = self.loss(**workspace['loss_params'])
        else:
            loss = self.loss()

        if self.base_workspace['model_type'] == 'normal':
            model = self.model(mapping_model,workspace['distance_metric'],loss)
        elif self.base_workspace['model_type'] == 'regularized':
            model = self.model(mapping_model,workspace['distance_metric'],loss,workspace['lamb'])
        else:
            assert False
        return model

    def train(self,workspace,model,data):

        # SGD performs poorly, the reason is not clear
        # optimizer = torch.optim.SGD([W],lr,momentum=0.9)
        optimizer = self.optimizer(model.parameters(), workspace['lr'])
        mini_batchs = DataLoader(data, batch_size=workspace['mini_batch_size'], shuffle=True,
                                 num_workers=0, pin_memory=True)
        best_acc = -inf
        for t in range(workspace['n_epochs']):

            if workspace['train_verbose']:
                print('epoch number: ', t)
                start = time.time()

            for i, mini_batch in enumerate(mini_batchs):

                mini_P_x, mini_P_xp, mini_P_xms = mini_batch

                mini_P_x = mini_P_x.to(model.device)  # can set non_blocking=True
                mini_P_xp = mini_P_xp.to(model.device)
                mini_P_xms = mini_P_xms.to(model.device)

                if self.base_workspace['model_type'] == 'regularized':
                    mini_emb = workspace['train_data'].number_emb
                    mini_emb = mini_emb.to(model.device)
                    dp, dm, norm_ratio = model(mini_P_x, mini_P_xp, mini_P_xms, mini_emb)
                    loss, acc = model.get_loss(dp, dm, norm_ratio)
                elif self.base_workspace['model_type'] == 'normal':
                    dp, dm = model(mini_P_x, mini_P_xp, mini_P_xms)
                    loss, acc = model.get_loss(dp, dm)
                else:
                    assert False

                print(acc)

                # print(loss)
                optimizer.zero_grad()

                with autograd.detect_anomaly():
                    # avoid nan gradient
                    loss.backward()

                # if 'val_data' not in workspace:
                if workspace['select_inter_model']:
                    if i % 5 == 0:
                        if acc.item() > best_acc:
                            model.mapping.best_W = model.mapping.W.data.clone()
                            best_acc = acc.item()
                            if workspace['train_verbose']:
                                print('specialized acc: ', best_acc)

                optimizer.step()

                if isinstance(model.mapping, SubspaceMapping):
                    model.mapping.project()
                    # print(model.mapping.W)

            if workspace['train_verbose']:
                print("train time per epoch: ", time.time() - start)

    def save_models(self,workspace,model):

        if workspace['select_inter_model']:
            model.mapping.W = torch.nn.Parameter(model.mapping.best_W)
        if workspace['save_model']:
            if 'loss_params' in workspace:
                loss_str = '_'.join([str(k) + '_' + str(v) for k, v in workspace['loss_params'].items()])
            else:
                loss_str = ''

            if workspace['mapping_type'] == 'nn':
                fname = '_'.join([str(i) for i in [workspace['n_hidden1'], workspace['n_out'],
                                                   workspace['emb_dim'], loss_str,
                                                   workspace['distance_metric'], f"{workspace['lr']:.4f}",
                                                   workspace['n_epochs']]]) + '.pt'
            elif workspace['mapping_type'] == 'subspace':
                fname = '_'.join([str(i) for i in [workspace['subspace_dim'], workspace['emb_dim'], loss_str,
                                                   workspace['distance_metric'], f"{workspace['lr']:.4f}",
                                                   workspace['n_epochs']]]) + '.pt'
            else:
                assert False
            torch.save(model.state_dict(),fname)

    def objective(self, feasible_point):

        workspace = self.prepare_workspace(feasible_point)

        model = self.prepare_models(workspace)

        # if 'val_data' in workspace:
        #     # init evaluate
        #     print('evaluate on val data')
        #     evaluate_acc, _ = model.evaluate(workspace['val_data'])
        #     print(evaluate_acc)

        self.train(workspace,model,workspace['train_data'])

        self.save_models(workspace,model)

        acc = self.evaluator.evaluate(model)

        return -acc

    def minimize(self,space,**min_args):

        return self.mini_func(self.objective, space, **min_args)