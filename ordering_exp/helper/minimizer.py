from math import inf
import time
import torch
from torch import autograd
from torch.utils.data import DataLoader, TensorDataset

from model import AxisMapping


class Minimizer(object):

    def __init__(self, base_workspace, optimize_types, mini_func):
        self.base_workspace = base_workspace
        self.model = base_workspace['model']
        self.loss = base_workspace['loss']
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

        if workspace['mapping_type'] == 'subspace':
            mapping_model = AxisMapping(workspace['emb_dim'])
        else:
            assert False

        if 'loss_params' in workspace:
            loss = self.loss(**workspace['loss_params'])
        else:
            loss = self.loss()

        model = self.model(mapping_model,loss)

        if 'val_data' in workspace:
            # init evaluate
            print('evaluate on val data')
            evaluate_acc, _ = model.evaluate(mini_batchs_val)
            print(evaluate_acc)

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

                mini_P_x, mini_P_xms = mini_batch # Bxd/Bxkxd
                mini_P_x = mini_P_x.to(model.device)  # can set non_blocking=True
                mini_P_xms = mini_P_xms.to(model.device)

                ap,am = model(mini_P_x, mini_P_xms)
                loss,acc = model.criterion(ap,am)

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
                            best_W = model.W.data.clone()
                            best_acc = acc.item()
                            if workspace['train_verbose']:
                                print('specialized acc: ', best_acc)

                optimizer.step()

                if isinstance(mapping_model,AxisMapping):
                    model.mapping.project()
                    # print(model.mapping.W)

            if workspace['train_verbose']:
                print("train: ", time.time() - start)

        # print("Deviation from the constraint: ",torch.norm(best_W.T @ best_W - torch.eye(dim).to(device)).item())
        if workspace['select_inter_model']:
            model.W = torch.nn.Parameter(best_W)
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
        evaluate_accs = {}
        for data in workspace['eval_data']:
            if data == 'val':
                print('evaluate on validation set')
                # print(model.mapping.W)
                evaluate_acc, _ = model.evaluate(mini_batchs_val)
            elif data == 'train':
                print('evaluate on training set')
                evaluate_acc, evaluate_loss = model.evaluate(mini_batchs)
            elif data == 'test':
                print('evaluate on test set')
                evaluate_acc, evaluate_loss = model.evaluate(mini_batchs_test)
            else:
                assert False
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