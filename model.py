import time

import scipy.linalg as scipy_linalg
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from torch.utils.data import DataLoader


class NeuralNetworkMapping(nn.Module):

    def __init__(self,nh1,nout):
        super(NeuralNetworkMapping, self).__init__()
        self.linear = nn.Linear(300,nh1)
        self.activation = nn.Tanh()
        self.out = nn.Linear(nh1,nout)
        self.out_activation = nn.Tanh()

    def forward(self, x):
        return self.out_activation(self.out(self.activation(self.linear(x))))
        # return self.activation(self.linear(x))

class SubspaceMapping(nn.Module):

    def __init__(self,dim,d):
        super(SubspaceMapping,self).__init__()
        # todo: make device explicit
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.W = nn.Parameter(torch.randn((dim, d),device=self.device))
        # self.W = torch.randn((dim, d), requires_grad=True,device=self.device)
        torch.nn.init.orthogonal_(self.W)
        self.W.data = self.W.T.data  # col orthognol

    def project(self):
        # project the variables back to the feasible set

        # find the nearest col orthogonal matrix
        # ref: http://people.csail.mit.edu/bkph/articles/Nearest_Orthonormal_Matrix.pdf
        # ref: https://math.stackexchange.com/q/2500881
        # ref: https://math.stackexchange.com/a/2215371
        # ref: https://github.com/pytorch/pytorch/issues/28293
        try:
            u, s, vh = scipy_linalg.svd(self.W.data.cpu().numpy(),full_matrices=False,lapack_driver='gesvd')
            self.W.data = torch.from_numpy(u @ vh).to(self.device)
            assert not torch.isnan(self.W.data).any(), 'W has nan values'
        except RuntimeError as e:
            # handle following error
            # RuntimeError: svd_cuda: the updating process of SBDSDC did not converge (error: 1)
            print('Runtime error: {0}'.format(e))
            print(self.W)
            print(self.W.grad)
            print('saving relevant info...')
            torch.save(self.W,'runtime_error_W.pt')
            torch.save(self.W.grad,'runtime_error_W_grad.pt')
            exit()

    def forward(self, x):
        # Bxd / B x n-2 x d
        # return shape Bxs / B x n-2 x s
        return x @ self.W

    def forward2origin(self, x):
        return x @ self.W @ self.W.T

class AxisMapping(nn.Module):

    def __init__(self,d):
        super(AxisMapping,self).__init__()
        # todo: make device explicit
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.w = nn.Parameter(torch.randn(d,device=self.device))
        self.w.data = self.w.data / torch.norm(self.w).data # unit

    def project(self):
        self.w.data = self.w.data / torch.norm(self.w).data # unit

    def forward(self, x):
        # Bxd / Bxkxd
        # return shape B/Bxk
        return x @ self.w


class PowerHingeLoss(_Loss):

    def forward(self, x):
        return torch.max(torch.zeros_like(x), 1 + x) ** 6

class LogisticLoss(_Loss):

    def __init__(self,beta):
        super(LogisticLoss,self).__init__()
        self.beta = beta

    def forward(self, x):
        # important: the element order of z is not the same with that in x
        y_pos = torch.exp(-self.beta * x[x >= 0])
        z_pos = (1 + y_pos) ** (-1)
        y_neg = torch.exp(self.beta * x[x < 0])
        z_neg = 1 - (1 + y_neg) ** (-1)

        # y = 1 / (1+torch.exp(-self.beta*x))
        # return y
        return torch.cat((z_pos, z_neg))

class SubspaceMag(nn.Module):

    def __init__(self, mapping_model, distance_type, loss):
        super(SubspaceMag, self).__init__()
        self.mapping = mapping_model
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.loss = loss
        if distance_type == 'euclidean':
            self.distance = F.pairwise_distance
        elif distance_type == 'cosine':
            self.distance = lambda x,y: 1 - F.cosine_similarity(x,y)

    def get_loss(self, dp, dm):

        objs = self.loss(dp - dm)
        loss = torch.mean(objs)
        acc = torch.mean((dp.data < dm.data).float())  # mini-batch acc, batch size is same as dp/dm

        return loss,acc

class OVAModel(SubspaceMag):

    def forward(self,mini_P_x, mini_P_xp, mini_P_xms):
        map_min_P_x = self.mapping(mini_P_x)
        map_min_P_xp = self.mapping(mini_P_xp)
        map_min_P_xms = self.mapping(mini_P_xms)
        dp = self.distance(map_min_P_x,map_min_P_xp)
        dm = torch.min(self.distance(map_min_P_x[:,:,None],map_min_P_xms.transpose(1,2)),dim=1)[0]

        return dp,dm

class RegularizedOVAModel(OVAModel):

    def __init__(self, mapping_model, distance_type, loss, lamb):
        super(RegularizedOVAModel, self).__init__(mapping_model, distance_type, loss)
        self.lamb = lamb

    def forward(self,mini_P_x, mini_P_xp, mini_P_xms, mini_emb=None):
        # https: // stackoverflow.com / a / 54155637 / 6609622
        # https: // stackoverflow.com / a / 805081 / 6609622
        dp,dm = super().forward(mini_P_x, mini_P_xp, mini_P_xms)
        mini_emb_subspace_origin = self.mapping.forward2origin(mini_emb)  # B'xd
        norm_ratio = torch.norm(mini_emb_subspace_origin, dim=1) / torch.norm(mini_emb, dim=1)

        return dp,dm,norm_ratio

    def get_loss(self, dp, dm, norm_ratio=None):
        objs = self.loss(dp - dm)
        loss = torch.mean(objs) - self.lamb * torch.mean(norm_ratio)
        acc = torch.mean((dp.data < dm.data).float())  # mini-batch acc, batch size is same as dp/dm

        return loss,acc

class SCModel(SubspaceMag):

    def forward(self, mini_P_x, mini_P_xp, mini_P_xm):
        map_min_P_x = self.mapping(mini_P_x)
        map_min_P_xp = self.mapping(mini_P_xp)
        map_min_P_xm = self.mapping(mini_P_xm)
        dp = self.distance(map_min_P_x, map_min_P_xp)
        dm = self.distance(map_min_P_x, map_min_P_xm)

        return dp, dm

class AxisOrdering(nn.Module):

    def __init__(self, mapping_model, loss):
        super(AxisOrdering, self).__init__()
        self.mapping = mapping_model
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.loss = loss

    def criterion(self, ap, am):

        objs = self.loss(ap - am)
        loss = torch.mean(objs)
        acc = torch.mean((ap.data < am.data).float())  # mini-batch acc, batch size is same as dp/dm

        return loss,acc

    def forward(self, mini_P_x, mini_P_xms):
        ap = self.mapping(mini_P_x) # B
        ams = self.mapping(mini_P_xms)# Bxk
        am = torch.min(ams,dim=1)[0] # B
        return ap,am

    def evaluate(self,data_batches):
        """
        evaluate w on data_batches, i.e., all data in data batches
        :param data_batches:
        :return:
        """
        losses, accs = [],[]
        # start = time.time()
        # num_mini_batches = 0

        # print(self.mapping.W)

        for mini_batch in data_batches:
            mini_P_x, mini_P_xms = mini_batch

            mini_P_x = mini_P_x.to(self.device)  # can set non_blocking=True
            mini_P_xms = mini_P_xms.to(self.device)

            ap, am = self.forward(mini_P_x, mini_P_xms)
            objs = self.loss(ap.data - am.data)
            loss = torch.sum(objs)
            acc = torch.sum((ap.data < am.data).float())

            print(acc)

            losses.append(loss)
            accs.append(acc)
            # num_mini_batches += 1
        # print(num_mini_batches)
        # print("evaluate: ", time.time() - start)
        loss = torch.sum(torch.stack(losses))/len(data_batches.dataset)
        acc = torch.sum(torch.stack(accs))/len(data_batches.dataset)
        return acc.item(),loss.item()