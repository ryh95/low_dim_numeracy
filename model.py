import time

import scipy.linalg as scipy_linalg
import torch
from torch import nn
import torch.nn.functional as F

class NeuralNetworkMapping(nn.Module):

    def __init__(self,nh1,nout):
        super(NeuralNetworkMapping, self).__init__()
        self.linear = nn.Linear(300,nh1)
        self.activation = nn.Tanh()
        self.out = nn.Linear(nh1,nout)
        self.out_activation = nn.Tanh()

    def forward(self, x):
        return self.out_activation(self.out(self.activation(self.linear(x))))

class SubspaceMapping(nn.Module):

    def __init__(self,dim,d):
        super(SubspaceMapping,self).__init__()
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

class Model(nn.Module):

    def __init__(self,mapping_model,distance_metric,beta):
        super(Model,self).__init__()
        self.mapping = mapping_model
        self.beta = beta
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if distance_metric == 'euclidean':
            self.distance = F.pairwise_distance
        elif distance_metric == 'cosine':
            self.distance = lambda x,y: 1 - F.cosine_similarity(x,y)

    def soft_indicator(self,x):
        """
        f(x) = (1+e^(-beta*x))^(-1)
        :param x:
        :param beta: the larger the beta,
        :return:
        """

        # important: the element order of z is not the same with that in x
        # y_pos = torch.exp(-self.beta * x[x >= 0])
        # z_pos = (1 + y_pos) ** (-1)
        # y_neg = torch.exp(self.beta * x[x < 0])
        # z_neg = 1 - (1 + y_neg) ** (-1)

        y = 1 / (1+torch.exp(-self.beta*x))
        return y
        # return torch.cat((z_pos, z_neg))

    def criterion(self,dp,dm):

        objs = self.soft_indicator(dm - dp)
        loss = -torch.mean(objs)
        acc = torch.mean((dp.data < dm.data).float())  # mini-batch acc, batch size is same as dp/dm

        return loss,acc


    def evaluate(self,data_batches):
        """
        evaluate W on data_batches, i.e., all data in data batches
        :param data_batches:
        :return:
        """
        losses, accs = [],[]
        # start = time.time()
        # num_mini_batches = 0

        # print(self.mapping.W)

        for mini_batch in data_batches:
            mini_P_x, mini_P_xp, mini_P_xms = mini_batch

            mini_P_x = mini_P_x.to(self.device)  # can set non_blocking=True
            mini_P_xp = mini_P_xp.to(self.device)
            mini_P_xms = mini_P_xms.to(self.device)

            dp, dm = self.forward(mini_P_x, mini_P_xp, mini_P_xms)
            objs = self.soft_indicator(dm.data - dp.data)
            loss = -torch.sum(objs)
            acc = torch.sum((dp.data < dm.data).float())

            print(acc)

            losses.append(loss)
            accs.append(acc)
            # num_mini_batches += 1
        # print(num_mini_batches)
        # print("evaluate: ", time.time() - start)
        loss = torch.sum(torch.stack(losses))/len(data_batches.dataset)
        acc = torch.sum(torch.stack(accs))/len(data_batches.dataset)
        return acc.item(),loss.item()

class OVAModel(Model):

    def forward(self,mini_P_x, mini_P_xp, mini_P_xms):
        map_min_P_x = self.mapping(mini_P_x)
        map_min_P_xp = self.mapping(mini_P_xp)
        map_min_P_xms = self.mapping(mini_P_xms)
        dp = self.distance(map_min_P_x,map_min_P_xp)
        dm = torch.min(self.distance(map_min_P_x[:,:,None],map_min_P_xms.transpose(1,2)),dim=1)[0]

        return dp,dm

class SCModel(Model):

    def forward(self, mini_P_x, mini_P_xp, mini_P_xm):
        map_min_P_x = self.mapping(mini_P_x)
        map_min_P_xp = self.mapping(mini_P_xp)
        map_min_P_xm = self.mapping(mini_P_xm)
        dp = self.distance(map_min_P_x, map_min_P_xp)
        dm = self.distance(map_min_P_x, map_min_P_xm)

        return dp, dm