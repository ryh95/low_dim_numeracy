import time

import torch
from torch import nn

class Subspace_Model(nn.Module):

    def __init__(self,dim,d,beta):
        super(Subspace_Model,self).__init__()
        self.beta = beta
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.W = nn.Parameter(torch.randn((dim, d),device=self.device))
        # self.W = torch.randn((dim, d), requires_grad=True,device=self.device)
        torch.nn.init.orthogonal_(self.W)
        self.W.data = self.W.T.data  # col orthognol


    def soft_indicator(self,x):
        """
        f(x) = (1+e^(-beta*x))^(-1)
        :param x:
        :param beta: the larger the beta,
        :return:
        """

        # important: the element order of z is not the same with that in x
        y_pos = torch.exp(-self.beta * x[x >= 0])
        z_pos = (1 + y_pos) ** (-1)
        y_neg = torch.exp(self.beta * x[x < 0])
        z_neg = 1 - (1 + y_neg) ** (-1)

        return torch.cat((z_pos, z_neg))

    def criterion(self,dp,dm):

        objs = self.soft_indicator(dm - dp)
        loss = -torch.mean(objs)
        acc = torch.mean((dp <= dm).float())  # mini-batch acc, batch size is same as dp/dm

        return loss,acc

    def project(self):
        # project the variables back to the feasible set

        # find the nearest col orthogonal matrix
        # ref: http://people.csail.mit.edu/bkph/articles/Nearest_Orthonormal_Matrix.pdf
        # ref: https://math.stackexchange.com/q/2500881
        # ref: https://math.stackexchange.com/a/2215371
        u, s, v = torch.svd(self.W.data)
        self.W.data = u @ v.T
        assert not torch.isnan(self.W.data).any(), 'W has nan values'


class OVA_Subspace_Model(Subspace_Model):


    def forward(self,mini_P_x, mini_P_xp, mini_P_xms):
        mini_P_x = mini_P_x.to(self.device)  # can set non_blocking=True
        mini_P_xp = mini_P_xp.to(self.device)
        mini_P_xms = mini_P_xms.to(self.device)
        dp = torch.norm(torch.matmul(mini_P_x - mini_P_xp, self.W), dim=1)
        dm = torch.min(torch.norm(torch.matmul(self.W.T, mini_P_x[:, :, None] - mini_P_xms), dim=1), dim=1)[0]

        return dp,dm

    def evaluate(self,data_batches):
        """
        evaluate W on data_batches, i.e., whole training data
        :param data_batches:
        :return:
        """
        losses, accs = [],[]
        start = time.time()
        # num_mini_batches = 0
        for mini_batch in data_batches:
            mini_P_x, mini_P_xp, mini_P_xms = mini_batch

            dp, dm = self.forward(mini_P_x, mini_P_xp, mini_P_xms)
            loss, acc = self.criterion(dp.data, dm.data)

            losses.append(loss)
            accs.append(acc)
            # num_mini_batches += 1
        # print(num_mini_batches)
        print("evaluate: ", time.time() - start)
        loss = torch.mean(torch.stack(losses))
        acc = torch.mean(torch.stack(accs))
        return acc.item(),loss.item()
