from collections import OrderedDict

import numpy as np
import torch
from torch.nn.functional import log_softmax
from tqdm import tqdm


class BasePredictionEvaluator(object):

    def __init__(self, vocab_emb_dict, number_emb_dict, cand_nums, sens, window_size):
        self.vocab_emb_dict = vocab_emb_dict  # nxd, ordered dict
        self.vocab2id = {k: i for i, k in enumerate(self.vocab_emb_dict.keys())}
        self.vocab_emb = torch.tensor(np.stack([v for v in vocab_emb_dict.values()])).float()
        self.number_emb_dict = number_emb_dict  # mxd, ordered dict
        self.id2num = {i: k for i, k in enumerate(self.number_emb_dict.keys())}
        self.num2id = {k: i for i, k in enumerate(self.number_emb_dict.keys())}
        self.number_emb = torch.tensor(np.stack([v for v in number_emb_dict.values()])).float()
        self.cand_nums, self.sens, self.window_size = cand_nums,sens,window_size

        self.log_prob = log_softmax(self.number_emb @ self.vocab_emb.T, dim=1)

    def compute_SA(self, pos, sen, window_size):
        # n: number
        # sen: list of string
        # window_size: int
        # return all numbers SA

        # obtain context words
        window_start = pos - window_size if pos - window_size >= 0 else 0
        window_end = pos + window_size if pos + window_size <= len(sen) else len(sen)
        context = sen[window_start:pos] + sen[pos+1:window_end+1]
        context_id = torch.tensor([self.vocab2id[w] for w in context if w in self.vocab2id])
        if context_id.nelement() == 0: return None
        SA = torch.sum(self.log_prob[:, context_id], dim=1)
        return SA

    def compute_SB(self):
        # todo: finish SB computing
        pass


    def evaluate(self):
        AEs,APEs,Ranks = [],[],[] # absolute errors, absolute percentage errors, ranks of targets
        for sen in tqdm(self.sens):
            inter_nums = set(sen) & set(self.cand_nums)
            for num in inter_nums:
                pos = sen.index(num)
                SA = self.compute_SA(pos,sen,self.window_size)
                if SA is None:
                    print('SA is none, continue')
                    continue
                pred_num = self.id2num[torch.argmax(SA).item()]
                ei = abs(float(num) - float(pred_num))
                if num == '0' or num == '0.0': continue
                pei = ei / float(num)
                APEs.append(pei)
                AEs.append(ei)
                rank = torch.sum(SA > SA[self.num2id[num]]) + 1
                Ranks.append(rank.item())
        self.MdAE,self.MdAPE,self.AVGR = np.median(AEs),np.median(APEs),np.mean(Ranks)
        return self.MdAE,self.MdAPE,self.AVGR

class SubspacePredictionEvaluator(object):

    def __init__(self, vocab_emb_dict, number_emb_dict, cand_nums, sens, window_size):
        self.best_model = None
        self.best_MdAE,self.best_MdAPE,self.best_AVGR = np.inf,np.inf,np.inf
        self.vocab_emb_dict, self.number_emb_dict, self.cand_nums, self.sens, self.window_size = \
            vocab_emb_dict, number_emb_dict, cand_nums, sens, window_size
        self.number_emb = torch.tensor(np.stack([v for v in number_emb_dict.values()])).float()


    def update_best_model(self, model):
        if self.AVGR < self.best_AVGR:
            self.best_MdAE,self.best_MdAPE,self.best_AVGR = self.MdAE,self.MdAPE,self.AVGR
            self.best_model = model

    def evaluate(self, model):
        Q = model.mapping.W.detach()
        # todo: auto set device
        projected_num_emb = self.number_emb.to('cuda') @ Q @ Q.T
        projected_num_emb_dict = OrderedDict(
            (k, projected_num_emb[i, :].cpu().numpy()) for i, k in enumerate(self.number_emb_dict.keys()))
        self.MdAE, self.MdAPE, self.AVGR = BasePredictionEvaluator(self.vocab_emb_dict, projected_num_emb_dict, self.cand_nums, self.sens, self.window_size)\
            .evaluate()
        return self.MdAE, self.MdAPE, self.AVGR