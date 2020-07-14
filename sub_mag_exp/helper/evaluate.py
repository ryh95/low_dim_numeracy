import torch
from torch.utils.data import DataLoader


class SubspaceMagEvaluator(object):

    def __init__(self, data=None):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.eval_data = data

    def evaluate(self, model):
        """
        evaluate W on data_batches, i.e., all data in data batches
        :param data: dataset of torch.utils.data
        :return:
        """
        losses, accs = [],[]
        # start = time.time()
        # num_mini_batches = 0

        # print(self.mapping.W)
        data_batches = DataLoader(self.eval_data, batch_size=256, shuffle=True, num_workers=0,
                   pin_memory=True)
        for mini_batch in data_batches:
            mini_P_x, mini_P_xp, mini_P_xms = mini_batch

            mini_P_x = mini_P_x.to(self.device)  # can set non_blocking=True
            mini_P_xp = mini_P_xp.to(self.device)
            mini_P_xms = mini_P_xms.to(self.device)

            dp, dm = model(mini_P_x, mini_P_xp, mini_P_xms)
            objs = model.loss(dp.data - dm.data)
            loss = torch.sum(objs)
            acc = torch.sum((dp.data < dm.data).float())

            print(acc)

            losses.append(loss)
            accs.append(acc)
            # num_mini_batches += 1
        # print(num_mini_batches)
        # print("evaluate: ", time.time() - start)
        loss = torch.sum(torch.stack(losses))/len(data_batches.dataset)
        acc = torch.sum(torch.stack(accs))/len(data_batches.dataset)
        return acc.item()