from os.path import join
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from ..config import DATA_DIR
from ..dataset import OVADataset, SCDataset


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
    if 'ova' in fdata or 'sc_k' in fdata:
        data = OVADataset(fdata,emb_conf)
    elif 'sc' in fdata:
        data = SCDataset(fdata,emb_conf)
    else:
        assert False
    if pre_load:
        # preload all train_data into memory to save time
        mini_batchs = DataLoader(data, batch_size=128, num_workers=6)
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