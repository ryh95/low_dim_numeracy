import os
import pickle
from os.path import join, isfile

import skopt
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from skopt import gp_minimize
from skopt.callbacks import CheckpointSaver
from skopt.space import Integer, Real, Categorical
from skopt.utils import dump
from torch.utils.data import ConcatDataset, DataLoader
import torch.nn.functional as F
import numpy as np
from config import DATA_DIR, EMB_DIR
from model import OVAModel, SCModel, SubspaceMapping
from subspace_magnitude_experiments.local_utils import load_dataset, Minimizer, init_evaluate, train_dev_test_split
from utils import cosine_distance, vocab2vec

experiments = 'ova'

if experiments == 'ova':
    model = OVAModel
elif experiments == 'sc':
    model = SCModel
else:
    assert False
num_sources = ['inter_nums']
# num_sources = ['ori']
fsrc_datas = {}
# fembs = ['glove.6B.300d_num']
fembs = ['word2vec-wiki']
for src in num_sources:
    fdatas = [src + '_' + experiments + '_' + type for type in ['train', 'dev', 'test']]
    # if no training data, train dev test split
    if not isfile(join(DATA_DIR, fdatas[0] + '.npy')):
        print('prepare train dev and test')
        X_root = np.load(join(DATA_DIR, src + '_' + experiments + '.npy'), allow_pickle=True)
        X_train,X_test = train_test_split(X_root,test_size=0.2)
        X_train,X_val = train_test_split(X_train,test_size=0.1875)
        for name,data in zip(['train','dev','test'],[X_train,X_val,X_test]):
            np.save(join(DATA_DIR, src + '_' + experiments+'_'+name),data)

    for femb in fembs:
        if femb == 'random': continue
        fdata_embs = [fdata+'_'+femb for fdata in fdatas]

        # if no embedding of training data, prepare and standardize
        if experiments == 'sc':

            if not isfile(join(EMB_DIR, fdata_embs[0] + '.pickle')):
                # prepare train/dev/test embedding
                embs = []
                nums = []
                X_root = np.load(join(DATA_DIR, src + '_' + experiments + '.npy'), allow_pickle=True)
                X_root_num = list(set(np.array(X_root).flat))
                X_root_num_demb, _ = vocab2vec(X_root_num, output_dir=EMB_DIR, output_name=fdata_embs[0],
                                               word_emb=join(EMB_DIR, femb + '.txt'), savefmt=['None'])
                for fdata in fdatas:
                    X = np.load(join(DATA_DIR, fdata + '.npy'), allow_pickle=True)
                    X_num = list(set(np.array(X).flat))
                    nums.append(X_num)
                    X_num_emb = np.zeros((len(X_num),300))
                    for i,nu in enumerate(X_num):
                        X_num_emb[i,:] = X_root_num_demb[nu]
                    embs.append(X_num_emb)

                # for fdata,fdata_emb in zip(fdatas,fdata_embs):
                #     X = np.load(join(DATA_DIR,fdata + '.npy'),allow_pickle=True)
                #     X_num = list(set(np.array(X).flat))
                #     nums.append(X_num)
                #     _,vocab_vec = vocab2vec(X_num, output_dir=EMB_DIR, output_name=fdata_emb,
                #                             word_emb=join(EMB_DIR, femb + '.txt'), savefmt=['None'])
                #     embs.append(vocab_vec)
                emb_train,emb_dev,emb_test = embs
                # standardize the embedding
                scaler = StandardScaler()
                emb_train = scaler.fit_transform(emb_train)
                emb_dev = scaler.transform(emb_dev)
                emb_test = scaler.transform(emb_test)
                embs = [emb_train,emb_dev,emb_test]
                for i in range(3):
                    with open(join(EMB_DIR, fdata_embs[i] + '.pickle'), 'wb') as handle:
                        pickle.dump({num:embs[i][j,:] for j,num in enumerate(nums[i])}, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            if not isfile(join(EMB_DIR,src + '_' + experiments+'_'+femb+'.pickle')):

                X_root = np.load(join(DATA_DIR, src + '_' + experiments + '.npy'), allow_pickle=True)
                X_root_num = list(set(np.array(X_root).flat))
                X_root_num_demb, _ = vocab2vec(X_root_num, output_dir=EMB_DIR, output_name=src + '_' + experiments+'_'+femb,
                                               word_emb=join(EMB_DIR, femb + '.txt'), savefmt=['pickle'])

            # todo: standardize the number embedding?


    fsrc_datas[src] = fdatas

base_workspace = {
    'train_verbose':True,
    'n_epochs':30,
    'mini_batch_size':256,
    'emb_dim':300,
    'model':model,
    'mapping_type':'nn',
    'save_model': True,
    'select_inter_model':False,
    'eval_data': ['val'],
    'working_status': 'optimize', # 'optimize'/ 'infer' / 'eval'
    'optimizer':torch.optim.Adam,
    'distance_metric':'cosine'
}
mini_func = gp_minimize
# optimize_types = ['subspace_dim','beta','lr']
optimize_types = ['n_hidden1','n_out','beta','lr']
minimizer = Minimizer(base_workspace, optimize_types, mini_func)

# embs = ['skipgram-2_num','skipgram-5_num','wiki-news-300d-1M-subword_num','crawl-300d-2M-subword_num', 'glove.840B.300d','glove.6B.300d']
# embs = ['random-1','random-2','random-3','random-4','random-5']
# embs = ['random']

# dataset = load_dataset(subspace_magnitude_experiments,{'emb_fname':'wiki-news-300d-1M-subword_num'})
# cosine_distance = lambda x,y: 1 - F.cosine_similarity(x,y)
# print(init_evaluate(dataset,cosine_distance))

# for fname in embs:
#     dataset = load_dataset('scmag_str',{'emb_fname':fname})
#
#     # evaluate the embedding at the original space
#     print(init_evaluate(dataset,cosine_distance))

# params = torch.load('128_300_1_cosine_0.0255_50.pt')
# model = model(128,300,1,base_workspace['distance_metric'])
# model.W = torch.nn.Parameter(params['W'])
# train_data = load_dataset('scmag_str', {'emb_fname':'random-1'},pre_load=False)
#
# num_emb = torch.stack([v for _,v in train_data.number_emb_dict.items()])
# s_num_emb = num_emb @ model.W.cpu()
#
# mini_batchs = DataLoader(train_data, batch_size=base_workspace['mini_batch_size'], shuffle=True, num_workers=0, pin_memory=True)
# acc,_ = model.evaluate(mini_batchs)
# print(acc)
# exit()

# for fname in embs:
#     emb_conf = {}
#     if 'random' in fname:
#         emb_conf['dim'] = 300
#     emb_conf['emb_fname'] = fname
#     train_data = load_dataset('scmag_str', emb_conf,pre_load=False)
#     print(len(train_data))
#     # val_data = load_dataset(fdev, emb_conf)
#     minimizer.base_workspace['train_data'] = train_data
#     # base_workspace['val_data'] = val_data
#     results_fname = '_'.join(['results', fname])
#     res = skopt.load(results_fname+'.pkl')
#     print(res.x)
#     eval_accs = minimizer.objective(res.x)
#     print('train acc: %f'%(eval_accs['train']))
#     # print('val acc: %f'%(eval_accs['val']))

for src in num_sources:
    for femb in fembs:
        emb_conf = {}
        if 'random' in femb:
            emb_conf['dim'] = 300
        emb_conf['emb_fname'] = femb
        test_dataset = load_dataset(fsrc_datas[src][-1], emb_conf, pre_load=False)
        print('test acc in original space of %s: %.4f'%(femb,init_evaluate(test_dataset,cosine_distance)))

for src in num_sources:
    for femb in fembs:
        emb_conf = {}
        if 'random' in femb:
            emb_conf['dim'] = 300
        emb_conf['emb_fname'] = femb
        datas = []
        for fdata in fsrc_datas[src]:
            datas.append(load_dataset(fdata, emb_conf, pre_load=False))

        minimizer.base_workspace['train_data'] = datas[0]
        minimizer.base_workspace['val_data'] = datas[1]

        # order should be the same as the "optimize_types"
        space = [Categorical([64,128]),
                 Categorical([32,64]),
                 Integer(1, 12),
                 Real(10 ** -5, 10 ** 0, "log-uniform"),
                 ]

        checkpoint_fname = '_'.join([src, experiments, femb, 'checkpoint.pkl'])
        checkpoint_callback = CheckpointSaver(checkpoint_fname, store_objective=False)

        # load checkpoint if there exists checkpoint, otherwise from scratch
        if os.path.isfile(checkpoint_fname):
            print('load checkpoint and continue optimization')
            int_res = skopt.load(checkpoint_fname)
            x0,y0 = int_res.x_iters,int_res.func_vals
            res = minimizer.minimize(space, x0=x0, y0=y0, n_calls=50-len(x0), callback=[checkpoint_callback], verbose=True)
        else:
            x0 = [128,64,6,0.001]
            res = minimizer.minimize(space, x0=x0, n_calls=50, callback=[checkpoint_callback], verbose=True)

        results_fname = '_'.join(['results', src, experiments, femb])
        dump(res, results_fname+'.pkl',store_objective=False)

        # train on the train and dev sets and then test on test sets
        minimizer.base_workspace['train_data'] = ConcatDataset([datas[0], datas[1]])
        del minimizer.base_workspace['val_data']
        minimizer.base_workspace['test_data'] = datas[2]

        minimizer.base_workspace['working_status'] = 'infer'
        minimizer.base_workspace['eval_data'] = ['test']

        test_acc = minimizer.objective(res.x)
        print('test acc: %f'% (test_acc))