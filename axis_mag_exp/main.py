import pickle
from os.path import join

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from termcolor import colored

from axis_mag_exp.experiments import MagnitudeAxisExperiments
from config import VOCAB_DIR, EMB_DIR

fembs = ['word2vec-wiki']
# test_models = ['ridge','kernel_ridge','kernel_ridge_separation']
test_models = ['pca','proj_pca','ridge']
nums_name = 'nums2'
nums = np.load(join(VOCAB_DIR, nums_name + '.npy'))
n_trials = 5
results = np.zeros((n_trials,len(fembs),len(test_models)))

for i in range(n_trials):

    sel_nums_train,sel_nums_test = train_test_split(nums, test_size=0.2)
    sel_nums_train,sel_nums_val = train_test_split(sel_nums_train,test_size=0.1875)

    for j,femb in enumerate(fembs):

        with open(join(EMB_DIR, nums_name + '_' + femb + '.pkl'), 'rb') as f:
            num_emb = pickle.load(f)

        Xs,ys = [],[]
        for sel_nums_split in [sel_nums_train,sel_nums_val,sel_nums_test]:
            Xs.append(np.stack([num_emb[num] for num in sel_nums_split]))
            ys.append(sel_nums_split.astype(np.float))
        X_train,X_val,X_test = Xs
        y_train,y_val,y_test = ys

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

        print(X_train.shape, X_val.shape, X_test.shape)

        # train,val,test split
        # train:0.65, val:0.15 test:0.2

        base_workspace = {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test,
            'X': np.concatenate([X_train, X_val]),
            'y': np.concatenate([y_train, y_val])
        }

        for k,model in enumerate(test_models):

            exp_name = femb+'_'+model+'_'+str(i)
            exp = MagnitudeAxisExperiments(exp_name, False, {'model':model, 'base_workspace':base_workspace})
            error = exp.run()
            # save results
            results[i][j][k] = error

            print(colored(femb+' '+model+' test root mse: ','red'), error)

    print(results[i,:,:])

np.save('results',results)