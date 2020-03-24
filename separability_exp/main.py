from os.path import splitext

import numpy as np
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

from .experiments import SeparableExperiments
from .utils import prepare_separation_data

fembs = ['word2vec-wiki.txt','word2vec-giga.txt','glove-giga.txt','glove-wiki.txt','fasttext-wiki.txt','fasttext-giga.txt']
# fembs = ['word2vec-wiki.txt']
# fembs = ['fasttext-giga.txt']
for femb in fembs:
    X,y = prepare_separation_data(femb)

    # preprocess the data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # avoud data copy
    if not X.flags['C_CONTIGUOUS'] or X.dtype != np.float64:
        print('X may be copied')

    # subspace_magnitude_experiments on number word separability
    experiments = []
    experiments.append(SeparableExperiments(splitext(femb)[0] + '_num_word_split', True, {'X': X, 'y': y}))

    # random choose some token in emb as numbers and refit the model
    # to test whether this kernel can separate arbitrary tokens

    for i in range(5):
        y = shuffle(y)
        exp = SeparableExperiments(splitext(femb)[0]+'_random_split'+'_'+str(i),True,{'X':X,'y':y})
        experiments.append(exp)
    f1s = Parallel(n_jobs=3)(delayed(exp.run)() for exp in experiments)

    print(splitext(femb)[0]+' average random split f1: ',np.mean(f1s[1:]))
    np.save(splitext(femb)[0] + '_random_split_f1', f1s[1:])
    print(splitext(femb)[0]+' num word split f1: ',f1s[0])
    np.save(splitext(femb)[0] + '_num_word_split_f1', f1s[0])