import time
from os.path import splitext

import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils import shuffle

from numeracy_experiments.local_utils import prepare_separation_data, parallel_predict, SeparableExperiments

fembs = ['skipgram-2.txt','skipgram-5.txt','glove.6B.300d.txt','glove.840B.300d.txt','crawl-300d-2M-subword.vec','wiki-news-300d-1M-subword.vec']
for femb in fembs:
    X,y = prepare_separation_data(femb)

    # preprocess the data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # avoud data copy
    if not X.flags['C_CONTIGUOUS'] or X.dtype != np.float64:
        print('X may be copied')

    # subspace_magnitude_experiments on number word separability
    # todo: max_iter may be different due to different embeddings
    svc = SVC(kernel='poly',C=1,degree=3,gamma=1/300,coef0=0,
              cache_size=3000,class_weight='balanced',verbose=True,max_iter=20000)
    start = time.time()
    svc.fit(X,y)
    print('fit time: ',time.time()-start)
    y_pred = parallel_predict(X,svc.predict,10)
    # np.save('svm_poly_num_word_predict.npy',y_pred)
    print('number word split f1: ',f1_score(y,y_pred))

    # random choose some token in emb as numbers and refit the model
    # to test whether this kernel can seperate arbitray tokens
    experiments = []
    for i in range(10):
        exp = SeparableExperiments(splitext(femb)[0]+'_random_split'+'_'+str(i),True)
        y = shuffle(y)
        exp.exp_data = {'X':X,'y':y}
        experiments.append(exp)
    f1s = np.concatenate(Parallel(n_jobs=8)(delayed(exp.run()) for exp in range(experiments)))
    print(splitext(femb)[0]+' average random split f1: ',np.mean(f1s))
    np.save(splitext(femb)[0]+'_random_split_f1',f1s)