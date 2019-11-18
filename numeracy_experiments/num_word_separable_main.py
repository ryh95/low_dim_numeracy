import time

import numpy as np
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils import shuffle

from numeracy_experiments.local_utils import prepare_separation_data, parallel_predict

femb = 'skipgram-5.txt'
X,y = prepare_separation_data(femb)

# if emb_type == 'pre-train':
#
#
# elif emb_type == 'random':


# preprocess the data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# avoud data copy
if not X.flags['C_CONTIGUOUS'] or X.dtype != np.float64:
    print('X may be copied')

# subspace_magnitude_experiments on number word separability
params = {
    'kernel': 'poly',
    'C':1.0,
    'degree': 3,
    'gamma': 1/(300*X.var()),
    'coef0': 0.0
}
svc = SVC(cache_size=8000,class_weight='balanced',verbose=True,max_iter=20000)
svc.set_params(**params)
start = time.time()
svc.fit(X,y)
print('fit time: ',time.time()-start)
y_pred = parallel_predict(X,svc.predict,10)
np.save('svm_poly_num_word_predict.npy',y_pred)
print('number word f1: ',f1_score(y,y_pred))

# random choose some token in emb as numbers and refit the model
# to test whether this kernel can seperate arbitray tokens
f1s = []
for _ in range(10):
    svc = SVC(cache_size=8000, class_weight='balanced', verbose=True, max_iter=20000)
    svc.set_params(**params)
    start = time.time()
    X = shuffle(X)
    svc.fit(X,y)
    print('fit time: ',time.time()-start)
    y_pred = parallel_predict(X, svc.predict, 10)
    f1 = f1_score(y,y_pred)
    print('random number word f1: ',f1)
    f1s.append(f1)
print('average random number word f1: ',np.mean(f1s))