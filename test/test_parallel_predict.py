import time
from math import ceil

# from sklearn.externals.joblib import Parallel, delayed
from joblib import Parallel, delayed
import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from archived_code.local_utils import prepare_separation_data

femb = 'skipgram-5.txt'
X,y = prepare_separation_data(femb)

scaler = StandardScaler()
X = scaler.fit_transform(X)

n_cores = 10
n_samples = X.shape[0]
slices = [(ceil(n_samples*i/n_cores), ceil(n_samples*(i+1)/n_cores)) for i in range(n_cores)]
print(slices)

params = {
    'kernel': 'poly',
    'C':1.0,
    'degree': 3,
    'gamma': 1/(300*X.var()),
    'coef0': 0.0
}
svc = SVC(cache_size=8000,class_weight='balanced',verbose=True)
svc.set_params(**params)
start = time.time()
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.99,stratify=y)
svc.fit(X_train,y_train)
print('fit time: ',time.time()-start)

start = time.time()
y_pred = np.concatenate(Parallel(n_jobs=n_cores)(delayed(svc.predict)(X[slices[i_core][0]:slices[i_core][1],:]) for i_core in range(n_cores)))
print(y_pred)
print('predict time: ',time.time()-start)
print('number word f1: ',f1_score(y,y_pred))