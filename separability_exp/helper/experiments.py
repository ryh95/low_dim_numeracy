import time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from .utils import parallel_predict


class SeparableExperiments(object):

    def __init__(self,exp_name,save_results,exp_data):
        self.name = exp_name
        self.save_results = save_results
        self.exp_data = exp_data

    def run(self):
        # space = [Real(1e-6, 1e+6, prior='log-uniform')]
        # optimize_types = ['C']
        # x0=[1.0]
        # model = SVC(kernel='poly',degree=3,gamma=1/(300*exp_data['X'].var()),coef0=0,
        #   cache_size=8000,class_weight='balanced',verbose=True,max_iter=15000)
        # fitting_X = exp_data['X']
        # fitting_y = exp_data['y']
        # base_workspace = {'model':model,'fitting_X':fitting_X,'fitting_y':fitting_y}
        # minimizer = Minimizer(base_workspace,optimize_types,gp_minimize)
        # res_gp = minimizer.mini_func(space,n_calls=11,verbose=True,x0=x0,n_jobs=-1)
        # if self.save_results:
        #     skopt.dump(res_gp,self.name+'.pkl',store_objective=False)
        # return -res_gp.fun

        # word2vec-wiki iter: 20000/ word2vec-giga iter: 20000
        # glove-wiki iter: 20000/ glove-giga iter: 30000
        # fasttext-wiki iter: 30000/ fasttext-giga iter: 30000

        # cache size: 4000
        iteration = 30000
        name = self.name.split('_')[0]
        if name == 'glove-wiki' or name == 'word2vec-wiki' or name == 'word2vec-giga':
            iteration = 20000

        # clf = SVC(kernel='poly', degree=3, gamma=1 / 300, coef0=0, C=1,
        #           cache_size=4000, class_weight='balanced', verbose=True, max_iter=iteration)
        # clf = LogisticRegression(class_weight='balanced',verbose=True,max_iter=iteration)

        start = time.time()
        f1s = []
        for _ in range(5):
            X_train,X_test,y_train,y_test = train_test_split(self.exp_data['X'], self.exp_data['y'],
                             test_size=0.9, stratify=self.exp_data['y'])
            # change to svc if use svm to classify
            clf = LogisticRegression(class_weight='balanced', verbose=True, max_iter=iteration)
            clf.fit(X_train, y_train)
            y_pred = parallel_predict(X_test, clf.predict, 10)
            f1 = f1_score(y_test, y_pred)
            f1s.append(f1)
        print('fit and predict time: ', time.time() - start)
        print(self.name,np.mean(f1s))
        return np.mean(f1s)