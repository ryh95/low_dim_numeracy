import time

from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score

from separability_exp.helper.kernel_kmeans import KernelKMeans
from separability_exp.helper.utils import prepare_separation_data

# fembs = ['word2vec-wiki.txt','word2vec-giga.txt','glove-giga.txt','glove-wiki.txt','fasttext-wiki.txt','fasttext-giga.txt']
fembs = ['word2vec-wiki.txt']
# fembs = ['fasttext-giga.txt']
for femb in fembs:
    X,y = prepare_separation_data(femb,sample_ratio=0.1)
    # start = time.time()
    # kmeans = KMeans(n_clusters=2, n_init=10, init='random', n_jobs=-1).fit(X)
    # print(time.time()-start)
    # nmi = normalized_mutual_info_score(y,kmeans.labels_,average_method='arithmetic')
    start = time.time()
    knkmeans = KernelKMeans(n_clusters=2, max_iter=100, tol=1e-3, random_state=None,
                 kernel="poly", gamma=1 / 300, degree=3, coef0=0,
                 kernel_params=None, verbose=1)
    y_pred = knkmeans.fit_predict(X)
    print(time.time() - start)
    nmi = normalized_mutual_info_score(y, y_pred, average_method='arithmetic')
    print(nmi)