from os.path import splitext, join

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from config import VOCAB_DIR
from archived_code.local_utils import prepare_separation_data, SepMagExp, load_num_emb
import numpy as np

fembs = ['word2vec-wiki.txt','word2vec-giga.txt','glove-wiki.txt','glove-giga.txt','fasttext-wiki.txt','fasttext-giga.txt']
sel_nums = np.load(join(VOCAB_DIR, 'inter_nums.npy'))
sel_nums_train,sel_nums_test = train_test_split(sel_nums,test_size=0.2)

for femb in fembs:
    X,y = prepare_separation_data(femb)

    # preprocess the data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # avoud data copy
    if not X.flags['C_CONTIGUOUS'] or X.dtype != np.float64:
        print('X may be copied')

    sel_X, sel_mag = load_num_emb(splitext(femb)[0], sel_nums_test)
    exp = SepMagExp(splitext(femb)[0] + '_sep_mag', True, {'X':X,'y':y,'sel_X':sel_X, 'sel_mag':sel_mag})
    error = exp.run()
    print(error)