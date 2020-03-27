import io
import pickle
from collections import defaultdict
from os.path import join, splitext

import numpy as np

import torch
from gensim.models import KeyedVectors
import scipy.io as sio
from tqdm import tqdm
import torch.nn.functional as F

from config import EMB_DIR


def vocab2vec(vocab, output_dir, output_name, fword_emb, savefmt, type='glove', normalize=False, oov_handle='random'):
    """

    :param vocab: list of words,each word should be a string
    :param output_dir: location to put vocab_vec numpy array
    :param fword_emb: word embedding file name
    :return: vocab_vec, a numpy array, order is the same with vocab, d*len(vocab)
    """
    print('preparing vocab vectors')
    vocab2pos = defaultdict(list)
    for i, word in enumerate(vocab):
        vocab2pos[word].append(i)
    # vocab2pos = {word:i for i,word in enumerate(vocab)}

    if type == 'glove':
        with open(join(EMB_DIR,fword_emb+'.txt'), 'r') as f:
            # todo: remove this if the embedding doesn't have header
            f.readline() # skip header
            _, *vec = next(f).rstrip().split(' ')
            vec_dim = len(vec)
    elif type == 'word2vec':
        word_dictionary = KeyedVectors.load_word2vec_format(join(EMB_DIR,fword_emb+'.bin') , binary=True)
        vec_dim = word_dictionary.vector_size
    else:
        assert 'type error'

    len_vocab = len(vocab)
    vocab_vec_ind = np.ones(len_vocab,dtype=bool)
    vocab_vec = np.zeros((vec_dim,len_vocab))

    if type == 'glove':
        mean_emb_vec = np.zeros(vec_dim)
        with open(join(EMB_DIR,fword_emb+'.txt'), 'r') as f:
            f.readline() # skip header
            i = 0
            for line in tqdm(f):
                word, *vec = line.rstrip().split(' ')
                vec = np.array(vec, dtype=float)
                mean_emb_vec += vec

                if word in vocab2pos:
                    if normalize:
                        vec = vec / np.linalg.norm(vec)
                    n_repeat = len(vocab2pos[word])
                    vocab_vec[:, np.array(vocab2pos[word])] = np.repeat(vec[:, np.newaxis], n_repeat, 1)
                    vocab_vec_ind[np.array(vocab2pos[word])] = False

                i += 1
        mean_emb_vec = mean_emb_vec / i
    elif type == 'word2vec':
        for id,word in enumerate(vocab):

            if word in word_dictionary:
                vec = word_dictionary[word]
                if normalize:
                    vec = vec / np.linalg.norm(vec)
                vocab_vec[:,id] = vec
                vocab_vec_ind[id] = False

        mean_emb_vec = np.mean(word_dictionary.vectors,axis=0)
    else:
        assert False,'please specify emb type'

    if normalize:
        mean_emb_vec = mean_emb_vec / np.linalg.norm(mean_emb_vec)
    # handling OOV words in vocab2vec
    # TODO: find better ways to handle OOV
    n_oov = sum(vocab_vec_ind)
    if oov_handle == 'random':
        print('%d words in vocab, %d words not found in word embedding file'% (len_vocab, n_oov))
        print('they are',np.array(vocab)[vocab_vec_ind])
        print('init them with random numbers')
        vecs = np.random.randn(vec_dim,n_oov)
        if normalize:
            vecs = vecs/np.linalg.norm(vecs,axis=0)
        vocab_vec[:,vocab_vec_ind] = vecs
    elif oov_handle == 'mean_emb_vec':
        print('%d words in vocab, %d words not found in word embedding file, init them with the mean vector' % (
            len_vocab, n_oov))
        vocab_vec[:, vocab_vec_ind] = np.repeat(mean_emb_vec[:,np.newaxis],n_oov,1)
    print('saving vocab vector file')

    word2vec = {word: vocab_vec[:, id] for id, word in enumerate(vocab)}
    for fmt in savefmt:
        if fmt == 'mat':
            sio.savemat(join(output_dir,output_name+'.mat'),{output_name:vocab_vec})
        elif fmt == 'npy':
            np.save(join(output_dir,output_name+'.npy'),vocab_vec)
        elif fmt == 'pickle':
            with open(join(output_dir,output_name+'.pkl'), 'wb') as handle:
                pickle.dump(word2vec, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return word2vec,vocab_vec

def is_number(s):
    try:
        return True if not np.isnan(float(s)) and not np.isinf(float(s)) else False
    except ValueError:
        return False

def is_valid_triple(triple):
    x,xp,xm = triple
    return abs(x-xp) < abs(x-xm)

def is_valid_relax_triple(triple):
    x, xp, xm = triple
    return abs(x - xp) <= abs(x - xm)

def preprocess_google_news_skip(emb_fname):
    """
    extract numbers from bin and save into txt
    :return:
    """
    word_dictionary = KeyedVectors.load_word2vec_format(emb_fname+'.bin', binary=True)
    with open(emb_fname+'_num'+'.txt','w') as f:
        for word in word_dictionary.vocab:
            if is_number(word):
                str_vec = ' '.join(str(n) for n in word_dictionary[word])
                f.write(word+' '+str_vec+'\n')

def extract_num_emb(emb_fname):
    """
    extract num embedding from skipgram/fasttext/glove embedding and save
    :param emb_fname:
    :return:
    """
    fin = io.open(emb_fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    if 'skipgram' in emb_fname or 'subword' in emb_fname:
        fin.readline()
    with open(splitext(emb_fname)[0] + '_num' + '.txt', 'w') as f:
        for line in tqdm(fin):
            word,*vec = line.rstrip().split(' ')
            if 'skipgram' in emb_fname:
                word = word.split('_')[0]
            if is_number(word):
                str_vec = ' '.join(vec)
                f.write(word + ' ' + str_vec + '\n')
    fin.close()


class KernelDistance(object):

    def __int__(self,*kernel_setting):

        self.kernel_setting = kernel_setting

    def kernel_sim(self,x,y,kernel_type,gamma,r,d):
        if len(x.size()) == 2 and len(y.size()) == 2:
            # x: B,d y: B,d
            inner_product = torch.sum(x * y,dim=1)
        elif len(x.size()) == 3 and len(y.size()) == 3:
            # x: B,d,n-2 y: B,d,n-2
            inner_product = torch.sum(x * y,dim=1)
        elif len(x.size()) == 2 and len(y.size()) == 3:
            # x: B,d y: B,d,n-2
            inner_product = (x[:,None,:] @ y).squeeze()
        else:
            assert False
        if kernel_type == 'poly':
            return (gamma * inner_product + r) ** d
        elif kernel_type == 'sigmoid':
            return torch.tanh(gamma * inner_product + r)

    def distance_func(self,x,y):
        _,kernel_type,d,gamma,r = self.kernel_setting
        k_xy = self.kernel_sim(x, y, kernel_type, gamma, r, d)
        k_xx = self.kernel_sim(x, x, kernel_type, gamma, r, d)
        if len(y.size()) == 3:
            k_xx = k_xx[:,None]
        k_yy = self.kernel_sim(y, y, kernel_type, gamma, r, d)
        return 1 - k_xy / ((k_xx ** 0.5) * (k_yy ** 0.5))

def cosine_distance(x,y):
    return 1 - F.cosine_similarity(x, y)