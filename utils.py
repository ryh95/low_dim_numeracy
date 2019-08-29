import pickle
from collections import defaultdict
from os.path import join

import numpy as np

import torch
from gensim.models import KeyedVectors
import scipy.io as sio
from tqdm import tqdm


def soft_indicator(x,beta):
    """
    f(x) = (1+e^(-beta*x))^(-1)
    :param x:
    :param beta: the larger the beta,
    :return:
    """
    return (1 + torch.exp(-beta*x)) ** (-1)

def vocab2vec(vocab, output_dir, output_name, word_emb, savefmt, type='glove', normalize=False, oov_handle='random'):
    """

    :param vocab: list of words(str)
    :param output_dir: location to put vocab_vec numpy array
    :param word_emb: word embedding file name
    :return: vocab_vec, a numpy array, order is the same with vocab, d*len(vocab)
    """
    print('preparing vocab vectors')
    vocab2pos = defaultdict(list)
    for i, word in enumerate(vocab):
        vocab2pos[word].append(i)
    # vocab2pos = {word:i for i,word in enumerate(vocab)}

    if type == 'glove':
        with open(word_emb, 'r') as f:
            _, *vec = next(f).rstrip().split(' ')
            vec_dim = len(vec)
    elif type == 'word2vec':
        word_dictionary = KeyedVectors.load_word2vec_format(word_emb, binary=True)
        vec_dim = word_dictionary.vector_size
    else:
        assert 'type error'

    len_vocab = len(vocab)
    vocab_vec_ind = np.ones(len_vocab,dtype=bool)
    vocab_vec = np.zeros((vec_dim,len_vocab))
    mean_emb_vec = np.zeros(vec_dim)

    if type == 'glove':
        with open(word_emb, 'r') as f:
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
            # TODO: test !!
            if not is_number(word):

                if word in word_dictionary:
                    vocab_vec[:,id] = word_dictionary[word]
                    vocab_vec_ind[id] = False

            else:

                if float(word).is_integer():
                    if str(int(float(word))) in word_dictionary:
                        vocab_vec[:, id] = word_dictionary[word]
                        vocab_vec_ind[id] = False
                else:
                    if word in word_dictionary:
                        vocab_vec[:, id] = word_dictionary[word]
                        vocab_vec_ind[id] = False
        mean_emb_vec = np.mean(word_dictionary.vectors,axis=0)

    if normalize:
        mean_emb_vec = mean_emb_vec / np.linalg.norm(mean_emb_vec)
    # handling OOV words in vocab2vec
    # TODO: find better ways to handle OOV
    n_oov = sum(vocab_vec_ind)
    if oov_handle == 'random':
        print('%d words in vocab, %d words not found in word embedding file, init them with random numbers' % (
        len_vocab, n_oov))
        vocab_vec[:,vocab_vec_ind] = np.random.rand(vec_dim,n_oov)
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
            with open(join(output_dir,output_name+'.pickle'), 'wb') as handle:
                pickle.dump(word2vec, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return word2vec,vocab_vec

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def is_valid_triple(triple):
    x,xp,xm = triple
    return abs(x-xp) < abs(x-xm)