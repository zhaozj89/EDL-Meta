from xml.dom import minidom
import nltk
import collections
import numpy as np
import joblib
import os
import argparse

import util


def _preprocessing(X):
    max_sentence_len = 0
    word_freqs = collections.Counter()
    for key, val in X.items():
        words = nltk.word_tokenize(val.lower())

        if len(words) > max_sentence_len:
            max_sentence_len = len(words)

        for word in words:
            word_freqs[word] += 1
    return word_freqs, max_sentence_len


def _words2indices_for_a_sentence(X, word2index):
    Xout = {}

    for key, val in X.items():
        words = nltk.word_tokenize(val.lower())
        seqs = []

        for word in words:
            if word in word2index:
                seqs.append(word2index[word])
            else:
                seqs.append(word2index["<UNK>"])

        Xout[key] = seqs

    return Xout


def _SemEval2007_Path2String(X_path, y_path):
    xmldoc = minidom.parse(X_path)
    itemlist = xmldoc.getElementsByTagName('instance')

    file = open(y_path)
    lines = file.read().split('\n')
    lines = lines[0:-1]

    assert len(itemlist) == len(lines), 'data size not uniform'

    X = {}
    y = {}

    for item in itemlist:
        id = int(item.attributes['id'].value)
        X[id] = item.firstChild.nodeValue

    for line in lines:
        items = line.split(' ')
        id = int(items[0])
        y[id] = []
        for i in range(1, len(items)):
            y[id].append(float(items[i]))

    return X, y


def Load_SemEval2007Data():
    DATA_X_PATH1 = "./AffectiveText.Semeval.2007/AffectiveText.test/affectivetext_test.xml"
    DATA_Y_PATH1 = "./AffectiveText.Semeval.2007/AffectiveText.test/affectivetext_test.emotions.gold"

    DATA_X_PATH2 = "./AffectiveText.Semeval.2007/AffectiveText.trial/affectivetext_trial.xml"
    DATA_Y_PATH2 = "./AffectiveText.Semeval.2007/AffectiveText.trial/affectivetext_trial.emotions.gold"

    X1, y1 = _SemEval2007_Path2String(DATA_X_PATH1, DATA_Y_PATH1)
    X2, y2 = _SemEval2007_Path2String(DATA_X_PATH2, DATA_Y_PATH2)
    X = X1.copy()
    y = y1.copy()
    X.update(X2)
    y.update(y2)

    word_frequency, max_sentence_len = _preprocessing(X)

    max_vocab = len(word_frequency)
    word2index = {x[0]: i + 2 for i, x in enumerate(word_frequency.most_common(max_vocab))}
    word2index["<PAD>"] = 0
    word2index["<UNK>"] = 1
    index2word = {v: k for k, v in word2index.items()}

    X = _words2indices_for_a_sentence(X, word2index)

    Xlist = []
    ylist = []
    for key, val in X.items():
        Xlist.append(X[key])

        y_sum = float(sum(y[key]))
        if y_sum == 0:
            ylist.append([1 / 6.0] * 6)
        else:
            ylist.append([val / y_sum for val in y[key]])

    for i, x in enumerate(Xlist):
        if len(x) < max_sentence_len:
            Xlist[i] = list(x) + [0] * (max_sentence_len - len(x))

    X = np.array(Xlist)
    y = np.array(ylist)

    vocabulary = [index2word[i] for i in range(max_vocab + 2)]
    return X, y, vocabulary


if __name__ == '__main__':
    print('[Start load_data ...]')

    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", help="saving root path of raw data")
    parser.add_argument('--label_portion', help='proportion of labels', type=float)
    parser.add_argument('--total_number', help='total number', type=int, default=1250)
    parser.add_argument("--seed", help="reproducible experiment with seeds", type=int)
    args = parser.parse_args()

    saved_path = './AffectiveText.Semeval.2007/data.pkl'
    if os.path.isfile(saved_path):
        [X, y, vocabulary] = joblib.load(saved_path)
    else:
        X, y, vocabulary = Load_SemEval2007Data()
        joblib.dump([X, y, vocabulary], saved_path)

    N = len(X)
    random_generator = np.random.RandomState(args.seed)
    inds_all = random_generator.permutation(N)

    if args.total_number <= 0:
        X = X[inds_all]
        y = y[inds_all]
    else:
        X = X[inds_all[:args.total_number]]
        y = y[inds_all[:args.total_number]]

    N = len(X)
    inds_all = np.array(range(N))
    test_num = int((1 - args.label_portion) * N)
    inds_train = inds_all[:N - test_num]
    inds_test = inds_all[N - test_num:]

    X_train = X[inds_train]
    y_train = y[inds_train]
    X_test = X[inds_test]
    y_test = y[inds_test]

    pretrained_embeddings = util.Load_pretrained_embeddings('./wv/extracted-python{}.pl')

    if not os.path.exists(os.path.join(args.file_path, 'data')):
        os.makedirs(os.path.join(args.file_path, 'data'))
    joblib.dump([vocabulary, pretrained_embeddings, \
                 X, y, X_train, X_test, y_train, y_test, inds_train, inds_test, inds_all], \
                os.path.join(args.file_path, 'data/raw.pkl'))

    print('[Finish load_data ...]')
