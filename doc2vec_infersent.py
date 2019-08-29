import joblib
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--file_path", help="saving root path of raw data", default='test')
args = parser.parse_args()

[vocabulary, pretrained_embeddings, \
 X, y, X_train, X_test, y_train, y_test, inds_train, inds_test, inds_all] \
    = joblib.load(os.path.join(args.file_path, 'data/raw.pkl'))

sentences = []
for i in range(len(X)):
    print('loading {:d}-th doc ... ...'.format(i))
    doc_idx = X[i]
    words = []
    for idx in doc_idx:
        if idx == 0 or idx==1:
            break
        words.append(vocabulary[idx])
    sentences.append(' '.join(words))

import torch

from models import InferSent

V = 1
MODEL_PATH = 'encoder/infersent%s.pkl' % V
params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                'pool_type': 'max', 'dpout_model': 0.0, 'version': V}
infersent = InferSent(params_model)
infersent.load_state_dict(torch.load(MODEL_PATH))

W2V_PATH = 'dataset/GloVe/glove.840B.300d.txt' \
    if V == 1 else 'dataset/fastText/crawl-300d-2M-subword.vec'
W2V_PATH = 'dataset/fastText/crawl-300d-2M-subword.vec'

infersent.set_w2v_path(W2V_PATH)

infersent.build_vocab(sentences, tokenize=True)

embeddings = infersent.encode(sentences, tokenize=True)

# infersent.visualize('A man plays an instrument.', tokenize=True)

print(embeddings.shape)

joblib.dump(embeddings, os.path.join(args.file_path, 'data/embeddings.pkl'))