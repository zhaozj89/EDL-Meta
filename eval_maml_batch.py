import torch
import numpy as np
import scipy.io
import joblib

from model import CNNText
from util import NPArrayList2PyTorchTensorList

import os
import argparse

# config #
ntrain = 50

print('[Start eval_maml ...]')

parser = argparse.ArgumentParser()
parser.add_argument("--file_path", help="saving root path of raw data")
parser.add_argument("--seed", help="reproducible experiment with seeds", type=int)
parser.add_argument("--out_dim", help="output dimension", type=int, default=6)
parser.add_argument('--model_idx', help='index of trained model', default=1000, type=int)
args = parser.parse_args()

RandomGenerator = np.random.RandomState(args.seed)

directory = os.path.join(args.file_path, 'trained')
[train_tasks, test_tasks, vocabulary, pretrained_embeddings, X_test, y_test] = \
    joblib.load(os.path.join(args.file_path, 'data/data.pkl'))

param_models = joblib.load(os.path.join(directory, 'store-' + str(args.model_idx) + '.pkl'))

y_test_numpy = np.copy(y_test)

model_parameters = NPArrayList2PyTorchTensorList(param_models[0][0])

vocab_size = len(vocabulary)
sentence_len = X_test.shape[1]

Xtest = torch.from_numpy(X_test).long().cuda()
ytest = torch.from_numpy(y_test).float().cuda()
n = len(Xtest)

model_eval = CNNText(vocab_size, sentence_len, pretrained_embeddings, args.out_dim).cuda()
model_eval.set_parameters(model_parameters)
model_eval.eval()

history_pred = None
inds = np.array(range(n))
for start in range(0, n, ntrain):
    mbinds = inds[start:start + ntrain]
    preds, _ = model_eval(Xtest[mbinds])

    history_pred = np.concatenate([history_pred, preds.clone().cpu().detach().numpy()]) \
        if history_pred is not None else preds.clone().cpu().detach().numpy()

history_pred = np.array(history_pred)

file_name = 'maml_batch.mat'

if not os.path.exists(os.path.join(args.file_path, 'results')):
    os.makedirs(os.path.join(args.file_path, 'results'))
scipy.io.savemat(os.path.join(args.file_path, 'results/' + file_name), dict(pred=history_pred, true=y_test_numpy))

print('[Finish eval_maml ...]')
