import torch
import numpy as np
import scipy.io
import joblib

import util
from model import CNNText
from loss import EDL_Loss
from util import NPArrayList2PyTorchTensorList

import os
import argparse

# config #
inner_rate = 0.01
loss_fn = EDL_Loss()
EPOCHES = 5
ntrain = 10

print('[Start eval_maml ...]')

parser = argparse.ArgumentParser()
parser.add_argument("--file_path", help="saving root path of raw data")
parser.add_argument("--seed", help="reproducible experiment with seeds", type=int)
parser.add_argument("--out_dim", help="output dimension", type=int, default=6)
parser.add_argument('--model_idx', help='index of trained model', default=1000, type=int)
args = parser.parse_args()

RandomGenerator = np.random.RandomState(args.seed)

directory = os.path.join(args.file_path, 'trained_random')
[train_tasks, test_tasks, vocabulary, pretrained_embeddings, X_test, y_test] = \
    joblib.load(os.path.join(args.file_path, 'data/data_random.pkl'))

param_models = joblib.load(os.path.join(directory, 'store-' + str(args.model_idx) + '.pkl'))

y_test_numpy = np.copy(y_test)

model_parameters = NPArrayList2PyTorchTensorList(param_models[0][0])

vocab_size = len(vocabulary)
sentence_len = X_test.shape[1]

history_pred = []
for it, task in enumerate(test_tasks):
    model_eval = CNNText(vocab_size, sentence_len, pretrained_embeddings, args.out_dim).cuda()
    model_eval.set_parameters(model_parameters)
    optimizer = torch.optim.SGD(model_eval.parameters(), lr=inner_rate)
    model_eval.train()

    Xtest, ytest = task.get_center()
    instance_X = torch.from_numpy(Xtest).long().cuda()
    instance_y = torch.from_numpy(ytest).float().cuda()

    Xtrain, ytrain = task.get_all()
    Xtrain = torch.from_numpy(Xtrain).long().cuda()
    ytrain = torch.from_numpy(ytrain).float().cuda()

    for ep in range(EPOCHES):
        m = len(Xtrain)
        inds = RandomGenerator.permutation(m)
        for start in range(0, m, ntrain):
            mbinds = inds[start:start + ntrain]
            preds, _ = model_eval(Xtrain[mbinds])
            preds = preds.cuda()
            loss = loss_fn(ytrain[mbinds], preds)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model_eval.eval()
    pred = util.Predict(instance_X, model_eval)
    history_pred.append(pred)

    print('[{:d}/{:d}] ...'.format(it, len(test_tasks)))

history_pred = np.array(history_pred)

file_name = 'maml_random.mat'

if not os.path.exists(os.path.join(args.file_path, 'results')):
    os.makedirs(os.path.join(args.file_path, 'results'))
scipy.io.savemat(os.path.join(args.file_path, 'results/' + file_name), dict(pred=history_pred, true=y_test_numpy))

print('[Finish eval_maml ...]')
