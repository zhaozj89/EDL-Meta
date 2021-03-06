import torch
import numpy as np
import scipy.io
import joblib

from model import CNNText
from loss import EDL_Loss

import os
import argparse

# config #
inner_rate = 0.1
ntrain = 50
loss_fn = EDL_Loss()

print('[Start cnn joint ...]')

parser = argparse.ArgumentParser()
parser.add_argument("--file_path", help="saving root path of raw data")
parser.add_argument("--seed", help="reproducible experiment with seeds", type=int)
parser.add_argument("--out_dim", help="output dimension", type=int, default=6)
args = parser.parse_args()

RandomGenerator = np.random.RandomState(args.seed)

[vocabulary, pretrained_embeddings, \
 X, y, X_train, X_test, y_train, y_test, inds_train, inds_test, inds_all] \
    = joblib.load(os.path.join(args.file_path, 'data/raw.pkl'))

Xtrain = torch.from_numpy(X_train).long().cuda()
ytrain = torch.from_numpy(y_train).float().cuda()
Xtest = torch.from_numpy(X_test).long().cuda()
ytest = torch.from_numpy(y_test).float().cuda()

vocab_size = len(vocabulary)
sentence_len = X.shape[1]

joint_model = CNNText(vocab_size, sentence_len, pretrained_embeddings, args.out_dim).cuda()
optimizer = torch.optim.SGD(joint_model.parameters(), lr=inner_rate)
for epoch in range(25):
    print('joint training {:d}'.format(epoch))
    joint_model.train()
    m = len(Xtrain)
    inds = RandomGenerator.permutation(m)
    for start in range(0, m, ntrain):
        mbinds = inds[start:start + ntrain]
        preds, _ = joint_model(Xtrain[mbinds])
        preds = preds.cuda()
        loss = loss_fn(ytrain[mbinds], preds)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

train_extracted_features = None
train_labels = None
test_extracted_features = None
test_preds = None

model_eval = CNNText(vocab_size, sentence_len, pretrained_embeddings, args.out_dim).cuda()
model_eval.set_parameters(joint_model.parameters())
model_eval.eval()

m = len(Xtrain)
inds = np.array(range(m))
for start in range(0, m, ntrain):
    mbinds = inds[start:start + ntrain]
    preds, features = model_eval(Xtrain[mbinds])
    preds = preds.cuda()

    train_extracted_features = \
        np.concatenate([train_extracted_features, features.cpu().detach().numpy()]) \
            if train_extracted_features is not None else features.cpu().detach().numpy()

    train_labels = \
        np.concatenate([train_labels, ytrain[mbinds].clone().cpu().detach().numpy()]) \
            if train_labels is not None else ytrain[mbinds].clone().cpu().detach().numpy()

n = len(Xtest)
inds = np.array(range(n))
for start in range(0, n, ntrain):
    mbinds = inds[start:start + ntrain]
    preds, features = model_eval(Xtest[mbinds])

    test_extracted_features = \
        np.concatenate([test_extracted_features, features.cpu().detach().numpy()]) \
            if test_extracted_features is not None else features.cpu().detach().numpy()

    test_preds = \
        np.concatenate([test_preds, preds.clone().cpu().detach().numpy()]) \
            if test_preds is not None else preds.clone().cpu().detach().numpy()

test_preds = np.array(test_preds)
train_extracted_features = np.array(train_extracted_features)
test_extracted_features = np.array(test_extracted_features)
train_labels = np.array(train_labels)

if not os.path.exists(os.path.join(args.file_path, 'results')):
    os.makedirs(os.path.join(args.file_path, 'results'))
scipy.io.savemat(os.path.join(args.file_path, 'results/cnn.mat'),
                 dict(pred=test_preds, true=y_test,
                      train_features=train_extracted_features,
                      train_labels=train_labels,
                      test_features=test_extracted_features))

print('[Finish cnn joint ...]')
