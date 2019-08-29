import torch
from torch import optim
import joblib
import os
import argparse
import numpy as np

from model import CNNText
from util import PyTorchParameterList2NPArrayList
from loss import EDL_Loss

# config #
inner_rate = 0.01
outer_rate = 0.1
n_subsets = 5
ntrain = 10
loss_fn = EDL_Loss()

print('[Start maml ...]')

parser = argparse.ArgumentParser()
parser.add_argument("--file_path", help="saving root path of raw data", default='./test')
parser.add_argument("--seed", help="reproducible experiment with seeds", type=int)
parser.add_argument('--is_doc2vec', help='use doc2vec clustering', default='0')
parser.add_argument("--out_dim", help="output dimension", type=int, default=6)
parser.add_argument('--niterations', help='number of iterations', default=1000, type=int)
args = parser.parse_args()

RandomGenerator = np.random.RandomState(args.seed)

if args.is_doc2vec is '1':
    data_file_path = os.path.join(args.file_path, 'data/data_doc2vec.pkl')
elif args.is_doc2vec is '0':
    data_file_path = os.path.join(args.file_path, 'data/data.pkl')
else:
    raise ValueError
[train_tasks, test_tasks, vocabulary, pretrained_embeddings, X_test, y_test] = joblib.load(data_file_path)

vocab_size = len(vocabulary)
sentence_len = X_test.shape[1]

model = CNNText(vocab_size, sentence_len, pretrained_embeddings, args.out_dim).cuda()
meta_optim = optim.SGD(model.parameters(), lr=outer_rate)

num_train_tasks = len(train_tasks)
for iteration in range(args.niterations + 1):
    total_loss = None
    for i in range(n_subsets):
        model.train()

        k = RandomGenerator.randint(0, high=num_train_tasks)
        task = train_tasks[k]

        Xtrain, ytrain = task.get_train()
        Xtest, ytest = task.get_test()
        Xtrain = torch.from_numpy(Xtrain).long().cuda()
        ytrain = torch.from_numpy(ytrain).float().cuda()
        Xtest = torch.from_numpy(Xtest).long().cuda()
        ytest = torch.from_numpy(ytest).float().cuda()

        m = len(Xtrain)
        fast_weights = None

        inds = RandomGenerator.permutation(m)
        mbinds = inds[0:ntrain]

        ypred, _ = model(Xtrain[mbinds], vars=fast_weights)
        ypred = ypred.cuda()

        # choose = RandomGenerator.randint(0, high=5)
        # if choose == 0:
        #     loss_fn = KL_Loss()
        # elif choose == 1:
        #     loss_fn = CE_Loss()
        # elif choose == 2:
        #     loss_fn = EDL_Loss()
        # elif choose == 3:
        #     loss_fn = Euclidean_Loss()
        # elif choose == 4:
        #     loss_fn = Cosine_Loss()

        loss = loss_fn(ytrain[mbinds], ypred)

        grad = torch.autograd.grad(loss, model.parameters())
        fast_weights = list(map(lambda p: p[1] - inner_rate * p[0], zip(grad, model.parameters())))

        n = len(Xtest)
        inds = RandomGenerator.permutation(n)

        model.eval()
        ypred, _ = model(Xtest[inds], vars=fast_weights)
        ypred = ypred.cuda()
        y = ytest[inds]

        if total_loss is None:
            total_loss = loss_fn(y, ypred)
        else:
            tmp_loss = loss_fn(y, ypred)
            total_loss += tmp_loss

    total_loss /= n_subsets

    model.train()

    meta_optim.zero_grad()
    total_loss.backward()
    meta_optim.step()

    print('[{:d}/{:d}] ... ...'.format(iteration, args.niterations))

    if iteration % 1000 == 0:
        param_models = []
        param = PyTorchParameterList2NPArrayList(model.parameters())
        param_models.append(param)

        if args.is_doc2vec is '1':
            directory = os.path.join(args.file_path, 'trained_doc2vec')
        elif args.is_doc2vec is '0':
            directory = os.path.join(args.file_path, 'trained')
        else:
            raise ValueError

        if not os.path.exists(directory):
            os.makedirs(directory)

        joblib.dump([param_models], os.path.join(directory, 'store-' + str(iteration) + '.pkl'))

print('[Finish maml ...]')
