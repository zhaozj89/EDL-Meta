import numpy as np
import joblib

from task import TestTask, Task
import os
import argparse

# config #
test_proportion = 0.5

print('[Start cluster_tensor ...]')

parser = argparse.ArgumentParser()
parser.add_argument("--file_path", help="saving root path of raw data", default='test')
parser.add_argument('--support_number', help='support number for testing', type=int, default=50)
parser.add_argument("--seed", help="reproducible experiment with seeds", type=int)
args = parser.parse_args()

RandomGenerator = np.random.RandomState(args.seed)

[vocabulary, pretrained_embeddings, \
 X, y, X_train, X_test, y_train, y_test, inds_train, inds_test, inds_all] \
    = joblib.load(os.path.join(args.file_path, 'data/raw.pkl'))


def construct_task(inds):
    tasks = []
    for counter, i in enumerate(inds):
        X_task = None
        y_task = None

        neighbor_inds = RandomGenerator.permutation(len(inds_train))
        for idx in range(args.support_number):
            neighbor_idx = inds_train[neighbor_inds[idx]]
            X_task = np.concatenate([X_task, X[neighbor_idx][None, :]], axis=0) \
                if X_task is not None else X[neighbor_idx][None, :]
            y_task = np.concatenate([y_task, y[neighbor_idx][None, :]], axis=0) \
                if y_task is not None else y[neighbor_idx][None, :]

        tasks.append(Task(X_task, y_task, X[i][None, :], y[i][None, :]))
    return tasks


def construct_test_task(inds):
    tasks = []
    for counter, i in enumerate(inds):
        X_task = None
        y_task = None

        neighbor_inds = RandomGenerator.permutation(len(inds_train))
        for idx in range(args.support_number):
            neighbor_idx = inds_train[neighbor_inds[idx]]
            X_task = np.concatenate([X_task, X[neighbor_idx][None, :]], axis=0) \
                if X_task is not None else X[neighbor_idx][None, :]
            y_task = np.concatenate([y_task, y[neighbor_idx][None, :]], axis=0) \
                if y_task is not None else y[neighbor_idx][None, :]

        tasks.append(TestTask(X_task, y_task, X[i][None, :], y[i][None, :]))
    return tasks


if not os.path.exists(os.path.join(args.file_path, 'data')):
    os.makedirs(args.file_path)

train_tasks = construct_task(inds_train)
test_tasks = construct_test_task(inds_test)
joblib.dump([train_tasks, test_tasks, vocabulary, pretrained_embeddings, X_test, y_test], \
            os.path.join(args.file_path, 'data/data_random.pkl'))

print('total number of train tasks:     {:d}'.format(len(train_tasks)))
print('total number of test tasks:      {:d}'.format(len(test_tasks)))
print('total number of train samples:   {:d}'.format(len(y_train)))
print('total number of test samples:    {:d}'.format(len(y_test)))

print('[Finish cluster ...]')
