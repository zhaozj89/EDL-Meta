import numpy as np
import joblib
from sklearn.neighbors import NearestNeighbors

from task import TestTask, Task
import os
import argparse

# config #
test_proportion = 0.5

print('[Start cluster_tensor ...]')

parser = argparse.ArgumentParser()
parser.add_argument("--file_path", help="saving root path of raw data", default='test')
parser.add_argument('--support_number', help='support number for testing', type=int, default=50)
args = parser.parse_args()

[vocabulary, pretrained_embeddings, \
 X, y, X_train, X_test, y_train, y_test, inds_train, inds_test, inds_all] \
    = joblib.load(os.path.join(args.file_path, 'data/raw.pkl'))

doc2vec_tensor = joblib.load(os.path.join(args.file_path, 'data/doc2vec_tensor.pkl'))

train_doc_tensor = doc2vec_tensor[inds_train]

nbrs = NearestNeighbors(n_neighbors=args.support_number + 1, algorithm='kd_tree', metric='euclidean').fit(
    train_doc_tensor)


def find_KNN_inds(test_idx):
    test_vec = doc2vec_tensor[test_idx]
    _, indices = nbrs.kneighbors(test_vec[None, :], n_neighbors=args.support_number+1)
    return indices[0][1:]


def construct_task(inds):
    tasks = []
    for counter, i in enumerate(inds):
        # print('[{:d}/{:d}] ... ...'.format(counter, len(inds)))
        if args.support_number == 0:
            tasks.append(Task(None, None, X[i][None, :], y[i][None, :], test_size=test_proportion))
            continue

        X_task = None
        y_task = None
        neighbor_inds = find_KNN_inds(i)
        neighbor_inds = neighbor_inds[:args.support_number]
        for idx in neighbor_inds:
            neighbor_idx = inds_train[idx]
            X_task = np.concatenate([X_task, X[neighbor_idx][None, :]], axis=0) \
                if X_task is not None else X[neighbor_idx][None, :]
            y_task = np.concatenate([y_task, y[neighbor_idx][None, :]], axis=0) \
                if y_task is not None else y[neighbor_idx][None, :]
        tasks.append(Task(X_task, y_task, X[i][None, :], y[i][None, :], test_size=test_proportion))
    return tasks


def construct_test_task(inds):
    tasks = []
    for counter, i in enumerate(inds):
        # print('[{:d}/{:d}] ... ...'.format(counter, len(inds)))
        if args.support_number == 0:
            tasks.append(TestTask(None, None, X[i][None, :], y[i][None, :]))
            continue

        X_task = None
        y_task = None
        neighbor_inds = find_KNN_inds(i)
        neighbor_inds = neighbor_inds[:args.support_number]
        for idx in neighbor_inds:
            neighbor_idx = inds_train[idx]
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
            os.path.join(args.file_path, 'data/data.pkl'))

print('total number of train tasks:     {:d}'.format(len(train_tasks)))
print('total number of test tasks:      {:d}'.format(len(test_tasks)))
print('total number of train samples:   {:d}'.format(len(y_train)))
print('total number of test samples:    {:d}'.format(len(y_test)))

print('[Finish cluster ...]')
