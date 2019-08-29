import numpy as np
from gensim.models.doc2vec import Doc2Vec
import joblib

from task import TestTask, Task
import os
import argparse

# config #
test_proportion = 0.5

print('[Start cluster ...]')

parser = argparse.ArgumentParser()
parser.add_argument("--file_path", help="saving root path of raw data")
parser.add_argument('--support_number', help='support number for testing', type=int, default=10)
args = parser.parse_args()

doc_model = Doc2Vec.load(os.path.join(args.file_path, 'data/doc2vec_linear.model'))

[vocabulary, pretrained_embeddings, \
 X, y, X_train, X_test, y_train, y_test, inds_train, inds_test, inds_all] \
    = joblib.load(os.path.join(args.file_path, 'data/raw.pkl'))


def construct_task(inds, Number):
    tasks = []
    for counter, i in enumerate(inds):
        print('[{:d}/{:d}] ... ...'.format(counter, len(inds)))
        if Number == 0:
            tasks.append(Task(None, None, X[i][None, :], y[i][None, :], test_size=test_proportion))
            continue

        X_task = None
        y_task = None
        similar_docs = doc_model.docvecs.most_similar(str(i), topn=Number, clip_start=0, clip_end=len(inds_train))
        neighbor_inds = []
        for doc in similar_docs:
            X_task = np.concatenate([X_task, X[int(doc[0])][None, :]], axis=0) if X_task is not None else X[int(
                doc[0])][None, :]
            y_task = np.concatenate([y_task, y[int(doc[0])][None, :]], axis=0) if y_task is not None else y[int(
                doc[0])][None, :]
            neighbor_inds.append(int(doc[0]))

        tasks.append(Task(X_task, y_task, X[i][None, :], y[i][None, :], test_size=test_proportion))
    return tasks


def construct_test_task(inds, Number):
    tasks = []
    for counter, i in enumerate(inds):
        print('[{:d}/{:d}] ... ...'.format(counter, len(inds)))
        if Number == 0:
            tasks.append(TestTask(None, None, X[i][None, :], y[i][None, :]))
            continue

        X_task = None
        y_task = None
        similar_docs = doc_model.docvecs.most_similar(str(i), topn=Number, clip_start=0, clip_end=len(inds_train))
        neighbor_inds = []
        for doc in similar_docs:
            X_task = np.concatenate([X_task, X[int(doc[0])][None, :]], axis=0) if X_task is not None else X[int(
                doc[0])][None, :]
            y_task = np.concatenate([y_task, y[int(doc[0])][None, :]], axis=0) if y_task is not None else y[int(
                doc[0])][None, :]
            neighbor_inds.append(int(doc[0]))
        tasks.append(TestTask(X_task, y_task, X[i][None, :], y[i][None, :]))
    return tasks


if not os.path.exists(os.path.join(args.file_path, 'data')):
    os.makedirs(args.file_path)

train_tasks = construct_task(inds_train, args.support_number)
test_tasks = construct_test_task(inds_test, args.support_number)
joblib.dump([train_tasks, test_tasks, vocabulary, pretrained_embeddings, X_test, y_test], \
            os.path.join(args.file_path, 'data/data_doc2vec.pkl'))

print('total number of train tasks:     {:d}'.format(len(train_tasks)))
print('total number of test tasks:      {:d}'.format(len(test_tasks)))
print('total number of train samples:   {:d}'.format(len(y_train)))
print('total number of test samples:    {:d}'.format(len(y_test)))

print('[Finish cluster ...]')
