import scipy.io
import joblib
import os
import argparse
import matlab.engine

print('[Start doc2vec_tensor ...]')

parser = argparse.ArgumentParser()
parser.add_argument("--file_path", help="saving root path of raw data", default='./test')
parser.add_argument("--win_size", help="window size", type=int, default=5)
parser.add_argument("--cp_rank", help="window size", type=int, default=10)
args = parser.parse_args()

# load data #

[vocabulary, pretrained_embeddings, \
 X, y, X_train, X_test, y_train, y_test, inds_train, inds_test, inds_all] \
    = joblib.load(os.path.join(args.file_path, 'data/raw.pkl'))


# build doc tensor #

vocab_size = len(vocabulary)
doc_size = X.shape[0]
sentence_size = X.shape[1]

class TwoDimDict():
    def __init__(self):
        self.data = {}
    def add(self, i, j, val):
        if i in self.data and j in self.data[i]:
            self.data[i][j] = val
        else:
            if i in self.data:
                self.data[i][j] = val
            else:
                self.data[i] = {}
                self.data[i][j] = val
    def get_item(self):
        for i, i_val in self.data.items():
            for j, j_val in self.data[i].items():
                yield (i, j, j_val)

coord_list = []
val_list = []
for k in range(doc_size):
    word_word_dict = TwoDimDict()

    print('build word_word_doc tensor, {:d}/{:d} ...'.format(k, doc_size))
    inds = list(range(sentence_size))
    for i in range(0, sentence_size-args.win_size+1):
        idx_i = i
        if X[k][idx_i]==0:
            break
        for j in range(args.win_size):
            idx_j = i+j
            if X[k][idx_j] == 0:
                continue
            word_word_dict.add(X[k][idx_i], X[k][idx_j], 1)


    for item in word_word_dict.get_item():
        coord_list.append((item[0], item[1], k))
        val_list.append(item[2])

# save #

if not os.path.exists(args.file_path):
    os.makedirs(args.file_path)

scipy.io.savemat(os.path.join(args.file_path, 'data/tmp_tensor_info.mat'),
                 dict(coord_list=coord_list, val_list=val_list,
                      vocab_size=vocab_size, doc_size=doc_size))


# matlab tensor cp #

eng = matlab.engine.start_matlab()
eng.tensor_cp(args.file_path, args.cp_rank, nargout=0)
eng.quit()

# read doc vec #

doc2vec = scipy.io.loadmat(os.path.join(args.file_path, 'data/tmp_doc2vec_mat.mat'))

# save #

joblib.dump(doc2vec['doc2vec'], os.path.join(args.file_path, 'data/doc2vec_tensor.pkl'))

print('[Finish doc2vec_tensor ...]')
