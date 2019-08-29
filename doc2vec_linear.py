from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import joblib
import os
import argparse

print('[Start doc2vec ...]')

parser = argparse.ArgumentParser()
parser.add_argument("--file_path", help="saving root path of raw data")
args = parser.parse_args()

# load data #

[vocabulary, pretrained_embeddings, \
 X, y, X_train, X_test, y_train, y_test, inds_train, inds_test, inds_all] \
    = joblib.load(os.path.join(args.file_path, 'data/raw.pkl'))

# tag data #

tagged_data = []
for i in range(len(X)):
    print('loading {:d}-th doc ... ...'.format(i))
    doc_idx = X[i]
    words = []
    for idx in doc_idx:
        if idx == 0:
            break
        words.append(vocabulary[idx])
    doc = TaggedDocument(words=words, tags=[str(i)])
    tagged_data.append(doc)

# train #

max_epochs = 20
vec_size = 100
alpha = 0.025

model = Doc2Vec(vector_size=vec_size, alpha=alpha, min_alpha=0.00025, min_count=1, dm=1)
model.build_vocab(tagged_data)

for epoch in range(max_epochs):
    print('iteration {0}'.format(epoch))
    model.train(tagged_data, total_examples=model.corpus_count, epochs=model.iter)
    # decrease the learning rate
    model.alpha -= 0.0002
    # fix the learning rate, no decay
    model.min_alpha = model.alpha

if not os.path.exists(args.file_path):
    os.makedirs(args.file_path)

model.save(os.path.join(args.file_path, 'data/doc2vec_linear.model'))

print('[Finish doc2vec ...]')
