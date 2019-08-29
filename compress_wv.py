import os
import sys
import time
import joblib
import pickle
import gensim

def customize_embeddings_from_pretrained_googlenews_w2v(pretrained_embedding_fpath):
    saved_path = '../data/RenCECps/loaded_data.pkl'
    if os.path.isfile(saved_path):
        [X, y, vocabulary] = joblib.load(saved_path)

    vocabulary_inv = {rank: word for rank, word in enumerate(vocabulary)}
    embedding_dim = 300

    directory = "./wv/exp2"
    if not os.path.exists(directory):
        os.makedirs(directory)
    fpath_pretrained_extracted = os.path.expanduser(
        "{}/extracted-python{}.pl".format(directory, sys.version_info.major))
    fpath_word_list = os.path.expanduser("{}/words.dat".format(directory))

    tic = time.time()
    model = gensim.models.KeyedVectors.load_word2vec_format(pretrained_embedding_fpath, binary=False)
    print('Please wait ... (it could take a while to load the file : {})'.format(pretrained_embedding_fpath))
    print('Done.  (time used: {:.1f}s)\n'.format(time.time() - tic))

    embedding_weights = {}

    found_cnt = 0
    not_found_cnt = 0
    words = []
    for id, word in vocabulary_inv.items():
        words.append(word)
        if word in model.vocab:
            embedding_weights[id] = model.word_vec(word)
            found_cnt += 1
        else:
            embedding_weights[id] = [0] * embedding_dim  # RandomGenerator.uniform(-0.05, 0.05, embedding_dim)
            not_found_cnt += 1
    with open(fpath_pretrained_extracted, 'wb') as f:
        pickle.dump(embedding_weights, f)
    f.close()
    with open(fpath_word_list, 'w') as f:
        f.write("\n".join(words))
    f.close()

    print('found {:d}'.format(found_cnt))
    print('not found {:d}'.format(not_found_cnt))


def main():
    path_to_googlenews_vectors = "../data/GoogleNews-vectors-negative300.bin"
    if not os.path.exists(path_to_googlenews_vectors):
        print('Sorry, file "{}" does not exist'.format(path_to_googlenews_vectors))
        sys.exit()

    print('Your path to the googlenews vector file is: ', path_to_googlenews_vectors)
    customize_embeddings_from_pretrained_googlenews_w2v(path_to_googlenews_vectors)


if __name__ == "__main__":
    main()
