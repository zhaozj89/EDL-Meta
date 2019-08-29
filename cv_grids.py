import os

import util

# DO NOT CHANGE (OLD ARCHIVE)
total_number = 600
label_portion = 0.1
out_dim, model_idx, niterations = util.ExpConfigurations()
root_path = 'exp/grid'

for cp_rank in [1, 5, 10, 20, 50, 100]:
    for neighbor in [1, 5, 10, 20, 50]:
        for counter, seed in enumerate([660, 60, 600, 6, 666]):
            print('rank: {:d}, neighbor: {:d}, count: {:d} ...'.format(cp_rank, neighbor, counter))

            file_path = os.path.join(root_path, 'rank'+str(cp_rank)+'_n'+str(neighbor), str(counter))

            if os.path.isfile(os.path.join(file_path, 'results/maml.mat')):
                continue

            # load data
            os.system('python load_data.py --file_path {:s} --label_portion {:f} --total_number {:d} --seed {:d}' \
                      .format(file_path, label_portion, total_number, seed))

            os.system('python doc2vec_tensor.py --file_path {:s} --cp_rank {:d}'.format(file_path, cp_rank))

            # maml
            os.system('python cluster_tensor.py --file_path {:s} --support_number {:d}' \
                      .format(file_path, neighbor))
            os.system('python maml.py --file_path {:s} --out_dim {:d} --niterations {:d}' \
                      .format(file_path, out_dim, niterations))
            os.system('python eval_maml.py --file_path {:s} --out_dim {:d} --model_idx {:d}' \
                      .format(file_path, out_dim, model_idx))