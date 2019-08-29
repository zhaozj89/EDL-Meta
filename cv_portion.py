import os
import multiprocessing

import util

cp_rank = 5
neighbor = 20

total_number = -1
out_dim, model_idx, niterations = util.ExpConfigurations()
root_path = 'exp/portion'

def worker(counter, seed):
    for label_portion in [0.1]:
        file_path = os.path.join(root_path, 'portion' + str(int(label_portion * 100)), str(counter))

        # load data
        os.system('python load_data.py --file_path {:s} --label_portion {:f} --total_number {:d} --seed {:d}' \
                  .format(file_path, label_portion, total_number, seed))


        # # cnn
        # os.system('python cnn_joint.py --file_path {:s} --out_dim {:d} --seed {:d}' \
        #           .format(file_path, out_dim, seed))

        # # doc2vec
        # # os.system('python doc2vec_linear.py --file_path {:s}'.format(file_path))
        os.system('python doc2vec_tensor.py --file_path {:s} --cp_rank {:d}'.format(file_path, cp_rank))
        # os.system('python doc2vec_infersent.py --file_path {:s}'.format(file_path))

        # os.system('python infersent_joint.py --file_path {:s} --out_dim {:d} --seed {:d}' \
        #           .format(file_path, out_dim, seed))

        # maml
        os.system('python cluster_tensor.py --file_path {:s} --support_number {:d}' \
                  .format(file_path, neighbor))
        os.system('python maml.py --file_path {:s} --out_dim {:d} --niterations {:d}  --seed {:d}' \
                  .format(file_path, out_dim, niterations, seed))
        os.system('python eval_maml.py --file_path {:s} --out_dim {:d} --model_idx {:d}  --seed {:d}' \
                  .format(file_path, out_dim, model_idx, seed))

        # # maml joint
        # os.system('python eval_maml_joint.py --file_path {:s} --out_dim {:d}' \
        #           .format(file_path, out_dim))
        #
        # # maml batch
        # os.system('python eval_maml_batch.py --file_path {:s} --out_dim {:d}' \
        #           .format(file_path, out_dim))

# jobs = []
# for counter, seed in enumerate([345, 543, 789, 987, 567]): #, 765, 123, 321, 456, 654]):
#     print(counter)
#     p = multiprocessing.Process(target=worker, args=(counter, seed, ))
#     jobs.append(p)
#     p.start()


for counter, seed in enumerate([345, 543, 789, 987, 567, 765, 123, 321, 456, 654]):
    worker(counter, seed)

