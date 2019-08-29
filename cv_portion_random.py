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

        # maml
        os.system('python cluster_random.py --file_path {:s} --support_number {:d} --seed {:d}' \
                  .format(file_path, neighbor, seed))
        os.system('python maml_random.py --file_path {:s} --out_dim {:d} --niterations {:d}  --seed {:d}' \
                  .format(file_path, out_dim, niterations, seed))
        os.system('python eval_maml_random.py --file_path {:s} --out_dim {:d} --model_idx {:d}  --seed {:d}' \
                  .format(file_path, out_dim, model_idx, seed))

jobs = []

for counter, seed in enumerate([345, 543, 789, 987, 567, 765, 123, 321, 456, 654]):
    print(counter)
    p = multiprocessing.Process(target=worker, args=(counter, seed, ))
    jobs.append(p)
    p.start()