import os
import multiprocessing

import util

total_number = -1
out_dim = 6
root_path = 'exp/portion'

def worker(counter, seed):
    for label_portion in [0.1]:
        file_path = os.path.join(root_path, 'portion' + str(int(label_portion * 100)), str(counter))

        # load data
        os.system('python load_data.py --file_path {:s} --label_portion {:f} --total_number {:d} --seed {:d}' \
                  .format(file_path, label_portion, total_number, seed))

        # linear
        os.system('python bert.py --file_path {:s} --out_dim {:d} --seed {:d}' \
                  .format(file_path, out_dim, seed))

for counter, seed in enumerate([345, 543, 789, 987, 567, 765, 123, 321, 456, 654]):
    print(counter)
    worker(counter, seed)
