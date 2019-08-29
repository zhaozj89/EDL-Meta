#!/usr/bin/env bash

total_number=-1
seed=456

file_path='test'
label_portion=0.1

out_dim=6

# load data
python load_data_exp1.py --file_path $file_path --label_portion $label_portion --total_number $total_number --seed $seed

# linear
python bert.py --file_path $file_path --out_dim $out_dim --seed $seed