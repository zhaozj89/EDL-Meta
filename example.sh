#!/usr/bin/env bash

total_number=-1
seed=456

file_path='test'
label_portion=0.9

out_dim=6
support_number=20
cp_rank=5

#out_dim=8
#support_number=5
#cp_rank=5

niterations=1000

# load data
python load_data.py --file_path $file_path --label_portion $label_portion --total_number $total_number --seed $seed

# cnn
python cnn_joint.py --file_path $file_path --out_dim $out_dim --seed $seed

## doc2vec
#python doc2vec_linear.py --file_path $file_path
python doc2vec_tensor.py --file_path $file_path --cp_rank $cp_rank

# maml
is_doc2vec='0'

python cluster_tensor.py --file_path $file_path --support_number $support_number
python maml.py --file_path $file_path --out_dim $out_dim --niterations $niterations --is_doc2vec $is_doc2vec --seed $seed
python eval_maml.py --file_path $file_path --out_dim $out_dim --model_idx $niterations --is_doc2vec $is_doc2vec --seed $seed

#is_doc2vec='1'
#
#python cluster_doc2vec.py --file_path $file_path --support_number $support_number
#python maml.py --file_path $file_path --out_dim $out_dim --niterations $niterations --is_doc2vec $is_doc2vec --seed $seed
#python eval_maml.py --file_path $file_path --out_dim $out_dim --model_idx $niterations --is_doc2vec $is_doc2vec --seed $seed
#
## random
#python cluster_random.py --file_path $file_path --support_number $support_number --seed $seed
#python maml_random.py --file_path $file_path --out_dim $out_dim --niterations $niterations --seed $seed
#python eval_maml_random.py --file_path $file_path --out_dim $out_dim --model_idx $niterations --seed $seed
#
#python eval_maml_joint.py --file_path $file_path --out_dim $out_dim --seed $seed
#python eval_maml_batch.py --file_path $file_path --out_dim $out_dim --model_idx $niterations --seed $seed