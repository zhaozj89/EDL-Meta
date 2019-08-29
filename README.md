## A meta-learning approach to text emotion distribution learning from small sample

#### Load data

* `load_data.py`

#### Embedding

* `bert.py`: BERT embedding

* `doc2vec_linear.py`: doc2vec embedding

* `doc2vec_tensor.py`: tensor embedding

* `doc2vec_infersent.py`: InferSent embedding

#### Task partition

* `cluster_doc2vec.py`: doc2vec

* `cluster_random.py`: random

* `cluster_tensor.py`: tensor embedding

* `cluster_infersent.py`: InferSent

#### Methods

* `cnn_joint.py`: CNN

* `infersent_joint.py`: InferSent or doc2vec

* `bert.py`: BERT

* `maml.py`: meta-training, `eval_maml.py`: meta-adaption

#### Experiment templates

* `cv_portion.py`: cross validation

* `example.sh`: example shell

## Note ##

* `LDLPackage-v1.2` and `tensor_toolbox` have their own licenses

* the results of label distribution learning methods, e.g., PT-X, AA-X, SA-X, can be calculated with `LDLPackage-v1.2/edl/cv_portion_ldl.m`

* grid search can be conducted by `cv_grids.py` and `LDLPackage-v1.2/edl/cv_grid.m`

* a quick demo can be experimented with `bash example.sh` and `LDLPackage-v1.2/edl/example.m`

* if you want to use a different word embedding, first use `compress_wv.py` to get a compressed word embedding file and put it in the corresponding path

* some codes should be tuned a bit (set the right paths) to work, feel free to contact me if you have any question

* running the BERT method needs [bert-as-service](https://github.com/hanxiao/bert-as-service)

* running the InferSent method needs [InferSent](https://github.com/facebookresearch/InferSent)

---
GNU GENERAL PUBLIC LICENSE