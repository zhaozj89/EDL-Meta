function tensor_cp(root_path,cp_rank)
  addpath('./tensor_toolbox');

  file_path = fullfile(root_path, 'data/tmp_tensor_info.mat');

  tensor_info = load(file_path);

  tensor = sptensor(double(tensor_info.coord_list)+1, double(tensor_info.val_list'), [double(tensor_info.vocab_size), double(tensor_info.vocab_size), ...
  double(tensor_info.doc_size)]);

  % cp is better
  %factors = tucker_als(tensor, double(cp_rank));
  factors = cp_als(tensor, double(cp_rank));

  doc2vec = factors.U{3};

  save(fullfile(root_path, 'data/tmp_doc2vec_mat.mat'), 'doc2vec');