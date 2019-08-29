addpath('../');
addpath('../measures');

clear all;
close all;
clc;

is_doc2vec = true;

root_path = '../../test';

cnn_result = CVResult;
maml_result = CVResult;
maml_doc2vec_result = CVResult;
maml_joint_result = CVResult;
maml_batch_result = CVResult;
maml_random_result = CVResult;

file_path = fullfile(root_path, 'results');

% cnn
[precision, recall, accuracy, F1, dist_name, distance] = ComputeEDLMetrics(fullfile(file_path, 'cnn.mat'));
cnn_result = CVResultAppend(cnn_result, precision, recall, accuracy, F1, distance, dist_name);   

fprintf('*******************************************************\n\n\n');
fprintf('cnn\n');
CVResultDisplay(cnn_result);

% maml
[precision, recall, accuracy, F1, dist_name, distance] = ComputeEDLMetrics(fullfile(file_path, ...
['maml.mat']));
maml_result = CVResultAppend(maml_result, precision, recall, accuracy, F1, distance, dist_name); 

fprintf('*******************************************************\n\n\n');
fprintf('maml\n');
CVResultDisplay(maml_result);
% 
% if is_doc2vec==true,
%     % maml doc2vec
%     [precision, recall, accuracy, F1, dist_name, distance] = ComputeEDLMetrics(fullfile(file_path, ...
%     ['maml_doc2vec.mat']));
%     maml_doc2vec_result = CVResultAppend(maml_doc2vec_result, precision, recall, accuracy, F1, distance, dist_name); 
% 
%     fprintf('*******************************************************\n\n\n');
%     fprintf('maml_doc2vec\n');
%     CVResultDisplay(maml_doc2vec_result);
% end
% 
% % maml joint
% [precision, recall, accuracy, F1, dist_name, distance] = ComputeEDLMetrics(fullfile(file_path, ...
% ['maml_joint.mat']));
% maml_joint_result = CVResultAppend(maml_joint_result, precision, recall, accuracy, F1, distance, dist_name); 
% 
% fprintf('*******************************************************\n\n\n');
% fprintf('maml_joint\n');
% CVResultDisplay(maml_joint_result);
% 
% % maml batch
% [precision, recall, accuracy, F1, dist_name, distance] = ComputeEDLMetrics(fullfile(file_path, ...
% ['maml_batch.mat']));
% maml_batch_result = CVResultAppend(maml_batch_result, precision, recall, accuracy, F1, distance, dist_name); 
% 
% fprintf('*******************************************************\n\n\n');
% fprintf('maml_batch\n');
% CVResultDisplay(maml_batch_result);
% 
% % maml random
% [precision, recall, accuracy, F1, dist_name, distance] = ComputeEDLMetrics(fullfile(file_path, ...
% ['maml_random.mat']));
% maml_random_result = CVResultAppend(maml_random_result, precision, recall, accuracy, F1, distance, dist_name); 
% 
% fprintf('*******************************************************\n\n\n');
% fprintf('maml_random\n');
% CVResultDisplay(maml_random_result);

