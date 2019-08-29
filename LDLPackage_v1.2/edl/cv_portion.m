addpath('../');
addpath('../measures');

clear all;
close all;
clc;

cv = [0:9];
portion = 10;
is_doc2vec = true;

root_path = fullfile('../../exp/portion');

linear_result = CVResult;
cnn_result = CVResult;
maml_result = CVResult;
maml_doc2vec_result = CVResult;
maml_joint_result = CVResult;
maml_batch_result = CVResult;
maml_random_result = CVResult;

for k = cv,
    file_path = fullfile(root_path, ['portion', num2str(portion)], num2str(k), 'results');

    [precision, recall, accuracy, F1, dist_name, distance] = ComputeEDLMetrics(fullfile(file_path, 'linear.mat'));
    linear_result = CVResultAppend(linear_result, precision, recall, accuracy, F1, distance, dist_name);  
    
    % cnn
%     [precision, recall, accuracy, F1, dist_name, distance] = ComputeEDLMetrics(fullfile(file_path, 'cnn.mat'));
%     cnn_result = CVResultAppend(cnn_result, precision, recall, accuracy, F1, distance, dist_name);  
    
    % maml
%     [precision, recall, accuracy, F1, dist_name, distance] = ComputeEDLMetrics(fullfile(file_path, ...
%     ['maml.mat']));
%     maml_result = CVResultAppend(maml_result, precision, recall, accuracy, F1, distance, dist_name); 

%     if is_doc2vec,
%         % maml doc2vec
%         [precision, recall, accuracy, F1, dist_name, distance] = ComputeEDLMetrics(fullfile(file_path, ...
%         ['maml_doc2vec.mat']));
%         maml_doc2vec_result = CVResultAppend(maml_doc2vec_result, precision, recall, accuracy, F1, distance, dist_name); 
%     end
% 
%     % maml joint
%     [precision, recall, accuracy, F1, dist_name, distance] = ComputeEDLMetrics(fullfile(file_path, ...
%     ['maml_joint.mat']));
%     maml_joint_result = CVResultAppend(maml_joint_result, precision, recall, accuracy, F1, distance, dist_name); 
% 
%     % maml batch
%     [precision, recall, accuracy, F1, dist_name, distance] = ComputeEDLMetrics(fullfile(file_path, ...
%     ['maml_batch.mat']));
%     maml_batch_result = CVResultAppend(maml_batch_result, precision, recall, accuracy, F1, distance, dist_name); 
    
%     % maml random
%     [precision, recall, accuracy, F1, dist_name, distance] = ComputeEDLMetrics(fullfile(file_path, ...
%     ['maml_random.mat']));
%     maml_random_result = CVResultAppend(maml_random_result, precision, recall, accuracy, F1, distance, dist_name); 
end

fprintf('*******************************************************\n\n\n');
fprintf('cnn\n');
CVResultDisplay(linear_result);


% fprintf('*******************************************************\n\n\n');
% fprintf('cnn\n');
% CVResultDisplay(cnn_result);

% fprintf('*******************************************************\n\n\n');
% fprintf('maml\n');
% CVResultDisplay(maml_result);

% if is_doc2vec,
%     fprintf('*******************************************************\n\n\n');
%     fprintf('maml_doc2vec\n');
%     CVResultDisplay(maml_doc2vec_result);
% end
% 
% % maml joint
% fprintf('*******************************************************\n\n\n');
% fprintf('maml_joint\n');
% CVResultDisplay(maml_joint_result);
% 
% % maml batch
% fprintf('*******************************************************\n\n\n');
% fprintf('maml_batch\n');
% CVResultDisplay(maml_batch_result);

% % maml random
% fprintf('*******************************************************\n\n\n');
% fprintf('maml_random\n');
% CVResultDisplay(maml_random_result);