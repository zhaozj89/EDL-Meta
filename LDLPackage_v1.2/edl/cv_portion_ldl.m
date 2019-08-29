addpath('../');
addpath('../measures');
addpath('../libsvm-3.23/matlab');

clear all;
close all;
clc;

tic;

method = 'LDSVR'; %'PT-SVM'; %'PT-Bayes'; %CPCNN'; %'BFGS'; %'IIS'; %'AA-BP'; 'LDSVR'; %'AA-KNN';
% portion_list = [5, 10, 30];
portion = 30;


root_path = fullfile('../../exp/portion');
ldl_result = CVResult;
for k = 0:9,
    display(['the ', num2str(k), '-th round ...']);
    
    file_path = fullfile(root_path, ['portion', num2str(portion)], num2str(k), 'results');

    % cnn
    [precision, recall, accuracy, F1, dist_name, distance] = ComputeEDLMetricsByFeatures(fullfile(file_path, 'cnn.mat'), method);
    ldl_result = CVResultAppend(ldl_result, precision, recall, accuracy, F1, distance, dist_name);  
end

fprintf('*******************************************************\n\n\n');
fprintf('%s\n', method);
CVResultDisplay(ldl_result);