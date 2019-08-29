% OLD ARCHIVE

addpath('../');
addpath('../measures');

clear all;
close all;
clc;

rank_list = [1, 5, 10, 20, 50, 100];
neighbor_list = [1, 5, 10, 20, 50];


acc_mat = zeros(length(rank_list), length(neighbor_list));
kl_mat = zeros(length(rank_list), length(neighbor_list));

root_path = fullfile('../../exp/grid');
maml_result(1:length(rank_list)*length(neighbor_list)) = CVResult;
counter = 1;
for i=1:numel(rank_list),
    cp_rank = rank_list(i);
    for j=1:numel(neighbor_list),
        neighbor = neighbor_list(j);
        for k=0:4,
            file_path = fullfile(root_path, ...
            ['rank', num2str(cp_rank), '_n', num2str(neighbor)], num2str(k));
            
            [acc, dist_name, distance] = ComputeEDLMetrics(fullfile(file_path, ...
            'results/maml.mat'));
            maml_result(counter) = CVResultAppend(maml_result(counter), acc, distance, dist_name);  
        end

        fprintf('*******************************************************\n\n\n');
        fprintf('rank_%d, neighbor_%d\n', cp_rank, neighbor);
        CVResultDisplay(maml_result(counter));

        acc_mat(i, j) = mean(maml_result(counter).acc_list);
        kl_mat(i, j) = mean(maml_result(counter).distance_lists{4});

        counter = counter+1;
    end
end

csvwrite(fullfile('excel', [exp, '_acc_grid.csv']), acc_mat);
csvwrite(fullfile('excel', [exp, '_kl_grid.csv']), kl_mat);



