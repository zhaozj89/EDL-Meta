classdef CVResult
    properties
        precision_list = [];
        recall_list = [];
        accuracy_list = [];
        F1_list = [];
        dist_name;
        distance_lists = cell(1, 6);
    end
    
    methods
        function obj = CVResultAppend(obj, precision, recall, accuracy, F1, distance, name)
            obj.precision_list = [obj.precision_list, precision];
            obj.recall_list = [obj.recall_list, recall];
            obj.accuracy_list = [obj.accuracy_list, accuracy];
            obj.F1_list = [obj.F1_list, F1];
            for k=1:6,
               obj.distance_lists{k} = [obj.distance_lists{k}, distance(k)];
            end
            obj.dist_name = name;
        end
        
        function [] = CVResultDisplay(obj),
            for k=1:6,
                fprintf([obj.dist_name{k}, ' avg: %f, std: %f\n'], mean(obj.distance_lists{k}), ...
                 std(obj.distance_lists{k}));
             end
            fprintf('precision avg: %f, std: %f\n', mean(obj.precision_list), std(obj.precision_list));
            fprintf('recall avg: %f, std: %f\n', mean(obj.recall_list), std(obj.recall_list));
            fprintf('F1 avg: %f, std: %f\n', mean(obj.F1_list), std(obj.F1_list));
            fprintf('accuracy avg: %f, std: %f\n', mean(obj.accuracy_list), std(obj.accuracy_list));
        end
    end
end

