function [precision, recall, accuracy, F1] = ComputeClassificationMetrics(pred, gt),
    known_label = [];
    pred_label = [];
    m = size(gt);
    num_sample = m(1);
    for i=1:m(1),
        [val1, idx1] = max(pred(i,:));
        [val2, idx2] = max(gt(i,:));
        known_label = [known_label idx2];
        pred_label = [pred_label idx1];
    end

    C = confusionmat(known_label,pred_label);
    precision_list = [];
    recall_list = [];
    m = size(C);
    for i=1:m(2),
       if sum(C(:,i))==0,
           precision_list = [precision_list 0];
       else,
           precision_list = [precision_list C(i,i)/sum(C(:,i))];
       end
    end
    
    precision = sum(precision_list)/numel(precision_list);
    
    for i=1:m(1),
        if sum(C(i,:))==0,
            recall_list = [recall_list 0];
        else,
            recall_list = [recall_list C(i,i)/sum(C(i,:))];
        end
    end
    
    recall = sum(recall_list)/numel(recall_list);
    
    accuracy_list = [];
    for i=1:m(1),
        accuracy_list = [accuracy_list C(i,i)];
    end
    
    accuracy = sum(accuracy_list)/num_sample;
    
    F1 = 2*(precision*recall)/(precision+recall);
end

