function [precision, recall, accuracy, F1, dist_name, distance] = ComputeEDLMetrics(file_path)
    eps = 1e-9;
    
    data = load(file_path);
    pred_shape = size(data.pred);
    pred = reshape(data.pred, pred_shape(1), pred_shape(end));
    gt = data.true;
    
    % compute classification
    [precision, recall, accuracy, F1] = ComputeClassificationMetrics(pred, gt);

    % clamp
    pred = max(pred, eps);
    pred = min(pred, 1-eps);
    gt = max(gt, eps);
    gt = min(gt, 1-eps);

    pred = bsxfun(@rdivide, pred, sum(pred,2));
    gt = bsxfun(@rdivide, gt, sum(gt,2));

    % compute edl
    [dist_name, distance] = ComputeEDLMeasures(gt, pred);
end

