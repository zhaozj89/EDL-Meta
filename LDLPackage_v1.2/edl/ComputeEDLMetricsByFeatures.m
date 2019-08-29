function [precision, recall, accuracy, F1, dist_name, distance] = ComputeEDLMetricsByFeatures(file_path,method)
    eps = 1e-9;

    data = load(file_path);
    train_features = double(data.train_features);
    train_labels = double(data.train_labels);

    test_features = double(data.test_features);
    gt = double(data.true);

    if strcmp(method, 'PT-Bayes'),
        model = ptbayesTrain(train_features, train_labels);
        pred = ptbayesPredict(model, test_features);
    end

    if strcmp(method, 'PT-SVM'),
        model = ptsvmTrain(train_features,train_labels);
        pred = ptsvmPredict(model, test_features);
    end

    if strcmp(method, 'AA-KNN'),
        pred = aaknn(train_features, train_labels, test_features, 4, 'L2');
    end

    if strcmp(method, 'AA-BP'),
        net = aabpTrain(train_features, train_labels);
        pred = aabpPredict(net, test_features);
    end

    if strcmp(method, 'LDSVR'),
        para.tol  = 1e-10; %tolerance during the iteration
        para.epsi = 0.1; %epsi-insensitive 
        para.C    = 1; %penalty parameter
        para.ker  = 'rbf'; %type of kernel function ('lin', 'poly', 'rbf', 'sam')
        para.par  = 1*mean(pdist(train_features)); %parameter of kernel function

        modelpara = ldsvrTrain(train_features,train_labels,para);
        pred = ldsvrPredict(test_features, train_features, modelpara);
    end

    if strcmp(method, 'IIS'),
        para.minValue = 1e-7; % the feature value to replace 0, default: 1e-7
        para.iter = 10; % learning iterations, default: 50 / 200 
        para.minDiff = 1e-4; % minimum log-likelihood difference for convergence, default: 1e-7
        para.regfactor = 0; % regularization factor, default: 0
        [weights] = iislldTrain(para, train_features, train_labels);
        pred = lldPredict(weights,test_features);
    end

    if strcmp(method, 'BFGS'),
        trainFeature = train_features;
        trainDistribution = train_labels;
        save('tmp.mat','trainFeature', 'trainDistribution');
        item=eye(size(train_features,2),size(train_labels,2));
        [weights,fval] = bfgslldTrain(@bfgsProcess,item);
        pred = lldPredict(weights,test_features);
    end

    if strcmp(method, 'CPCNN'),
        cpnnStructure.hNumber = 50; % the number of hidden layer, default: 50.
        cpnnStructure.iNumber = size(train_features,2); % the number of input layer, default: 262.
        cpnnStructure.epochs = 100; % the number of iteration times, default: 100.
        cpnnStructure.goal = 5 ; % accurate to five decimal places, default: 5.
        cpnnStructure.showResult = true; %whether show the result. True for show, false for not.
        cpnnStructure=cpnn(cpnnStructure);
        para.itaP = 1.2;
        para.itaN = 0.5;
        model=cpnnTrain(train_features,train_labels,cpnnStructure,para);
        pred=cpnnPredict(test_features,model);
    end
    
    % compute classification
    [precision, recall, accuracy, F1] = ComputeClassificationMetrics(pred, gt);

    % clamp
    pred = max(pred, eps);
    pred = min(pred, 1-eps);
    gt = max(gt, eps);
    gt = min(gt, 1-eps);
    
    pred = bsxfun(@rdivide, pred, sum(pred,2));
    gt = bsxfun(@rdivide, gt, sum(gt,2));

    [dist_name, distance] = ComputeEDLMeasures(gt, pred);
end

