%PTSVMDEMO	The example of PTsvm algorithm.
%
%	Description  
%   A demo of using PTsvm algorithm.
%
%	See also   
%   PTSVMTRAIN, PTSVMPREDICT
%
%   Copyright: Xin Geng (xgeng@seu.edu.cn)
%   School of Computer Science and Engineering, Southeast University
%   Nanjing 211189, P.R.China
%

clear;
clc;
% Load the trainData and testData.
load yeastcoldDataSet;

% To show the result of this demo qucikly, here only use a part of trainData to speed up the training time.
% please annotate following tow statementss when you use ptsvm.
trainFeature = trainFeature(1:1000, :);
trainDistribution = trainDistribution(1:1000,:);

tic;
% Training of PTsvm
model = ptsvmTrain(trainFeature,trainDistribution);
fprintf('Training time of PT-SVM: %8.7f \n', toc);

%Prediction of PTsvm
preDistribution = ptsvmPredict(model, testFeature);
fprintf('Finish prediction of PT-SVM,. \n');

% To visualize two distribution and display some selected metrics of distance
for i=1:testNum
    % Show the comparisons between the predicted distribution
	[disName, distance] = computeMeasures(testDistribution(i,:), preDistribution(i,:));
    % Draw the picture of the real and prediced distribution.
    drawDistribution(testDistribution(i,:),preDistribution(i,:),disName, distance);
end
