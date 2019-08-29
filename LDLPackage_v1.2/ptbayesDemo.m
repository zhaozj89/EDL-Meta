%PTBAYESDEMO	The example of PTBayes algorithm.
%
%	Description  
%   A demo of using PTBayes algorithm.
% 
% See also
%       PTBAYESTRAIN, RESAMPLE, BAYES, PTBAYESPREDICT
%   
%   Copyright: Xin Geng (xgeng@seu.edu.cn)
%   School of Computer Science and Engineering, Southeast University
%   Nanjing 211189, P.R.China
%

clear;
clc;
% Load the trainData and testData.
load yeastcoldDataSet;

tic;
% Training of PTBayes
model = ptbayesTrain(trainFeature,trainDistribution);
fprintf('Training time of PT-Bayes: %8.7f \n', toc);
%Prediction of PTBayes
preDistribution = ptbayesPredict(model, testFeature);
fprintf('Finish prediction of PT-Bayes. \n');

% To visualize two distribution and display some selected metrics of distance
for i=1:testNum
    % Show the comparisons between the predicted distribution
	[disName, distance] = computeMeasures(testDistribution(i,:), preDistribution(i,:));
    % Draw the picture of the real and prediced distribution.
    drawDistribution(testDistribution(i,:),preDistribution(i,:),disName, distance);
end



