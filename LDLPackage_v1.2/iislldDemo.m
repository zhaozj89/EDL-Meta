%IISLLDDEMO	The example of IISLLD algorithm.
%
%	Description
%   We establish a maximum entropy model and use IIS algorithm to estimate
%   the parameters. In this way, we can get our LDL model. Then a new 
%   distribution can be predicted based on this model.
% 
%	See also
%	LLDPREDICT, IISLLDTRAIN
%	
%   Copyright: Xin Geng (xgeng@seu.edu.cn)
%   School of Computer Science and Engineering, Southeast University
%   Nanjing 211189, P.R.China
%
clear;
clc;
% Load the data set.
load movieDataSet;

% Initialize the model parameters.
para.minValue = 1e-7; % the feature value to replace 0, default: 1e-7
para.iter = 10; % learning iterations, default: 50 / 200 
para.minDiff = 1e-4; % minimum log-likelihood difference for convergence, default: 1e-7
para.regfactor = 0; % regularization factor, default: 0

tic;
% The training part of IISLLD algorithm.
[weights] = iislldTrain(para, trainFeature, trainDistribution);
fprintf('Training time of IIS-LLD: %8.7f \n', toc);

% Prediction
preDistribution = lldPredict(weights,testFeature);
fprintf('Finish prediction of IIS-LLD. \n');

% To visualize two distribution and display some selected metrics of distance
for i=1:testNum
    % Show the comparisons between the predicted distribution
	[disName, distance] = computeMeasures(testDistribution(i,:), preDistribution(i,:));
    % Draw the picture of the real and prediced distribution.
    drawDistribution(testDistribution(i,:),preDistribution(i,:),disName, distance);
    %sign=input('Press any key to continue:');
end
