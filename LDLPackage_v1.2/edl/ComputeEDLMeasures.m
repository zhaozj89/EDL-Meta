function [disName distance] = ComputeEDLMeasures(testDistribution, preDistribution)
%COMPUTEMEASURES        computes different measures between two distributions.
%
%	Description
%   [DISNAME, DISTANCE] = COMPUTEMEASURES(TESTDISTRIBUTION,PREDISTRIBUTION) 
%   computes different measures between tested distributions and  predicted distributions.
%
%   Inputs,
%       TESTDISTRIBUTION:  the matrix of tested distributions  with instances in rows (m x k)
%       PREDISTRIBUTION:  the matrix of predicted distributions with instances in rows (m x k)
%
%   Outputs,
%       DISNAME:   the name of selected measures.
%       DISTANCE:  mean values of selected measures
%
%
%   Copyright: Xin Geng (xgeng@seu.edu.cn)
%   School of Computer Science and Engineering, Southeast University
%   Nanjing 211189, P.R.China
%
disName = {'euclideandist','sorensendist','squaredxdist','kldist','fidelity','intersection'};

distance = zeros(1,6);
% compute Measurements
distance(1,1)=euclideandist(testDistribution, preDistribution);
distance(1,2)=sorensendist(testDistribution, preDistribution);  
distance(1,3)=squaredxdist(testDistribution, preDistribution);
distance(1,4)=kldist(testDistribution, preDistribution);
distance(1,5)=fidelity(testDistribution, preDistribution);
distance(1,6)=intersection(testDistribution, preDistribution);
end

