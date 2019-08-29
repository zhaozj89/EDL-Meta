function distance=euclideandist(rd,pd)
%EUCLIDEANDIST	Calculate the average Euclidean distance between the predicted label
%               distribution and the real label distribution.
%
%	Description
%   DISTANCE = EUCLIDEANDIST(RD, PD) calculate the Euclidean distance between the predicted label
%   distribution and the real label distribution.
%
%   Inputs,
%       RD: real label distribution
%       PD: predicted label distribution
%
%   Outputs,
%       DISTANCE: Euclidean distance
%
%	See also
%	KLDIST, SQUAREDXDIST, SORENSENDIST
%	
%
[rows,cols]=size(rd); % Size of rd, rows*cols
for i=1:rows
    dist(i)=0;
    for j=1:cols
        dist(i)=dist(i)+abs(rd(i,j)-pd(i,j))^2;
    end
    dist(i)=sqrt(dist(i));
end
totalDist=0;
for i=1:rows
    totalDist=totalDist+dist(i);
end
averageDist=totalDist/rows; % Average distance 
distance=averageDist;
end