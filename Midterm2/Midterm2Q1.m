%% EECE5644 Midterm Exam 2 
% Question 1
% Benjamin Gincley
% 16 November 2019
%% Load Data
data = csvread('Q1.csv');
labels = data(:,3);
nSamples = size(data,1);
uniqueX1 = unique(data(:,1));
uniqueX2 = unique(data(:,2));
uniqueLabels = unique(data(:,3));
figure(); hold on
scatter(data(labels==-1,1),data(labels==-1,2),'ro')
scatter(data(labels==1,1),data(labels==1,2),'kx')
xlabel('x1'); ylabel('x2'); title('Measurements x1 and x2 for Classes -1 and 1');
legend('Class -1','Class 1'); grid on
%% Partition
cvp = cvpartition(labels,'HoldOut',0.1);
trainSet = cvp.training;
testSet = cvp.test;
%% Prepare for tree
[x1sort,x1_I] = sort(data(:,1));
[x2sort,x2_I] = sort(data(:,2));
x1sortlabels = labels(x1_I);
x2sortlabels = labels(x2_I);

pClassA = sum(x1sortlabels==-1)/length(x1sortlabels);
nClassB = sum(x1sortlabels==1)/length(x1sortlabels);
purityLeft=zeros(nSamples-1,1);
purityRight=zeros(nSamples-1,1);
for i=1:nSamples-1
    nLeft = i;
    nRight = nSamples-nLeft;
    comparator = x1sort(nLeft);
    
    poolLeft = x1sort<=comparator;
    poolRight = x1sort>comparator;
    labelsLeft = x1sortlabels(poolLeft);
    labelsRight = x1sortlabels(poolRight);
    
    nlabelALeft(i) = sum(labelsLeft==-1);
    nlabelBLeft(i) = sum(labelsLeft==1);
    nlabelALeft(i) = sum(labelsLeft==-1);
    nlabelBLeft(i) = sum(labelsLeft==1);
    nlabelARight(i) = sum(labelsRight==-1);
    nlabelBRight(i) = sum(labelsRight==1);
    purityLeft(i) = nlabelALeft(i)*(1-nlabelALeft(i)/nLeft)+nlabelBLeft(i)*(1-nlabelBLeft(i)/nLeft);
    purityRight(i) = nlabelARight(i)*(1-nlabelARight(i)/nRight)+nlabelBRight(i)*(1-nlabelBRight(i)/nRight);
    weightedPurity(i) = (nLeft*purityLeft(i)+nRight*purityRight(i))/nSamples;
end
[lowestEntropy(1),optimSplit(1)] = min(weightedPurity)
for i=1:nSamples-1
    nLeft = i;
    nRight = nSamples-nLeft;
    comparator = x2sort(nLeft);
    
    poolLeft = x2sort<=comparator;
    poolRight = x2sort>comparator;
    labelsLeft = x2sortlabels(poolLeft);
    labelsRight = x2sortlabels(poolRight);
    
    nlabelALeft(i) = sum(labelsLeft==-1);
    nlabelBLeft(i) = sum(labelsLeft==1);
    nlabelALeft(i) = sum(labelsLeft==-1);
    nlabelBLeft(i) = sum(labelsLeft==1);
    nlabelARight(i) = sum(labelsRight==-1);
    nlabelBRight(i) = sum(labelsRight==1);
    purityLeft(i) = nlabelALeft(i)*(1-nlabelALeft(i)/nLeft)+nlabelBLeft(i)*(1-nlabelBLeft(i)/nLeft);
    purityRight(i) = nlabelARight(i)*(1-nlabelARight(i)/nRight)+nlabelBRight(i)*(1-nlabelBRight(i)/nRight);
    entropyLeft(i) = [];
    weightedPurity(i) = (nLeft*purityLeft(i)+nRight*purityRight(i))/nSamples;
end
[lowestEntropy(2),optimSplit(2)] = min(weightedPurity)
[globallyLowestEntropy,xOptim] = min(lowestEntropy)
globallyOptimSplit = optimSplit(xOptim)