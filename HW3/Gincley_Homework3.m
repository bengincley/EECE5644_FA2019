%% Homework 3
% Benjamin Gincley
% EECE 5644
% 4 November 2019
%% Generate Data
rng(0)
p = [0.1;0.3;0.2;0.4];
mu = 5*[1 1; 1 -1; -1 1; -1 -1];
for i = 1:4
    sigma(:,:,i) = rand(1)*eye(2);
end
dist = gmdistribution(mu,sigma,p);
S1 = random(dist,10);
S2 = random(dist,100);
S3 = random(dist,1000);
S4 = random(dist,10000);
components = 7;
normAvgLL = zeros(4,components);
kfolds = 10;
%% Sample Set 1
currentsample = S1;
nsamples = size(currentsample,1);
% figure()
% scatter(currentsample(:,1),currentsample(:,2),'k.')
% hold on
% Segment into Folds
[trainset,testset] = SampleSplit(nsamples,kfolds,currentsample);
figure()
hold on
scatter(trainset(:,1,1),trainset(:,2,1),'ko')
% scatter(testset(:,1,1),testset(:,2,1),'ro')
% scatter(testset(:,1,2),testset(:,2,2),'go')
% scatter(testset(:,1,3),testset(:,2,3),'bo')

% Fit Model
normAvgLL(1,:) = FitGMModel(components,kfolds,trainset);
%% Sample Set 2
currentsample = S2;
nsamples = size(currentsample,1);
[trainset,testset] = SampleSplit(nsamples,kfolds,currentsample);
normAvgLL(2,:) = FitGMModel(components,kfolds,trainset);
%% Sample Set 3
currentsample = S3;
nsamples = size(currentsample,1);
[trainset,testset] = SampleSplit(nsamples,kfolds,currentsample);
normAvgLL(3,:) = FitGMModel(components,kfolds,trainset);
%% Sample Set 4
currentsample = S4;
nsamples = size(currentsample,1);
[trainset,testset] = SampleSplit(nsamples,kfolds,currentsample);
normAvgLL(4,:) = FitGMModel(components,kfolds,trainset);
%% Plot Results
figure()
ylim([0 1])
line(1:components,normAvgLL,'Marker','o')
title('Normalized Log-Likelihood for GM Models fitting 4 True Components with 1-7 Components Across 4 Data Set Sizes')
xlabel('Number of Model Components')
ylabel('Normalized Negative Log-likelihood')
legend('N = 10 Samples','N = 100 Samples','N = 1000 Samples','N = 10000 Samples')
%gmPDF = @(x1,x2)reshape(pdf(GMModel,[x1(:) x2(:)]),size(x1));
%g = gca;
%fcontour(gmPDF,[g.XLim g.YLim])
% title('{\bf Scatter Plot and Fitted Gaussian Mixture Contours}')
% scatter(predictedMu(:,1),predictedMu(:,2),100,'bd','filled')
% hold off