%% Homework 2
% Written by Benjamin Gincley | 7 October 2019
%% Setup
totalSamples = 400;
nMeasurements = 2;
s = rng;
r = rand(1,totalSamples);
%% Part 1
prior = 0.5;
nclass1 = size(find(r<prior),2);
nclass2 = totalSamples - nclass1;
mu1 = [0;0];
mu2 = [3;3];
sigma1 = [1 0;0 1];
sigma2 = sigma1;
sample1 = sample_generator(nclass1,nMeasurements,mu1,sigma1).';
sample2 = sample_generator(nclass2,nMeasurements,mu2,sigma2).';
labeled_sample = sample_generator_hw2(totalSamples,nclass1,sample1,sample2);
fig1 = figure();
hw2_gmm_plot(sample1,sample2,'Part 1: Means [0,0] [3,3] | Equal Prior | Equal Covariance')
%% Part 2
sigma1 = [3 1;1 0.8];
sigma2 = sigma1;
sample1 = sample_generator(nclass1,nMeasurements,mu1,sigma1).';
sample2 = sample_generator(nclass2,nMeasurements,mu2,sigma2).';
labeled_sample = sample_generator_hw2(totalSamples,nclass1,sample1,sample2);
fig2 = figure();
hw2_gmm_plot(sample1,sample2,'Part 2: Means [0,0] [3,3] | Equal Prior | Unequal Covariance')
%% Part 3
mu2 = [2;2];
sigma1 = [2,0.5;0.5,1];
sigma2 = [2,-1.9;-1.9,5];
sample1 = sample_generator(nclass1,nMeasurements,mu1,sigma1).';
sample2 = sample_generator(nclass2,nMeasurements,mu2,sigma2).';
labeled_sample = sample_generator_hw2(totalSamples,nclass1,sample1,sample2);
fig3 = figure();
hw2_gmm_plot(sample1,sample2,'Part 3: Means [0,0] [2,2] | Equal Prior | Unequal Covariance')
%% Prior Change
prior = 0.05;
%% Part 4
nclass1 = size(find(r<prior),2);
nclass2 = totalSamples - nclass1;
mu1 = [0;0];
mu2 = [3;3];
sigma1 = [1 0;0 1];
sigma2 = sigma1;
sample1 = sample_generator(nclass1,nMeasurements,mu1,sigma1).';
sample2 = sample_generator(nclass2,nMeasurements,mu2,sigma2).';
labeled_sample = sample_generator_hw2(totalSamples,nclass1,sample1,sample2);
fig4 = figure();
hw2_gmm_plot(sample1,sample2,'Part 4: Means [0,0] [3,3] | Unequal Prior | Equal Covariance')
%% Part 5
sigma1 = [3 1;1 0.8];
sigma2 = sigma1;
sample1 = sample_generator(nclass1,nMeasurements,mu1,sigma1).';
sample2 = sample_generator(nclass2,nMeasurements,mu2,sigma2).';
labeled_sample = sample_generator_hw2(totalSamples,nclass1,sample1,sample2);
fig5 = figure();
hw2_gmm_plot(sample1,sample2,'Part 5: Means [0,0] [3,3] | Unequal Prior | Unequal Covariance')
%% Part 6
mu2 = [2;2];
sigma1 = [2,0.5;0.5,1];
sigma2 = [2,-1.9;-1.9,5];
sample1 = sample_generator(nclass1,nMeasurements,mu1,sigma1).';
sample2 = sample_generator(nclass2,nMeasurements,mu2,sigma2).';
labeled_sample = sample_generator_hw2(totalSamples,nclass1,sample1,sample2);
fig3 = figure();
hw2_gmm_plot(sample1,sample2,'Part 6: Means [0,0] [2,2] | Unequal Prior | Unequal Covariance')
%% Likelihood
likelihood1 = mvnpdf(labeled_sample(:,1:2),mu1.',sigma1.');
likelihood2 = mvnpdf(labeled_sample(:,1:2),mu2.',sigma2.');
%hold on
% fig7 = figure();
% scatter3(labeled_sample(:,1),labeled_sample(:,2),likelihood1,'kx')
% fig8 = figure();
% scatter3(labeled_sample(:,1),labeled_sample(:,2),likelihood2,'r+')

MAP1 = likelihood1 * prior;
MAP2 = likelihood2 * (1-prior);

class_predict = MAP1<MAP2;
correct = sum(class_predict==labeled_sample(:,3))/totalSamples
error = 1-correct
n_error = error*totalSamples
predicted_sample = zeros(totalSamples,3);
predicted_sample(:,1:2) = labeled_sample(:,1:2);
predicted_sample(:,3) = class_predict;
fig8 = figure();
hw2_gmm_plot(labeled_sample(class_predict==0,1:2),labeled_sample(class_predict==1,1:2),...
    'MAP Prediction for Part 6')
