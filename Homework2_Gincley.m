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
% Generate a gaussian distribution with specific parameters
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
% Calculates likelihood from multivariate pdf
likelihood1 = mvnpdf(labeled_sample(:,1:2),mu1.',sigma1.');
likelihood2 = mvnpdf(labeled_sample(:,1:2),mu2.',sigma2.');
% Sanity check plot
%hold on
% fig7 = figure();
% scatter3(labeled_sample(:,1),labeled_sample(:,2),likelihood1,'kx')
% fig8 = figure();
% scatter3(labeled_sample(:,1),labeled_sample(:,2),likelihood2,'r+')
% Calculates MAP for class 1 and 2
MAP1 = likelihood1 * prior;
MAP2 = likelihood2 * (1-prior);
% Use MAP to predict class, evaluate performance, and plot
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

%% Question 3
labels = labeled_sample(:,3);
sample = labeled_sample(:,1:2);
% Find centroid
centroid1 = [mean(sample(labels==0,1)),mean(sample(labels==0,2))];
centroid2 = [mean(sample(labels==1,1)),mean(sample(labels==1,2))];
% Sanity Check
hw2_gmm_plot(sample(class_predict==0,1:2),sample(class_predict==1,1:2),...
    'Centroid Check')
hold on
plot(centroid1(1),centroid1(2),'*b')
plot(centroid2(1),centroid2(2),'*g')
hold off

% Find between-class scatter
mu = [mean(sample(:,1)),mean(sample(:,2))];
m1 = sum(sample(labels==0,1:2)/nclass1,1);
m2 = sum(sample(labels==1,1:2)/nclass2,1);
SB = (m1 - m2).' * (m1 - m2);
% Find within-class scatter
S1 = (sample-m1).' * (sample-m1);
S2 = (sample-m2).' * (sample-m2);
SW = S1 + S2;
% Find w0
A = inv(SW) * SB;
w = [0.5;0.5];
for i=1:10
    w = A * w;
    w = w/max(w);
end
w;
w0 = inv(SW) * (m1 - m2).';

% Projection onto 1 dimension
projection = sample * w;

% Evaluate LDA model, fisher
model = fitcdiscr(sample,labels);
[fisher,scores] = predict(model,sample);
f_nclass1 = totalSamples - sum(fisher);
f_nclass2 = sum(fisher);
nErrors = sum(fisher(:,1) ~= labels);
errorRate = nErrors / totalSamples
% Plot fisher score on 1D projection
figX = figure();
sgtitle(sprintf('LDA Fisher Scores and Decision Labels; Error rate = %.3f',errorRate))
subplot(1,2,1)
hold on
scatter(projection(fisher==0),scores(fisher==0,1),'kx')
scatter(projection(fisher==1),scores(fisher==1,1),'r+')
hold off
ylabel('Fisher Score'); xlabel('1D Projection'); title('Fisher Score against 1D Projection Value');
legend('Class 1', 'Class 2');

% fig = figure();
% lim = [-4 8];
% scatter(sample(:,1),sample(:,2),3,scores(:,1),'+')
% xlabel('x1'); ylabel('x2')
% xlim(lim); ylim(lim);
% title('Class 1 Fisher Classification Score')
% c = colorbar;
subplot(1,2,2)
hw2_gmm_plot(sample(fisher==0,:),sample(fisher==1,:),...
    'Fisher Prediction for Part 6')