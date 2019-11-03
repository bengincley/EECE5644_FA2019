%% Homework 3
% Benjamin Gincley
% EECE 5644
% 4 November 2019
%% Generate Data
rng(0)
p = [0.3;0.7];
priorThresh = cumsum([0;p]);
mu{1} = [3,3]; mu{2} = [-1,4];
sigma{1} = [1 0.5; 0.5 1]; sigma{2} = [5 -1.5; -1.5 3];
nClass = 2;
nSamples = 999;
classTemp = rand(nSamples,1);
data = cell(nClass,1);
classLabel = cell(nClass,1);
for idx = 1:nClass
    nSamplesClass = nnz(classTemp>=priorThresh(idx) & classTemp<priorThresh(idx+1));
    data{idx} = mvnrnd(mu{idx},sigma{idx},nSamplesClass);
    classLabel{idx} = ones(nSamplesClass,1)*idx;
end
data = cell2mat(data);
classLabel = cell2mat(classLabel);
nClass1 = sum(classLabel == 1);
nClass2 = sum(classLabel == 2);
figure()
hold on
scatter(data(classLabel == 1,1),data(classLabel == 1,2),'k.')
scatter(data(classLabel == 2,1),data(classLabel == 2,2),'b.')
%% LDA Model
lda = fitcdiscr(data,classLabel);
ldaClass = resubPredict(lda);
ldaResubErr = resubLoss(lda);

% Fisher LDA Classifer (using true model parameters) 
Sb = (mu{1}'-mu{2}')*(mu{1}'-mu{2}')'; 
Sw = sigma{1} + sigma{2}; 
[V,D] = eig(inv(Sw)*Sb); % LDA solution satisfies alpha Sw w = Sb w; ie w is a generalized eigenvector of (Sw,Sb) 
[~,ind] = sort(diag(D),'descend'); 
wLDA = V(:,ind(1)); % Fisher LDA projection vector 
yLDA = wLDA'*data'; % All data projected on to the line spanned by wLDA 
wLDA = sign(mean(yLDA(find(classLabel==2)))-mean(yLDA(find(classLabel==1))))*wLDA; % ensures class1 falls on the + side of the axis 
b = 1.39;
discriminantScoreLDA = b+sign(mean(yLDA(find(classLabel==2)))-mean(yLDA(find(classLabel==1))))*yLDA; % flip yLDA accordingly 
predictedLDA = zeros(nSamples,1);
predictedLDA(discriminantScoreLDA < 0) = 1;
predictedLDA(discriminantScoreLDA >= 0) = 2;
correctLDA = zeros(nSamples,1);
incorrectLDA = zeros(nSamples,1);
correctLDA(predictedLDA == classLabel) = 1;
incorrectLDA(predictedLDA ~= classLabel) = 1;
errorLDA = sum(incorrectLDA)/nSamples

hold on
scatter(data((correctLDA == 1),1),data((correctLDA == 1),2),100,'gs')
scatter(data((correctLDA == 0),1),data((correctLDA == 0),2),100,'rs')

%% Logistic Model
w = wLDA;
y = 1./(1+exp(w'*data'+b))';
logisticPredict = zeros(nSamples,1);
logisticPredict(y>=0.5) = 1;
logisticPredict(y<0.5) = 2;
predictClass1 = sum(logisticPredict==1);
%% Optimize
dataClass1 = data(classLabel==1,:);
dataClass2 = data(classLabel==2,:);
wOptim = zeros(nClass1,2);
for i = 1:nClass1
    dataVal = dataClass1(i,:);
    fun = @(w)1./(1+exp(w'*dataVal'+1.39));
    x0 = wLDA;
    wOptim(i,:) = fminsearch(fun,x0);
end
WOptim = min(wOptim);
for i = 1:nClass1
    dataVal = dataClass1(i,:);
    fun = @(b)1./(1+exp(WOptim*dataVal'+b));
    x0 = 1.39;
    bOptim(i,:) = fminsearch(fun,x0);
end
BOptim = min(bOptim);
%% repeat logistic
w = WOptim;
y = 1./(1+exp(w*data'+BOptim))';
logisticPredict = zeros(nSamples,1);
logisticPredict(y>=0.5) = 2;
logisticPredict(y<0.5) = 1;
predictClass1 = sum(logisticPredict==1);
logisticCorrect = zeros(nSamples,1);
logisticCorrect(logisticPredict == classLabel) = 1;
errorLogistic = 1-(sum(logisticCorrect) / nSamples)
scatter(data(logisticCorrect==1,1),data(logisticCorrect==1,2),'go')
scatter(data(logisticCorrect==0,1),data(logisticCorrect==0,2),'ro')
%% MAP Classifier
likelihood1 = mvnpdf(data,mu{1},sigma{1});
likelihood2 = mvnpdf(data,mu{2},sigma{2});
Posterior(:,1) = likelihood1 * p(1);
Posterior(:,2) = likelihood2 * p(2);
[val, MAP_predict] = max(Posterior,[],2);
errorMAP = sum(MAP_predict ~= classLabel)/nSamples
MAPCorrect = zeros(nSamples,1);
MAPCorrect(MAP_predict == classLabel) = 1;
scatter(data(MAPCorrect==1,1),data(MAPCorrect==1,2),'gx')
scatter(data(MAPCorrect==0,1),data(MAPCorrect==0,2),'rx')
%% Cumulative Plot
title('Combined Plot: Classification by 3 Models: MAP, LDA, Logistic')
xlabel('x1')
ylabel('x2')
legend('True Class 1','True Class 2', 'Correct LDA', 'Incorrect LDA', 'Correct Logistic', 'Incorrect Logistic', 'Correct MAP', 'Incorrect MAP')
hold off
%% Individual Plots
figure()
hold on
scatter(data(classLabel == 1,1),data(classLabel == 1,2),'k.')
scatter(data(classLabel == 2,1),data(classLabel == 2,2),'b.')
scatter(data((correctLDA == 1),1),data((correctLDA == 1),2),'gs')
scatter(data((correctLDA == 0),1),data((correctLDA == 0),2),'rs')
title('Classification by LDA Model: 8.71% Error')
xlabel('x1')
ylabel('x2')
legend('True Class 1','True Class 2', 'Correct LDA', 'Incorrect LDA')
hold off

figure()
hold on
scatter(data(classLabel == 1,1),data(classLabel == 1,2),'k.')
scatter(data(classLabel == 2,1),data(classLabel == 2,2),'b.')
scatter(data(logisticCorrect==1,1),data(logisticCorrect==1,2),'go')
scatter(data(logisticCorrect==0,1),data(logisticCorrect==0,2),'ro')
title('Classification by Logistic Model: 29.23% Error')
xlabel('x1')
ylabel('x2')
legend('True Class 1','True Class 2', 'Correct Logistic', 'Incorrect Logistic')
hold off

figure()
hold on
scatter(data(classLabel == 1,1),data(classLabel == 1,2),'k.')
scatter(data(classLabel == 2,1),data(classLabel == 2,2),'b.')
scatter(data(MAPCorrect==1,1),data(MAPCorrect==1,2),'gx')
scatter(data(MAPCorrect==0,1),data(MAPCorrect==0,2),'rx')
title('Classification by MAP Model: 8.71% Error')
xlabel('x1')
ylabel('x2')
legend('True Class 1','True Class 2', 'Correct MAP', 'Incorrect MAP')
hold off