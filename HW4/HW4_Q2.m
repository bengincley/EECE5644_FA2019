%% HW4 Q2
% EECE5644
% Benjamin Gincley
% 8 November 2019
%% Generate Samples
rng(0);
nClasses = 2;
prior = [0.35;0.65];
nSamples = 1000;
mu{1} = [0;0]; sigma{1} = 1*eye(2);
classRandProb = rand(nSamples, 1); 
priorThresh = cumsum([0; prior]); 
data = cell(nClasses, 1); 
classLabel = cell(nClasses, 1);   
for idxClass = 1:nClasses    
    nSamplesClass = nnz(classRandProb>=priorThresh(idxClass) & classRandProb<priorThresh(idxClass+1));     % Generate samples according to class dependent parameters     
    if idxClass == 1
        data{idxClass} = mvnrnd(mu{idxClass}, sigma{idxClass}, nSamplesClass);     % Set class labels     
    else
        radius = rand([nSamplesClass 1])+2;
        angle = 2*pi*rand([nSamplesClass 1])-pi;
        [point(:,1), point(:,2)] = pol2cart(angle,radius);
        data{idxClass} = point;
    end
    classLabel{idxClass} = ones(nSamplesClass,1) * idxClass;  
end
data = cell2mat(data); 
classLabel = cell2mat(classLabel);
varNames = {'x1','x2','class'};
labeledData = cat(2,data,classLabel);
DATA = table(labeledData(:,1),labeledData(:,2),labeledData(:,3),'VariableNames',varNames);

% Plot
figure(); hold on
scatter(data(classLabel==1,1),data(classLabel==1,2),'.k')
scatter(data(classLabel==2,1),data(classLabel==2,2),'.b')
title('Training Data Points')
xlabel('x1')
ylabel('x2')
legend('Class -1','Class +1')
%% Search for Optimal Hyperparameters
nC = 60;
c = linspace(0.01,10,nC);
lossLinSVM = zeros(nC,1);
lossGauSVM = zeros(nC,1);
lossGauSVMScale = zeros(nC,1);
minVal = zeros(3,1);
minIdx = zeros(3,1);
for i=1:nC
    linSVMModel = fitcsvm(DATA,'class','Kfold',10,'KernelFunction','linear','Standardize',true,'BoxConstraint',c(i));
    lossLinSVM(i) = kfoldLoss(linSVMModel);

    gauSVMModel = fitcsvm(DATA,'class','Kfold',10,'KernelFunction','gaussian','Standardize',true,'BoxConstraint',c(i));
    lossGauSVM(i) = kfoldLoss(gauSVMModel);
end
[minVal(1),minIdx(1)] = min(lossLinSVM);
[minVal(2),minIdx(2)] = min(lossGauSVM);
optimBoxConstLin = c(minIdx(1));
optimBoxConstGau = c(minIdx(2));

for i = 1:nC
    gauSVMModel = fitcsvm(DATA,'class','Kfold',10,'KernelFunction','gaussian','Standardize',true,'BoxConstraint',optimBoxConstGau,'KernelScale',c(i));
    lossGauSVMScale(i) = kfoldLoss(gauSVMModel);
end
[minVal(3),minIdx(3)] = min(lossGauSVMScale);
optimScaleGau = cc(minIdx(3));

figure(); hold on
plot(c,lossLinSVM)
plot(c,lossGauSVM)
plot(c,lossGauSVMScale)
legend('Linear SVM C','Gaussian SVM C','Gaussian SVM Scale')
title('Hyperparameter optimization for SVMs')
xlabel('Hyperparameter value')
ylabel('Loss')
%% Optimal Classifier Generation and Evaluation
optimLinSVMModel = fitcsvm(DATA,'class','KernelFunction','linear','Standardize',true,'BoxConstraint',optimBoxConstLin);
optimGauSVMModel = fitcsvm(DATA,'class','KernelFunction','gaussian','Standardize',true,'BoxConstraint',optimBoxConstGau,'KernelScale',optimScaleGau);

predictLinSVMTrain = predict(optimLinSVMModel,DATA);
predictGauSVMTrain = predict(optimGauSVMModel,DATA);
lossLinTrain = sum(predictLinSVMTrain~=classLabel)/nSamples
lossGauTrain = sum(predictGauSVMTrain~=classLabel)/nSamples
lims = [-4 4];
figure(); hold on
scatter(data(classLabel==1,1),data(classLabel==1,2),'.k')
scatter(data(classLabel==2,1),data(classLabel==2,2),'.b')
scatter(data(predictLinSVMTrain==classLabel,1),data(predictLinSVMTrain==classLabel,2),'go')
scatter(data(predictLinSVMTrain~=classLabel,1),data(predictLinSVMTrain~=classLabel,2),'ro')
scatter(data(predictGauSVMTrain==classLabel,1),data(predictGauSVMTrain==classLabel,2),'gx')
scatter(data(predictGauSVMTrain~=classLabel,1),data(predictGauSVMTrain~=classLabel,2),'rx')
legend('True Class -1','True Class +1','Linear SVM Correct','Linear SVM Incorrect','Gaussian SVM Correct','Gaussian SVM Incorrect')
xlabel('x1')
ylabel('x2')
title('Optimized Linear and Guassian Kernel SVMs evaluated on training points')
xlim(lims)
ylim(lims)
%% New Test Data
rng(2)
classRandProb = rand(nSamples, 1); 
priorThresh = cumsum([0; prior]); 
data2 = cell(nClasses, 1); 
classLabel2 = cell(nClasses, 1);   
for idxClass = 1:nClasses    
    nSamplesClass = nnz(classRandProb>=priorThresh(idxClass) & classRandProb<priorThresh(idxClass+1));     % Generate samples according to class dependent parameters     
    if idxClass == 1
        data2{idxClass} = mvnrnd(mu{idxClass}, sigma{idxClass}, nSamplesClass);     % Set class labels     
    else
        radius2 = rand([nSamplesClass 1])+2;
        angle2 = 2*pi*rand([nSamplesClass 1])-pi;
        [point2(:,1), point2(:,2)] = pol2cart(angle2,radius2);
        data2{idxClass} = point2;
    end
    classLabel2{idxClass} = ones(nSamplesClass,1) * idxClass;  
end
data2 = cell2mat(data2); 
classLabel2 = cell2mat(classLabel2);
labeledData2 = cat(2,data2,classLabel2);
NEWDATA = table(labeledData2(:,1),labeledData2(:,2),labeledData2(:,3),'VariableNames',varNames);
%% Predict on Test Data and Plot Result
predictLinSVMTest = predict(optimLinSVMModel,NEWDATA);
predictGauSVMTest = predict(optimGauSVMModel,NEWDATA);
lossLinTest = sum(predictLinSVMTest~=classLabel2)/nSamples
lossGauTest = sum(predictGauSVMTest~=classLabel2)/nSamples

figure(); hold on
scatter(data2(classLabel2==1,1),data2(classLabel2==1,2),'k.')
scatter(data2(classLabel2==2,1),data2(classLabel2==2,2),'b.')
scatter(data2(predictLinSVMTest==classLabel2,1),data2(predictLinSVMTest==classLabel2,2),'go')
scatter(data2(predictLinSVMTest~=classLabel2,1),data2(predictLinSVMTest~=classLabel2,2),'ro')
scatter(data2(predictGauSVMTest==classLabel2,1),data2(predictGauSVMTest==classLabel2,2),'gx')
scatter(data2(predictGauSVMTest~=classLabel2,1),data2(predictGauSVMTest~=classLabel2,2),'rx')
legend('True Class -1','True Class +1','Linear SVM Correct','Linear SVM Incorrect','Gaussian SVM Correct','Gaussian SVM Incorrect')
xlabel('x1')
ylabel('x2')
title('Optimized Linear and Guassian Kernel SVMs evaluated using 1000 new test points')
xlim(lims)
ylim(lims)