function [trainset,testset] = SampleSplit(nsamples,kfolds,currentsample)
    foldsize = nsamples / kfolds;    
    testidx = zeros(10,nsamples);
    for i = 1:kfolds
        testidx(i,:) = zeros(1,nsamples);
        testidx(i,(i-1)*foldsize+linspace(1,foldsize,foldsize)) = 1;
    end
    testidx = logical(testidx)';
    trainidx = ~testidx;
    trainset = zeros(0.9*nsamples,2,kfolds);
    testset = zeros(0.1*nsamples,2,kfolds);
    for i = 1:kfolds
        trainset(:,:,i) = currentsample(trainidx(:,i),:);
        testset(:,:,i) = currentsample(testidx(:,i),:);
    end