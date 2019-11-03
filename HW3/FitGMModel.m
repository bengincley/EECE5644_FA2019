function [normAvgLL] = FitGMModel(components,kfolds,trainset)
    predictedLL = zeros(kfolds,1);
    for c = 1:components
        for i = 1:kfolds
            try
                GMModel = fitgmdist(trainset(:,:,i),c);
            catch exception
                disp('There was an error fitting the Gaussian mixture model')
                error = exception.message
            end
            predictedLL(i) = GMModel.NegativeLogLikelihood;
        end
        avgLL(c) = mean(predictedLL);
    end
    normAvgLL = avgLL./max(avgLL);