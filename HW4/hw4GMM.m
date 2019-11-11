function [labels] = hw4GMM(features,nMixtures)
    labels = zeros(154401,1);
    likelihood = zeros(154401,5);

    GMModel = fitgmdist(features,nMixtures);
    prior = GMModel.ComponentProportion
    likelihood(:,1) = mvnpdf(features,GMModel.mu(1,:),GMModel.Sigma(:,:,1))*prior(1);
    likelihood(:,2) = mvnpdf(features,GMModel.mu(2,:),GMModel.Sigma(:,:,2))*prior(2);
    if nMixtures>2
        likelihood(:,3) = mvnpdf(features,GMModel.mu(3,:),GMModel.Sigma(:,:,3))*prior(3);
        if nMixtures >3
            likelihood(:,4) = mvnpdf(features,GMModel.mu(4,:),GMModel.Sigma(:,:,4))*prior(4);
            if nMixtures > 4
                likelihood(:,5) = mvnpdf(features,GMModel.mu(5,:),GMModel.Sigma(:,:,5))*prior(5);
            end
        end
    end
    
    [maxLikelihoodVal,maxLikelihoodIdx] = max(likelihood,[],2);
    labels(maxLikelihoodIdx==1) = 1;
    labels(maxLikelihoodIdx==2) = 2;
    labels(maxLikelihoodIdx==3) = 3;
    labels(maxLikelihoodIdx==4) = 4;
    labels(maxLikelihoodIdx==5) = 5;