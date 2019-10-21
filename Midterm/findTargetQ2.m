function [posterior] = findTargetQ2(c,x,landmark,r,var,logprior)
nTiles = size(x,2);
posterior = zeros(size(x,2),size(x,2));
loglikelihood = zeros(1,4);
for i = 1:nTiles
    for j = 1:nTiles
        testpoint = [x(i);x(j)];
        for k = 1:c
            testpointdist = norm(testpoint-landmark(:,k));
            loglikelihood(k) = log(mvnpdf(testpointdist,r(k),var(1,1)));
        end
        sumloglikelihood = sum(loglikelihood);
        posterior(i,j) = sumloglikelihood + logprior;
    end
end