function sample = sample_generator_hw2(totalSamples,nclass1,sample1,sample2)
    sample = zeros(totalSamples,3);
    sample(1:nclass1,1:2) = sample1;
    sample(1:nclass1,3) = 0;
    sample(nclass1+1:end,1:2) = sample2;
    sample(nclass1+1:end,3) = 1;
end