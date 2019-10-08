function x = sample_generator(N,n,mu,sigma)     
    % Assumes that mu is a column vector and sigma is a square matrix.     
    % Output x will be of dimensions n by N.     
    z = normrnd(0,1,n,N);     
    x = sigma^(0.5) * z + repmat(mu,[1,N]); 
end