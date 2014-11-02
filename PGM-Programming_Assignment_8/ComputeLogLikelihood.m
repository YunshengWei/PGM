function loglikelihood = ComputeLogLikelihood(P, G, dataset)
% returns the (natural) log-likelihood of data given the model and graph structure
%
% Inputs:
% P: struct array parameters (explained in PA description)
% G: graph structure and parameterization (explained in PA description)
%
%    NOTICE that G could be either 10x2 (same graph shared by all classes)
%    or 10x2x2 (each class has its own graph). your code should compute
%    the log-likelihood using the right graph.
%
% dataset: N x 10 x 3, N poses represented by 10 parts in (y, x, alpha)
% 
% Output:
% loglikelihood: log-likelihood of the data (scalar)
%
% Copyright (C) Daphne Koller, Stanford Univerity, 2012

N = size(dataset,1); % number of examples
K = length(P.c); % number of classes
Q = size(G, 1); % number of parts

loglikelihood = 0;
% You should compute the log likelihood of data as in eq. (12) and (13)
% in the PA description
% Hint: Use lognormpdf instead of log(normpdf) to prevent underflow.
%       You may use log(sum(exp(logProb))) to do addition in the original
%       space, sum(Prob).
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% YOUR CODE HERE
shared = true;
if length(size(G)) == 3
    shared = false;
end

for i = 1:N
    p = 0;
    for j = 1:K
        prob = log(P.c(j));
        for k = 1:Q
            pa = 0;
            if shared
                if G(k, 1) ~= 0
                    pa = G(k, 2);
                end
            else
                if G(k, 1, j) ~= 0
                    pa = G(k, 2, j);
                end
            end
            if pa == 0
                prob = prob + lognormpdf(dataset(i, k, 1), ...
                       P.clg(k).mu_y(j), P.clg(k).sigma_y(j));
                prob = prob + lognormpdf(dataset(i, k, 2), ...
                       P.clg(k).mu_x(j), P.clg(k).sigma_x(j));
                prob = prob + lognormpdf(dataset(i, k, 3), ...
                       P.clg(k).mu_angle(j), P.clg(k).sigma_angle(j));
            else
                prob = prob + lognormpdf(dataset(i, k, 1), ...
                       P.clg(k).theta(j, 1:4) * ...
                       [1; squeeze(dataset(i, pa, :))], P.clg(k).sigma_y(j));
                prob = prob + lognormpdf(dataset(i, k, 2), ...
                       P.clg(k).theta(j, 5:8) * ...
                       [1; squeeze(dataset(i, pa, :))], P.clg(k).sigma_x(j));
                prob = prob + lognormpdf(dataset(i, k, 3), ...
                       P.clg(k).theta(j, 9:12) * ...
                       [1; squeeze(dataset(i, pa, :))], P.clg(k).sigma_angle(j));
            end
        end
        p = p + exp(prob);
    end
    loglikelihood = loglikelihood + log(p);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
