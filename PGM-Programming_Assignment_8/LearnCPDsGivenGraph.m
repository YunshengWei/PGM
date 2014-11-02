function [P, loglikelihood] = LearnCPDsGivenGraph(dataset, G, labels)
%
% Inputs:
% dataset: N x 10 x 3, N poses represented by 10 parts in (y, x, alpha)
% G: graph parameterization as explained in PA description
% labels: N x 2 true class labels for the examples. labels(i,j)=1 if the 
%         the ith example belongs to class j and 0 elsewhere        
%
% Outputs:
% P: struct array parameters (explained in PA description)
% loglikelihood: log-likelihood of the data (scalar)
%
% Copyright (C) Daphne Koller, Stanford Univerity, 2012

K = size(labels,2);
Q = size(G, 1);

P.clg = repmat(struct('mu_y', [], 'sigma_y', [], ...
                      'mu_x', [], 'sigma_x', [], ...
                      'mu_angle', [], 'sigma_angle', [], ...
                      'theta',[]), 1, Q);

% estimate parameters
% fill in P.c, MLE for class probabilities
% fill in P.clg for each body part and each class
% choose the right parameterization based on G(i,1)
% compute the likelihood - you may want to use ComputeLogLikelihood.m
% you just implemented.
%%%%%%%%%%%%%%%%%%%%%%%%%
% YOUR CODE HERE
P.c = mean(labels);
for i = 1:Q
    for j = 1:K
        data_subset = dataset(logical(labels(:, j)), :, :);
        data_subset_i = squeeze(data_subset(:, i, :));
        if G(i, 1) == 0
            mu = mean(data_subset_i);
            P.clg(i).mu_y(j) = mu(1);
            P.clg(i).mu_x(j) = mu(2);
            P.clg(i).mu_angle(j) = mu(3);
            sigma = std(data_subset_i, 1);
            P.clg(i).sigma_y(j) = sigma(1);
            P.clg(i).sigma_x(j) = sigma(2);
            P.clg(i).sigma_angle(j) = sigma(3);
        else
            pa = G(i, 2);
            data_subset_pa = squeeze(data_subset(:, pa, :));
            [P.clg(i).theta(j, [2, 3, 4, 1]), P.clg(i).sigma_y(j)] = ...
                FitLinearGaussianParameters(data_subset_i(:, 1), data_subset_pa);
            [P.clg(i).theta(j, [6, 7, 8, 5]), P.clg(i).sigma_x(j)] = ...
                FitLinearGaussianParameters(data_subset_i(:, 2), data_subset_pa);
            [P.clg(i).theta(j, [10 ,11 ,12, 9]), P.clg(i).sigma_angle(j)] = ...
                FitLinearGaussianParameters(data_subset_i(:, 3), data_subset_pa);
        end
    end
end
loglikelihood = ComputeLogLikelihood(P, G, dataset);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('log likelihood: %f\n', loglikelihood);

