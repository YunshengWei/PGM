function [P, G, loglikelihood] = LearnGraphAndCPDs(dataset, labels)

% dataset: N x 10 x 3, N poses represented by 10 parts in (y, x, alpha) 
% labels: N x 2 true class labels for the examples. labels(i,j)=1 if the 
%         the ith example belongs to class j
%
% Copyright (C) Daphne Koller, Stanford Univerity, 2012

N = size(dataset, 1);
K = size(labels,2);
Q = size(dataset, 2);

G = zeros(10,2,K); % graph structures to learn
% initialization
for k=1:K
    G(2:end,:,k) = ones(9,2);
end

% estimate graph structure for each class
for k=1:K
    % fill in G(:,:,k)
    % use ConvertAtoG to convert a maximum spanning tree to a graph G
    %%%%%%%%%%%%%%%%%%%%%%%%%
    % YOUR CODE HERE
    data_subset = squeeze(dataset(logical(labels(:, k)), :, :));
    [A, ~] = LearnGraphStructure(data_subset);
    G(: , :, k) = ConvertAtoG(A);
    %%%%%%%%%%%%%%%%%%%%%%%%%
end

% estimate parameters

P.c = mean(labels);
% compute P.c

% the following code can be copied from LearnCPDsGivenGraph.m
% with little or no modification
%%%%%%%%%%%%%%%%%%%%%%%%%
% YOUR CODE HERE
P.clg = repmat(struct('mu_y', [], 'sigma_y', [], ...
                      'mu_x', [], 'sigma_x', [], ...
                      'mu_angle', [], 'sigma_angle', [], ...
                      'theta',[]), 1, Q);
                  
shared = true;
if length(size(G)) == 3
    shared = false;
end

for i = 1:Q
    for j = 1:K
        data_subset = dataset(logical(labels(:, j)), :, :);
        data_subset_i = squeeze(data_subset(:, i, :));
        pa = 0;
        if shared
            if G(i, 1) ~= 0
                pa = G(i, 2);
            end
        else
            if G(i, 1, j) ~= 0
                pa = G(i, 2, j);
            end
        end
        if pa == 0
            mu = mean(data_subset_i);
            P.clg(i).mu_y(j) = mu(1);
            P.clg(i).mu_x(j) = mu(2);
            P.clg(i).mu_angle(j) = mu(3);
            sigma = std(data_subset_i, 1);
            P.clg(i).sigma_y(j) = sigma(1);
            P.clg(i).sigma_x(j) = sigma(2);
            P.clg(i).sigma_angle(j) = sigma(3);
        else
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