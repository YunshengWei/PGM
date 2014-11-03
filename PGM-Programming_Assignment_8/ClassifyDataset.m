function accuracy = ClassifyDataset(dataset, labels, P, G)
% returns the accuracy of the model P and graph G on the dataset 
%
% Inputs:
% dataset: N x 10 x 3, N test instances represented by 10 parts
% labels:  N x 2 true class labels for the instances.
%          labels(i,j)=1 if the ith instance belongs to class j 
% P: struct array model parameters (explained in PA description)
% G: graph structure and parameterization (explained in PA description) 
%
% Outputs:
% accuracy: fraction of correctly classified instances (scalar)
%
% Copyright (C) Daphne Koller, Stanford Univerity, 2012

N = size(dataset, 1);
K = size(labels, 2);
Q = size(G, 1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% YOUR CODE HERE
correct = 0;

shared = true;
if length(size(G)) == 3
    shared = false;
end

for i = 1:N
    probs = zeros(K, 1);
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
        probs(j) = prob;
    end
    [~, predict] = max(probs);
    if labels(i, predict) == 1
        correct = correct + 1;
    end
end

accuracy = correct / N;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf('Accuracy: %.2f\n', accuracy);