% File: EM_cluster.m
%
% Copyright (C) Daphne Koller, Stanford Univerity, 2012

function [P, loglikelihood, ClassProb] = EM_cluster(poseData, G, InitialClassProb, maxIter)

% INPUTS
% poseData: N x 10 x 3 matrix, where N is number of poses;
%   poseData(i,:,:) yields the 10x3 matrix for pose i.
% G: graph parameterization as explained in PA8
% InitialClassProb: N x K, initial allocation of the N poses to the K
%   classes. InitialClassProb(i,j) is the probability that example i belongs
%   to class j
% maxIter: max number of iterations to run EM

% OUTPUTS
% P: structure holding the learned parameters as described in the PA
% loglikelihood: #(iterations run) x 1 vector of loglikelihoods stored for
%   each iteration
% ClassProb: N x K, conditional class probability of the N examples to the
%   K classes in the final iteration. ClassProb(i,j) is the probability that
%   example i belongs to class j

% Initialize variables
N = size(poseData, 1);
K = size(InitialClassProb, 2);
Q = size(poseData, 2);

ClassProb = InitialClassProb;

loglikelihood = zeros(maxIter,1);

P.c = [];
P.clg = repmat(struct('mu_y', [], 'sigma_y', [], ...
                      'mu_x', [], 'sigma_x', [], ...
                      'mu_angle', [], 'sigma_angle', [], ...
                      'theta',[]), 1, Q);

% EM algorithm
for iter=1:maxIter
  % M-STEP to estimate parameters for Gaussians
  %
  % Fill in P.c with the estimates for prior class probabilities
  % Fill in P.clg for each body part and each class
  % Make sure to choose the right parameterization based on G(i,1)
  %
  % Hint: This part should be similar to your work from PA8
  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % YOUR CODE HERE
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  P.c = mean(ClassProb);
  for k = 1:K
      ClassProb_k = ClassProb(:, k);
      for i = 1:Q
          pa = G(i, 2);
          if pa == 0
              [P.clg(i).mu_y(k), P.clg(i).sigma_y(k)] = FitG(poseData(:, i, 1), ClassProb_k);
              [P.clg(i).mu_x(k), P.clg(i).sigma_x(k)] = FitG(poseData(:, i, 2), ClassProb_k);
              [P.clg(i).mu_angle(k), P.clg(i).sigma_angle(k)] = FitG(poseData(:, i, 3), ClassProb_k);
          else
              poseData_pa = squeeze(poseData(:, pa, :));
              [P.clg(i).theta(k, [2, 3, 4, 1]), P.clg(i).sigma_y(k)] = ...
                  FitLG(poseData(:, i, 1), poseData_pa, ClassProb_k);
              [P.clg(i).theta(k, [6, 7, 8, 5]), P.clg(i).sigma_x(k)] = ...
                  FitLG(poseData(:, i, 2), poseData_pa, ClassProb_k);
              [P.clg(i).theta(k, [10 ,11 ,12, 9]), P.clg(i).sigma_angle(k)] = ...
                  FitLG(poseData(:, i, 3), poseData_pa, ClassProb_k);
          end
      end
  end
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
  % E-STEP to re-estimate ClassProb using the new parameters
  %
  % Update ClassProb with the new conditional class probabilities.
  % Recall that ClassProb(i,j) is the probability that example i belongs to
  % class j.
  %
  % You should compute everything in log space, and only convert to
  % probability space at the end.
  %
  % Tip: To make things faster, try to reduce the number of calls to
  % lognormpdf, and inline the function (i.e., copy the lognormpdf code
  % into this file)
  %
  % Hint: You should use the logsumexp() function here to do
  % probability normalization in log space to avoid numerical issues
  
  ClassProb = zeros(N,K);
  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % YOUR CODE HERE
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  for k = 1:K
      ClassProb(:, k) = log(P.c(k));
      for i = 1:Q
          pa = G(i, 2);
          if pa ==0
              mu = repmat([P.clg(i).mu_y(k), P.clg(i).mu_x(k), P.clg(i).mu_angle(k)], N, 1);
          else
              poseData_pa = squeeze(poseData(:, pa, :));
              mu = zeros(N, 3);
              mu(:, 1) = P.clg(i).theta(k, 1) + poseData_pa * P.clg(i).theta(k, 2:4)';
              mu(:, 2) = P.clg(i).theta(k, 5) + poseData_pa * P.clg(i).theta(k, 6:8)';
              mu(:, 3) = P.clg(i).theta(k, 9) + poseData_pa * P.clg(i).theta(k, 10:12)';
          end
          ClassProb(:, k) = ClassProb(:, k) + ...
                  lognormpdf(poseData(:, i, 1), mu(:, 1), P.clg(i).sigma_y(k)) + ...
                  lognormpdf(poseData(:, i, 2), mu(:, 2), P.clg(i).sigma_x(k)) + ...
                  lognormpdf(poseData(:, i, 3), mu(:, 3), P.clg(i).sigma_angle(k));
      end
  end
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
  % Compute log likelihood of dataset for this iteration
  % Hint: You should use the logsumexp() function here
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % YOUR CODE HERE
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  logSumProb = logsumexp(ClassProb);
  loglikelihood(iter) = sum(logSumProb);
  ClassProb = exp(bsxfun(@minus, ClassProb, logSumProb));
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
  % Print out loglikelihood
  fprintf('EM iteration %d: log likelihood: %f\n', ...
    iter, loglikelihood(iter));
  if exist('OCTAVE_VERSION')
    fflush(stdout);
  end
  
  % Check for overfitting: when loglikelihood decreases
  if iter > 1
    if loglikelihood(iter) < loglikelihood(iter-1)
      break;
    end
  end
  
end

% Remove iterations if we exited early
loglikelihood = loglikelihood(1:iter);
