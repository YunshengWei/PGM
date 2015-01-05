% File: EM_HMM.m
%
% Copyright (C) Daphne Koller, Stanford Univerity, 2012

function [P, loglikelihood, ClassProb, PairProb] = EM_HMM(actionData, poseData, G, InitialClassProb, InitialPairProb, maxIter)

% INPUTS
% actionData: structure holding the actions as described in the PA
% poseData: N x 10 x 3 matrix, where N is number of poses in all actions
% G: graph parameterization as explained in PA description
% InitialClassProb: N x K matrix, initial allocation of the N poses to the K
%   states. InitialClassProb(i,j) is the probability that example i belongs
%   to state j.
%   This is described in more detail in the PA.
% InitialPairProb: V x K^2 matrix, where V is the total number of pose
%   transitions in all HMM action models, and K is the number of states.
%   This is described in more detail in the PA.
% maxIter: max number of iterations to run EM

% OUTPUTS
% P: structure holding the learned parameters as described in the PA
% loglikelihood: #(iterations run) x 1 vector of loglikelihoods stored for
%   each iteration
% ClassProb: N x K matrix of the conditional class probability of the N examples to the
%   K states in the final iteration. ClassProb(i,j) is the probability that
%   example i belongs to state j. This is described in more detail in the PA.
% PairProb: V x K^2 matrix, where V is the total number of pose transitions
%   in all HMM action models, and K is the number of states. This is
%   described in more detail in the PA.

% Initialize variables
N = size(poseData, 1);
Q = size(poseData, 2); % number of parts
K = size(InitialClassProb, 2);
L = size(actionData, 2); % number of actions
V = size(InitialPairProb, 1);

ClassProb = InitialClassProb;
PairProb = InitialPairProb;

loglikelihood = zeros(maxIter,1);

P.clg = repmat(struct('mu_y', [], 'sigma_y', [], ...
                      'mu_x', [], 'sigma_x', [], ...
                      'mu_angle', [], 'sigma_angle', [], ...
                      'theta',[]), 1, Q);
% EM algorithm
for iter=1:maxIter
  
  % M-STEP to estimate parameters for Gaussians
  % Fill in P.c, the initial state prior probability (NOT the class probability as in PA8 and EM_cluster.m)
  % Fill in P.clg for each body part and each class
  % Make sure to choose the right parameterization based on G(i,1)
  % Hint: This part should be similar to your work from PA8 and EM_cluster.m
  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % YOUR CODE HERE
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  P.c = zeros(1,K);
  for i = 1:L
      P.c = P.c + ClassProb(actionData(i).marg_ind(1), :);
  end
  P.c = P.c / L;
  
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
  
  % M-STEP to estimate parameters for transition matrix
  % Fill in P.transMatrix, the transition matrix for states
  % P.transMatrix(i,j) is the probability of transitioning from state i to state j
  P.transMatrix = zeros(K,K);
  
  % Add Dirichlet prior based on size of poseData to avoid 0 probabilities
  P.transMatrix = P.transMatrix + size(PairProb,1) * .05;
  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % YOUR CODE HERE
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  P.transMatrix = P.transMatrix + reshape(sum(PairProb), K, K);
  P.transMatrix = P.transMatrix ./ repmat(sum(P.transMatrix, 2), 1, K);
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
    
  % E-STEP preparation: compute the emission model factors (emission probabilities) in log space for each 
  % of the poses in all actions = log( P(Pose | State) )
  % Hint: This part should be similar to (but NOT the same as) your code in EM_cluster.m
  
  logEmissionProb = zeros(N,K);
  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % YOUR CODE HERE
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  for k = 1:K
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
          logEmissionProb(:, k) = logEmissionProb(:, k) + ...
                  lognormpdf(poseData(:, i, 1), mu(:, 1), P.clg(i).sigma_y(k)) + ...
                  lognormpdf(poseData(:, i, 2), mu(:, 2), P.clg(i).sigma_x(k)) + ...
                  lognormpdf(poseData(:, i, 3), mu(:, 3), P.clg(i).sigma_angle(k));
      end
  end
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
    
  % E-STEP to compute expected sufficient statistics
  % ClassProb contains the conditional class probabilities for each pose in all actions
  % PairProb contains the expected sufficient statistics for the transition CPDs (pairwise transition probabilities)
  % Also compute log likelihood of dataset for this iteration
  % You should do inference and compute everything in log space, only converting to probability space at the end
  % Hint: You should use the logsumexp() function here to do probability normalization in log space to avoid numerical issues
  
  ClassProb = zeros(N,K);
  PairProb = zeros(V,K^2);
  loglikelihood(iter) = 0;
  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % YOUR CODE HERE
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  for i = 1:L
      index = length(actionData(i).marg_ind);
      Factors = repmat(struct('var', [], 'card', [], 'val', []), ...
          1, index * 2);
      % singleton factors
      for j = 1:index
          Factors(j).var = j;
          Factors(j).card = K;
          Factors(j).val = logEmissionProb(actionData(i).marg_ind(j), :);
      end
      j = index + 1;
      Factors(j).var = 1;
      Factors(j).card = K;
      Factors(j).val = log(P.c);
      
      % doubleton factors
      for j = 1:length(actionData(i).pair_ind)
          t = j + index + 1;
          Factors(t).var = [j, j + 1];
          Factors(t).card = [K, K];
          Factors(t).val = log(P.transMatrix(:))';
      end
      
      [M, PCalibrated] = ComputeExactMarginalsHMM(Factors);
      for j = 1:length(M)
          ClassProb(actionData(i).marg_ind(M(j).var), :) = exp(M(j).val);
      end
      
      used = false;
      logSumProb = 0;
      for j = 1:length(PCalibrated.cliqueList)
          factor = PCalibrated.cliqueList(j);
          if length(factor.var) == 1
              continue;
          end
          if ~used
              used = true;
              logSumProb = logsumexp(factor.val);
              loglikelihood(iter) = loglikelihood(iter) + logSumProb;
          end
          PairProb(actionData(i).pair_ind(factor.var(1)), :) = exp(factor.val - logSumProb);
      end
  end
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
  % Print out loglikelihood
  fprintf('EM iteration %d: log likelihood: %f\n', ...
    iter, loglikelihood(iter));
  if exist('OCTAVE_VERSION')
    fflush(stdout);
  end
  
  % Check for overfitting by decreasing loglikelihood
  if iter > 1
    if loglikelihood(iter) < loglikelihood(iter-1)
      break;
    end
  end
  
end

% Remove iterations if we exited early
loglikelihood = loglikelihood(1:iter);
