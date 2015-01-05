% File: RecognizeActions.m
%
% Copyright (C) Daphne Koller, Stanford Univerity, 2012

function [accuracy, predicted_labels] = RecognizeActions(datasetTrain, datasetTest, G, maxIter)

% INPUTS
% datasetTrain: dataset for training models, see PA for details
% datasetTest: dataset for testing models, see PA for details
% G: graph parameterization as explained in PA decription
% maxIter: max number of iterations to run for EM

% OUTPUTS
% accuracy: recognition accuracy, defined as (#correctly classified examples / #total examples)
% predicted_labels: N x 1 vector with the predicted labels for each of the instances in datasetTest, with N being the number of unknown test instances


% Train a model for each action
% Note that all actions share the same graph parameterization and number of max iterations
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% YOUR CODE HERE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
A = length(datasetTrain);
for i = 1:A
    P(i) = EM_HMM(datasetTrain(i).actionData, datasetTrain(i).poseData, G, ...
        datasetTrain(i).InitialClassProb, datasetTrain(i).InitialPairProb, maxIter);
    
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% Classify each of the instances in datasetTest
% Compute and return the predicted labels and accuracy
% Accuracy is defined as (#correctly classified examples / #total examples)
% Note that all actions share the same graph parameterization

L = length(datasetTest.actionData);
N = size(datasetTest.poseData, 1);

loglikelihood = zeros(L, A);
actionData = datasetTest.actionData;
poseData = datasetTest.poseData;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% YOUR CODE HERE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for h = 1:A
    K = size(datasetTrain(h).InitialClassProb, 2);
    Q = size(G, 1);
    logEmissionProb = zeros(N,K);
    for k = 1:K
        for i = 1:Q
            pa = G(i, 2);
            if pa ==0
                mu = repmat([P(h).clg(i).mu_y(k), P(h).clg(i).mu_x(k), P(h).clg(i).mu_angle(k)], N, 1);
            else
                poseData_pa = squeeze(poseData(:, pa, :));
                mu = zeros(N, 3);
                mu(:, 1) = P(h).clg(i).theta(k, 1) + poseData_pa * P(h).clg(i).theta(k, 2:4)';
                mu(:, 2) = P(h).clg(i).theta(k, 5) + poseData_pa * P(h).clg(i).theta(k, 6:8)';
                mu(:, 3) = P(h).clg(i).theta(k, 9) + poseData_pa * P(h).clg(i).theta(k, 10:12)';
            end
            logEmissionProb(:, k) = logEmissionProb(:, k) + ...
                    lognormpdf(poseData(:, i, 1), mu(:, 1), P(h).clg(i).sigma_y(k)) + ...
                    lognormpdf(poseData(:, i, 2), mu(:, 2), P(h).clg(i).sigma_x(k)) + ...
                    lognormpdf(poseData(:, i, 3), mu(:, 3), P(h).clg(i).sigma_angle(k));
        end
    end
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
        Factors(j).val = log(P(h).c);
      
        % doubleton factors
        for j = 1:length(actionData(i).pair_ind)
            t = j + index + 1;
            Factors(t).var = [j, j + 1];
            Factors(t).card = [K, K];
            Factors(t).val = log(P(h).transMatrix(:))';
        end
      
        [~, PCalibrated] = ComputeExactMarginalsHMM(Factors);
        for j = 1:length(PCalibrated.cliqueList)
            factor = PCalibrated.cliqueList(j);
            if length(factor.var) == 1
                continue;
            end
            loglikelihood(i, h) = logsumexp(factor.val);
            break;
        end
    end
end

[~, predicted_labels] = max(loglikelihood, [], 2);
accuracy = mean(predicted_labels == datasetTest.labels);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
