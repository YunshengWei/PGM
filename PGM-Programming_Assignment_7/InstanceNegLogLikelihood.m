% function [nll, grad] = InstanceNegLogLikelihood(X, y, theta, modelParams)
% returns the negative log-likelihood and its gradient, given a CRF with parameters theta,
% on data (X, y). 
%
% Inputs:
% X            Data.                           (numCharacters x numImageFeatures matrix)
%              X(:,1) is all ones, i.e., it encodes the intercept/bias term.
% y            Data labels.                    (numCharacters x 1 vector)
% theta        CRF weights/parameters.         (1 x numParams vector)
%              These are shared among the various singleton / pairwise features.
% modelParams  Struct with three fields:
%   .numHiddenStates     in our case, set to 26 (26 possible characters)
%   .numObservedStates   in our case, set to 2  (each pixel is either on or off)
%   .lambda              the regularization parameter lambda
%
% Outputs:
% nll          Negative log-likelihood of the data.    (scalar)
% grad         Gradient of nll with respect to theta   (numParams x 1 vector)
%
% Copyright (C) Daphne Koller, Stanford Univerity, 2012

function [nll, grad] = InstanceNegLogLikelihood(X, y, theta, modelParams)

    % featureSet is a struct with two fields:
    %    .numParams - the number of parameters in the CRF (this is not numImageFeatures
    %                 nor numFeatures, because of parameter sharing)
    %    .features  - an array comprising the features in the CRF.
    %
    % Each feature is a binary indicator variable, represented by a struct 
    % with three fields:
    %    .var          - a vector containing the variables in the scope of this feature
    %    .assignment   - the assignment that this indicator variable corresponds to
    %    .paramIdx     - the index in theta that this feature corresponds to
    %
    % For example, if we have:
    %   
    %   feature = struct('var', [2 3], 'assignment', [5 6], 'paramIdx', 8);
    %
    % then feature is an indicator function over Y_2 and Y_3, which takes on a value of 1
    % if Y_2 = 5 and Y_3 = 6 (which would be 'e' and 'f'), and 0 otherwise. 
    % Its contribution to the log-likelihood would be theta(8) if it's 1, and 0 otherwise.
    %
    % If you're interested in the implementation details of CRFs, 
    % feel free to read through GenerateAllFeatures.m and the functions it calls!
    % For the purposes of this assignment, though, you don't
    % have to understand how this code works. (It's complicated.)
    
    featureSet = GenerateAllFeatures(X, modelParams);

    % Use the featureSet to calculate nll and grad.
    % This is the main part of the assignment, and it is very tricky - be careful!
    % You might want to code up your own numerical gradient checker to make sure
    % your answers are correct.
    %
    % Hint: you can use CliqueTreeCalibrate to calculate logZ effectively. 
    %       We have halfway-modified CliqueTreeCalibrate; complete our implementation 
    %       if you want to use it to compute logZ.
    
    nll = 0;
    %%%
    % Your code here:
    numCharacters = length(y);
    card = [modelParams.numHiddenStates modelParams.numHiddenStates];
    singletonFactors = repmat(struct('var', [], 'card', [], 'val', []), numCharacters, 1);
    pairwiseFactors = repmat(struct('var', [], 'card', [], 'val', []), numCharacters - 1, 1);
    for i = 1:numCharacters
        singletonFactors(i).var = i;
        singletonFactors(i).card = modelParams.numHiddenStates;
        singletonFactors(i).val = zeros(1, prod(singletonFactors(i).card));
    end
    for i = 1:numCharacters - 1
        pairwiseFactors(i).var = [i, i + 1];
        pairwiseFactors(i).card = card;
        pairwiseFactors(i).val = zeros(1, prod(pairwiseFactors(i).card));
    end

    
    for i = 1:length(featureSet.features)
        feature = featureSet.features(i);
        var = feature.var;
        assignment = feature.assignment;
        if length(var) == 1
            singletonFactors(var).val(assignment) = ...
                singletonFactors(var).val(assignment) ...
                + theta(feature.paramIdx);
        elseif length(var) == 2
            % var should be [k, k+1]
            id = AssignmentToIndex(assignment, card);
            pairwiseFactors(var(1)).val(id) = ...
                pairwiseFactors(var(1)).val(id) ...
                + theta(feature.paramIdx);
        else
            error('Factors must be eithor singleton or pairwise.');
        end
        if isequal(y(var), assignment)
            nll = nll - theta(feature.paramIdx);
        end
    end
    for i = 1:length(singletonFactors)
        singletonFactors(i).val = exp(singletonFactors(i).val);
    end
    for i = 1:length(pairwiseFactors)
        pairwiseFactors(i).val = exp(pairwiseFactors(i).val);
    end
     
    factors = [singletonFactors; pairwiseFactors];
    P = CreateCliqueTree(factors);
    [P, logZ] = CliqueTreeCalibrate(P, false);
    nll = nll + logZ + modelParams.lambda / 2 * (theta' * theta);
    
    grad = modelParams.lambda * theta;
    singletonMarginals = repmat(struct('var', 0, 'card', 0, 'val', []), length(y), 1);
    for i = 1:length(y)
        clique = struct('var', 0, 'card', 0, 'val', []);
        currentVar = i;
        for j = 1:length(P.cliqueList)
            % Find a clique with the variable of interest
            if ~isempty(find(ismember(P.cliqueList(j).var, currentVar), 1))
                % A clique with the variable has been indentified
                clique = P.cliqueList(j);
                break
            end
        end
        singletonMarginals(i) = FactorMarginalization(clique, setdiff(clique.var, currentVar));
        if any(singletonMarginals(i).val ~= 0)
            % Normalize
            singletonMarginals(i).val = singletonMarginals(i).val/sum(singletonMarginals(i).val);
        end
    end
    pairwiseMarginals = repmat(struct('var', 0, 'card', 0, 'val', []), length(y) - 1, 1);
    for i = 1:length(P.cliqueList)
        idx = P.cliqueList(i).var(1);
        pairwiseMarginals(idx) = P.cliqueList(i);
        pairwiseMarginals(idx).val = pairwiseMarginals(idx).val / sum(pairwiseMarginals(idx).val);
    end
    
    for i = 1:length(featureSet.features)
        feature = featureSet.features(i);
        featureVal = isequal(feature.assignment, y(feature.var));
        grad(feature.paramIdx) = grad(feature.paramIdx) - featureVal;
        if length(feature.var) == 1
            expected = singletonMarginals(feature.var).val(...
                AssignmentToIndex(feature.assignment, modelParams.numHiddenStates));
        else
            expected = pairwiseMarginals(feature.var(1)).val(...
                AssignmentToIndex(feature.assignment, card));
        end
        grad(feature.paramIdx) = grad(feature.paramIdx) + expected;
    end
end
