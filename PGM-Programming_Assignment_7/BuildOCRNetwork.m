function F = BuildOCRNetwork(N, theta, featureSet, card)
% Provide your own code here
    singletonFactors = repmat(struct('var', [], 'card', [], 'val', []), N, 1);
    pairwiseFactors = repmat(struct('var', [], 'card', [], 'val', []), N - 1, 1);
    for i = 1:N
        singletonFactors(i).var = i;
        singletonFactors(i).card = card;
        singletonFactors(i).val = zeros(1, card);
    end
    for i = 1:N - 1
        pairwiseFactors(i).var = [i, i + 1];
        pairwiseFactors(i).card = [card card];
        pairwiseFactors(i).val = zeros(1, card * card);
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

        end
    end
    
    for i = 1:length(singletonFactors)
        singletonFactors(i).val = exp(singletonFactors(i).val);
    end
    for i = 1:length(pairwiseFactors)
        pairwiseFactors(i).val = exp(pairwiseFactors(i).val);
    end
     
    F = [singletonFactors; pairwiseFactors];
end