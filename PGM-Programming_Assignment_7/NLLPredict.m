function pred = NLLPredict (X, theta, modelParams)

    featureSet = GenerateAllFeatures(X, modelParams);
    F = BuildOCRNetwork(size(X,1), theta, featureSet, modelParams.numHiddenStates);
    % Run MAP inference
    M = ComputeExactMarginalsBP(F, [], 1);
    pred = MaxDecoding(M);

end