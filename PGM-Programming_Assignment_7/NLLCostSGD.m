function [cost, grad] = NLLCostSGD(trainData, theta, modelParams, i)
    i = mod(i, length(trainData)) + 1;

    [cost grad] = InstanceNegLogLikelihood(trainData(i).X, trainData(i).y, theta, modelParams);

end