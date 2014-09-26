function thetaOpt = NLLTrainSGD(trainData, modelParams, iter)

    numParams = modelParams.numHiddenStates + modelParams.numHiddenStates^2 + modelParams.numHiddenStates*size(trainData(1).X,2)*modelParams.numObservedStates;

    % This sets up an anonymous function gradFn
    % such that gradFn(theta, i) = NLLCostSGD(trainData, theta, modelParams, i).
     gradFn = @(theta, i)NLLCostSGD(trainData, theta, modelParams, i);

    % Calculate optimal theta values
    thetaOpt = StochasticGradientDescent(gradFn, zeros(1, numParams), iter);

end