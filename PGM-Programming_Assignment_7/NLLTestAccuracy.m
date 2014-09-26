function [trainCharAcc trainWordAcc testCharAcc testWordAcc] = NLLTestAccuracy(trainData, testData, theta, modelParams)

    % Train set accuracy
    charCount = 0;
    wordCount = 0;
    totalChars = 0;
    for i = 1:length(trainData)
        pred = NLLPredict(trainData(i).X, theta, modelParams);
        acc = (pred == trainData(i).y);
        charCount = charCount + sum(acc);
        totalChars = totalChars + length(acc);
        wordCount = wordCount + all(acc);
    end
    trainCharAcc = charCount/totalChars;
    trainWordAcc = wordCount/length(trainData);

    % Test set accuracy
    charCount = 0;
    wordCount = 0;
    totalChars = 0;
    for i = 1:length(testData)
        pred = NLLPredict(testData(i).X, theta, modelParams);
        acc = (pred == testData(i).y);
        charCount = charCount + sum(acc);
        totalChars = totalChars + length(acc);
        wordCount = wordCount + all(acc);
    end
    testCharAcc = charCount/totalChars;
    testWordAcc = wordCount/length(testData);
end