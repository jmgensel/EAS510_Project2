from helpers import gaussian

# Read given .csv, create classifier, then test.
def main (testData, trainData):
    # Initialization
    predictions = {}
    testDataActualClasses = {}

    # Calculate the Bayesian classifier.
    (meanY, meanN, sdY, sdN, fractionFlag, fractionNoFlag) = calculateClassifier(trainData)

    # Estimate classes for test data.
    rowNum = 0
    for row in testData:
        # Save actual class to check later
        testDataActualClasses[rowNum] = row["at_risk_flag"]

        scoreY = fractionFlag
        scoreN = fractionNoFlag

        for k, v in row.items():
            scoreY *= gaussian(v, meanY[k], sdY[k])
            scoreN *= gaussian(v, meanN[k], sdN[k])

        if scoreY < scoreN:
            predictions[rowNum] = 1.0
        else:
            predictions[rowNum] = 0.0

        rowNum += 1

    # Check if estimates were correct.
    numCorrect = 0
    numIncorrect = 0
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for k, v in predictions.items():
        if v == testDataActualClasses[k]:
            numCorrect += 1
            if v == 1.0: TP += 1
            else: TN += 1
        else:
            numIncorrect += 1
            if v == 1.0: FP += 1
            else: FN += 1
    acc = numCorrect / (numCorrect + numIncorrect)
    pre = TP / numCorrect
    rec = TP / (TP+FN)

    # print(str(numCorrect) + " predictions were accurate, and " + str(numIncorrect) + " were incorrect.")
    print("Accuracy: " + str(acc) + "; Precision: " + str(pre) + "; Recall: " + str(rec))

# Use training data to calculate a
def calculateClassifier(data):
    valsY = {key: [] for item in data for key in item}
    valsN = {key: [] for item in data for key in item}
    meanY = {key: 0 for item in data for key in item}
    meanN = {key: 0 for item in data for key in item}
    sdY = {key: 0 for item in data for key in item}
    sdN = {key: 0 for item in data for key in item}
    cNoneY = {key: 0 for item in data for key in item}
    cNoneN = {key: 0 for item in data for key in item}

    cFlag = 0
    cNoFlag = 0

    for row in data:
        # Get the flag
        flag = bool(int(row["at_risk_flag"]))

        # Sum values for division later.
        if not flag:
            cNoFlag += 1
            for k, v in row.items():
                if v is not None and k!="at_risk_flag":
                    meanN[k] += v
                    valsN[k].append(v)
                    cNoneN[k] += 1
        else:
            cFlag += 1
            for k, v in row.items():
                if v is not None and k!="at_risk_flag":
                    meanY[k] += v
                    valsY[k].append(v)
                    cNoneY[k] += 1

    # Divide for averages of each tallied value.
    for k, v in meanY.items():
        if k!="at_risk_flag":
            meanY[k] = v / cNoneY[k] if cNoneY[k]>0 else 0
    for k, v in meanN.items():
        if k!="at_risk_flag":
            meanN[k] = v / cNoneN[k] if cNoneN[k]>0 else 0

    # Calculate SD
    for k, sd in sdY.items():
        mean = meanY.get(k)
        for v in valsY.get(k): sd += (v-mean) ** 2
        sd = sd / cNoneY[k] if cNoneY[k]>0 else 0
        sdY[k] = sd ** 0.5

    for k, sd in sdN.items():
        mean = meanN.get(k)
        for v in valsN.get(k): sd += (v-mean) ** 2
        sd = sd / cNoneN[k] if cNoneN[k]>0 else 0
        sdN[k] = sd ** 0.5

    fractionFlag = cFlag / (cFlag + cNoFlag)
    fractionNoFlag = cNoFlag / (cFlag + cNoFlag)

    return (
        meanY, meanN, sdY, sdN, fractionFlag, fractionNoFlag
    )