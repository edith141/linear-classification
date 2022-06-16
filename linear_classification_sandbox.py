from math import exp
from random import shuffle
# prepare dataset
def getDataSet():
    # col1 = []
    # col2 = []
    # col3 = []
    dataset = []
    with open("examples/earthquake-clean.data.txt", 'r') as f:
        for line in f:
            first, sec, thrd = line.split(",")
            # col1.append(first)
            # col2.append(sec)
            # col3.append(thrd)
            dataset.append([float(first), float(sec), float(thrd)])
    shuffle(dataset)
    return dataset
    # return [[1,1,1],[1,0,1],[0,0,0],[0,1,1],[1,1,1],[1,0,1],[0,0,0],[0,1,1],[1,1,1],[1,0,1],[0,0,0],[0,1,1]]
 
#  spilt data set into train, test
def trainTestSplit2(dataset, ratio):
    elems = len(dataset)
    middle = int(elems * ratio)
    print(f"\n\n\nDataSet Split:")
    print(f"\nTrain Data: \n{dataset[:middle]}")
    print(f"\nTest Data: \n{dataset[middle:]}")
    print("\n")
    return dataset[:middle], dataset[middle:]

def getAccuracy(act, pred):
    corr = 0
    for i in range(len(act)):
        if act[i] == pred[i]:
            corr+=1
    return (corr / float(len(act)) * 100)

# func Predict -> Make a prediction of y (yHat) with the given coefficients
# yHat = 1.0 / (1.0 + e^(-(b0 + b1 * x1))) 
# b0 -> bias/intercept
# b1 -> coeff. for current (single) value of x
def predict(row, coefficients): 
	yhat = coefficients[0]
	for i in range(len(row)-1):
		yhat += coefficients[i + 1] * row[i]
    # the sigmoid(x) function -> 1 / (1 + e^(x-1))
	return 1.0 / (1.0 + exp(-yhat))
 
# Estimate logistic regression coefficients using stochastic gradient descent
def coefficients_sgd(train, l_rate, n_epoch):
	coef = [0.0 for i in range(len(train[0]))]
	for epoch in range(n_epoch):
		sum_error = 0
		for row in train:
			yhat = predict(row, coef)
			error = row[-1] - yhat
            # error sum squared for logging
			sum_error += error**2
            # coeff one updating here - the y-intercept or the bias
            # this is in the outer loop as this doesn't depend on individual values of x input.
			coef[0] = coef[0] + l_rate * error * yhat * (1.0 - yhat)
			for i in range(len(row)-1):
                # this is the weight coeff. to the input x and is hence dependent on each value of x.
                # so it is updated in each iteration..
				coef[i + 1] = (coef[i + 1] + l_rate * error * yhat * (1.0 - yhat) * row[i])
		print('>epoch=%d, lrate=%.3f, error=%.3f, b=%.3f, w=%.3f' % (epoch, l_rate, sum_error, coef[0], coef[i+1]))
	return coef


def logLinearReg(train, test, LR, epochs):
    preds = list()
    coef = coefficients_sgd(train, LR, epochs)
    for row in test:
        yHat = predict(row, coef)
        yHat = round(yHat)
        preds.append(yHat)
    return preds


def evalAlgo(dataset, algo, *args):
    trainDataSet, testDataSet = trainTestSplit2(getDataSet(), 0.7)
    predictedVals = algo(trainDataSet, testDataSet, *args)
    actualVals = [int(row[-1]) for row in testDataSet]
    print(f"\n\n\nActual Y Values: \n{actualVals}")
    print(f"\nPredicted Y Values: \n{predictedVals}")
    accuracy = getAccuracy(actualVals, predictedVals)

    return accuracy


# Calculate coefficients
# dataset = getDataSet()
# l_rate = 0.3
# n_epoch = 40000
# coef = coefficients_sgd(dataset, l_rate, n_epoch)
# print(coef)
def get_avg_from_arr(arr):
    sum = 0
    N = len(arr)
    for elem in arr:
        sum+=elem
    return (sum/N)

LR = 0.17
epochs = 8000
# scores = evalAlgo(getDataSet(), logLinearReg, LR, epochs)
print("\n\n\n")
# print(f"Final Accuracy: {scores}")

runs = []


for run in range(5):
    score = evalAlgo(getDataSet(), logLinearReg, LR, epochs)
    runs.append(score)
    
print(f'''\n\n
            Total runs to get an avg accuracy of the model: 5\n
            Accuracy array: {runs}\n
            Avg accuracy: {get_avg_from_arr(runs)}%\n\n''')