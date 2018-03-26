import numpy as np
import sys

firstWeight = None
secondWeight = None
preNewFirstWeight = None
preNewSecondWeight = None

# Initialize the weights with random numbers in [-1, 1]
def weightInitiailize(inputNum, outputNum):
	weight = np.zeros((inputNum, outputNum))
	for i in range(inputNum):
		for j in range(outputNum):
			weight[i][j] = getRandom()
	return weight

def getRandom():
	return np.random.uniform(-1, 1)

# Initialize the input, the inputs are like [0, 0, 0, 0]
def initailInput():
	input = []
	for i in range(2):
		for j in range(2):
			for k in range(2):
				for l in range(2):
					input.append([i, j, k, l])
	
	return np.reshape(input,(len(input), len(input[0])))

# Return the output for each element in the input array
# The output is correspondingly whether the input array has an odd number of 1s
# For example, if the input is [1, 1, 0 , 1], the output is 1
def getOutput(input):
	row, col = input.shape
	res = np.zeros((row, 1))
	for i in range(row):
		num = 0
		for j in range(col):
			if input[i][j] == 1:
				num += 1
		if num % 2 == 1:
			res[i] = 1

	return res

#Logistic regression
def logisticRegression(x):
	return 1/ (1 + np.exp(-x))

# Derivative for logistic regression
def derivative(x):
	return logisticRegression(x) * (1 - logisticRegression(x))

def matrixFunction(input, func):
	row, col = input.shape
	result = np.zeros((row, col))
	for i in range(row):
		for j in range(col):
			result[i][j] = func(input[i][j])
	return result

#Add bias on the first column
def bias(input):
	row, col = input.shape
	bias = np.zeros((row, 1))
	for i in range(row):
		bias[i][0] = 1
	return np.concatenate((bias, input), axis = 1)

def training(input, output):
	#Add bias, to obtain 1*5 vector
	firstInput = bias(input)

	#5*4, 4 for the hidden number units
	global firstWeight
	# First layer
	firstDot = np.dot(firstInput, firstWeight)
	#1*4
	firstY = matrixFunction(firstDot, logisticRegression)

	global secondWeight
	#1*5
	secondInput = bias(firstY)
	# Second layer
	# 1*5 5*1
	secondDot = np.dot(secondInput, secondWeight)
	secondY = matrixFunction(secondDot, logisticRegression)

	secondDiff = output - secondY

	if abs(secondDiff) < 0.05:
		return

	# back propagation
	# (1, 1)
	secondDelta = matrixFunction(secondDot, derivative) * secondDiff

	newSecondWeight = (learningRate * np.dot(secondDelta, secondInput)).T

	secondWeightNoBias = secondWeight[1:, 0:1]

	firstDiff = np.dot(secondWeightNoBias, secondDelta)

	firstDelta = matrixFunction(firstDot, derivative).T * firstDiff

	newFirstWeight = (learningRate * np.dot(firstDelta, firstInput)).T

	# Used for weights update with momentum
	global preNewFirstWeight
	global preNewSecondWeight

	if preNewFirstWeight is not None and preNewSecondWeight is not None:
		newFirstWeight = momentum * preNewFirstWeight + newFirstWeight
		newSecondWeight = momentum * preNewSecondWeight + newSecondWeight

	preNewFirstWeight = newFirstWeight
	preNewSecondWeight = newSecondWeight
	firstWeight = firstWeight + newFirstWeight
	secondWeight = secondWeight + newSecondWeight

	return

# Test if training is completed
def trainingCompleted(input, desiredOutput):
	inputRow, inputCol = input.shape
	for i in range(inputRow):
		diff = getOutputDiff(np.reshape(input[i], (1, len(input[i]))), np.reshape(desiredOutput[i], (1, len(desiredOutput[i]))))
		
		if abs(diff) > 0.05:
			return False
	return True

def getOutputDiff(input, output):
	firstInput = bias(input)
	firstDot = np.dot(firstInput, firstWeight)
	firstY = matrixFunction(firstDot, logisticRegression)

	secondInput = bias(firstY)
	secondDot = np.dot(secondInput, secondWeight)
	secondY = matrixFunction(secondDot, logisticRegression)

	return abs(output - secondY)


def run (learningRate, momentum):
	global globalLearningRate
	globalLearningRate = learningRate

	global globalMomentum
	globalMomentum = momentum

	iteration = 0

	global firstWeight
	firstWeight = initialFirstWeight

	global secondWeight
	secondWeight = initialSecondWeight
	
	while True:
		completed = False
		for j in range (inputRow):
			curInput = np.reshape(input[j], (1, len(input[j])))
			curOutput = desiredOutput[j]
			training(curInput, curOutput)
			if trainingCompleted(input, desiredOutput):
				completed = True
				break
		iteration += 1
		if iteration % 10000 == 0:
			print "iteration reach", iteration
		if completed:
			break
	print "total iteration= ", iteration

if __name__ == "__main__":
	global momentum
	momentum = 0
	global learningRate 
	learningRate = 0
	if len(sys.argv) != 2 and len(sys.argv) != 3:
		print "Please enter the learning rate and momentum"
		sys.exit(0)
	elif len(sys.argv) == 2:
		learningRate = float(sys.argv[1])
	else:
		learningRate = float(sys.argv[1])
		momentum = float(sys.argv[2])
	
	input = initailInput()
	inputRow, inputCol = input.shape

	desiredOutput = getOutput(input)

	global initialFirstWeight
	initialFirstWeight = weightInitiailize(5, 4)

	global initialSecondWeight
	initialSecondWeight = weightInitiailize(5, 1)

	print "learning rate = ", learningRate, "momentum = ", momentum
	for num in range(1):
		run(learningRate, momentum)



