import numpy as np
import numpy.random as random
import time
import regex
import typing as typ
import os


activationFunctionType = typ.Union[str, typ.Tuple[typ.Callable[[float], float], typ.Callable[[float], float]]]

class NeuralNet:
    def __init__(self, heights: list[int], activationFunction: activationFunctionType, outputActivationFunction: activationFunctionType, initialization: tuple[str, tuple]):
        self.heights = heights
        self.depth = len(self.heights)
        self.gaps = self.depth - 1

        self.aliases = {
            "soft step": "sigmoid",
            "logistic": "sigmoid",
            "hyperbolic tangent": "tanh",
            "leaky rectified linear unit": "LReLu",
            "leaky ReLu": "LReLu",
            "SiL": "SiLu",
            "swish-1": "SiLu",
            "sigmoid shrinkage": "SiLu",
            "sigmoid linear unit": "SiLu",
        }
        self.knownActivations = {
            "linear": lambda x: x,
            "sigmoid": lambda x: np.divide(1, (1 + np.exp(-x))),
            "ReLu": lambda x: np.where(x > 0, x, 0),
            "tanh": lambda x: np.divide(np.exp(x) - np.exp(-x), np.exp(x) + np.exp(-x)),
            "softplus": lambda x: np.log(1 + np.exp(x)),
            "LReLu": lambda x: np.where(x<=0, np.multiply(0.01, x), x),
            "SiLu": lambda x: np.divide(x, 1 + np.exp(-x)),
            "gaussian": lambda x: np.exp(-np.square(x)),
            "softmax": lambda x: np.exp(x)/np.sum(np.exp(x))
        }
        self.knownDerivatives = {
            "linear": lambda x: 1,
            "sigmoid": lambda x: np.divide(np.exp(x), np.square(1 + np.exp(x))),
            "ReLu": lambda x: np.where(x <= 0, 0, 1),
            "tanh": lambda x: np.divide(4, np.exp(2*x) + 2 + np.exp(-2*x)),
            "softplus": lambda x: np.divide(1, 1 + np.exp(-x)),
            "LReLu": lambda x: np.where(x <= 0, 0.01, 1),
            "SiLu": lambda x: np.divide(1 + np.exp(-x) + np.multiply(x, np.exp(-x)), np.square(1 + np.exp(-x))),
            "gaussian": lambda x: -2*x*np.exp(-np.square(x)),
            "softmax": lambda x: (np.sum(np.exp(x))*np.exp(x) - np.exp(2*x))/np.power(np.sum(np.exp(x)), 2)
        }
        self.activationFunction = self.interpretActivationInput(activationFunction)
        self.outputActivationFunction = self.interpretActivationInput(outputActivationFunction)

        self.weights = None
        self.biases = None    
        self.knownWeightInitializations = {
            "zero": {
                "function": lambda layerNum, inputs, outputs, params: np.zeros((outputs, inputs)),
                "parameter expectations": ()
            },
            "uniform": {
                "function": lambda layerNum, inputs, outputs, params: random.uniform(params[0], params[1], (outputs, inputs)),
                "parameter expectations": ("minumum", "maximum")
            },
            "gaussian": {
                "function": lambda layerNum, inputs, outputs, params: random.normal(0, params[0], (outputs, inputs)),
                "parameter expectations": ("standard deviation")
            },
            "uniform xavier": {
                "function": lambda layerNum, inputs, outputs, params: random.uniform(-np.sqrt(6/(inputs + outputs)), np.sqrt(6/(inputs + outputs)), (outputs, inputs)),
                "parameter expectations": ()
            },
            "normal xavier":{
                "function": lambda layerNum, inputs, outputs, params: random.normal(0, np.sqrt(2/(inputs + outputs)), (outputs, inputs)),
                "parameter expectations": ()
            },
            "uniform he": {
                "function": lambda layerNum, inputs, outputs, params: random.uniform(-np.sqrt(6/inputs), np.sqrt(6/outputs), (outputs, inputs)),
                "parameter expectations": ()
            },
            "normal he": {
                "function": lambda layerNum, inputs, outputs, params: random.normal(0, np.sqrt(2/inputs), (outputs, inputs)),
                "parameter expectations": ()
            },
            "lecun": {
                "function": lambda layerNum, inputs, outputs, params: random.normal(0, 1/inputs, (outputs, inputs)),
                "parameter expectations": ()
            },
            "manual": {
                "function": lambda layerNum, inputs, outputs, params: params[layerNum],
                "parameter expectations": tuple(["weightMatrix " + str(i) for i in range(1, self.gaps + 1)])
            }
        }
        self.interpretinitialization(initialization)
   
        self.activations = []
        self.ZValues = []
        self.ZValueDerivatives = []

        for height in self.heights:
            self.activations.append(np.zeros(height))
        for height in self.heights[1:]:
            self.ZValues.append(np.zeros(height))
            self.ZValueDerivatives.append(np.zeros(height))

        self.parentDirectoryPath = os.path.split(__file__)[0]
        self.newestWeightLogPtr = None
        self.newestBiasLogPtr = None
        self.costOutputObj = None

    #Idk
    def checkValidFilePath(self, path) -> bool:
        return True
        return os.path.isdir(os.path.dirname(path))
    def checkFilePathIsTxt(self, path) -> bool:
        return True
        return (os.path.splitext(path) == '.txt')

    #Interpreting parameters
    def interpretActivationInput(self, input):
        if isinstance(input, tuple):
            if len(input) != 2:
                raise ValueError(f'{self} passed {input} for custom activation function intialization (either output or normal), but tuple was not of length 2; The first index should be the activation function and the second should be it\'s derivative')
            if (not callable(input[0])) or (not callable(input[1])):
                raise ValueError(f'{self} was initialized with custom activation functions, but {input} contains non-functions. It should be two lambda objects in a tuple, one activation function, the other it\'s derivative.')          
            return input
        
        if type(input) != type(''):
            raise ValueError("Neural Net was passed >" + str(input) + "< instead of a tuple of functions or string as one of the activation functions")

        for alias, name in self.aliases.items():
            if alias.lower() == input.lower():
                input = name
        for knownFunctionName, knownFunction in self.knownActivations.items():
            if input.lower() == knownFunctionName.lower():
                return (knownFunction, self.knownDerivatives[knownFunctionName])
            
        raise ValueError(f"Neural Net was passed {input}, which is not a recognized activation function. Choose from {list(self.knownActivations)}")
    def interpretinitialization(self, input):
        if input == None:
            return
        self.biases = self.createBiases()
        if not isinstance(input, tuple):
            raise ValueError(f"Initialization parameter of {self} must be a two element tuple in the form ([initilization type], [parameters]) instead of \'{input}\'.")
        if input[0].lower() not in self.knownWeightInitializations:
            raise ValueError(f"Initialization type (the first element of the initialization tuple) passed to {self} was {input[0]}. Instead choose from {list(self.knownWeightInitializations.keys())}.")
        if np.shape(input[1]) != np.shape(self.knownWeightInitializations[input[0].lower()]['parameter expectations']):
            raise ValueError(f"Parameters passed to {self} (the second element of the initilization parameter of Neural Net creation) was {input[1]} when the form {self.knownWeightInitializations[input[0]]['parameter expectations']} was expected")
        self.weights = self.createWeights(self.knownWeightInitializations[input[0].lower()]['function'], input[1]) 

    #Initialization
    def createBiases(self) -> list[np.ndarray]:
        biases = []
        for layer, height in enumerate(self.heights[1:]):
            biases.append(np.zeros(height))
        return biases
    def createWeights(self, randFunc: callable, params: tuple) -> list[np.ndarray]:
        weights = []
        for layer, height in enumerate(self.heights[:-1]):
            weights.append(randFunc(layer, height, self.heights[layer + 1], params))
        return weights
    def loadParametersFromFile(self, parametersFilePath: str) -> None:
        with open(parameters, 'r').read() as parameters:
            pass

    #Running the net and getting output
    def forwardpropagate(self) -> np.ndarray:
        for layerNum in range(1, self.depth - 1): #Forward propagates through pulling a transformed version of the previous layer
            self.ZValues[layerNum - 1] = (self.weights[layerNum - 1] @ self.activations[layerNum - 1]) + self.biases[layerNum - 1] 
            self.activations[layerNum] = self.activationFunction[0](self.ZValues[layerNum - 1])
        self.ZValues[-1] = (self.weights[-1] @ self.activations[-2]) + self.biases[-1]
        self.activations[-1] = self.outputActivationFunction[0](self.ZValues[-1])
    def setInputs(self, vector: np.ndarray) -> None:
        self.activations[0] = vector
    def getOutput(self) -> np.ndarray:
        return self.activations[-1]
    def runNet(self, input: np.ndarray) -> np.ndarray:
       self.setInputs(input) 
       self.forwardpropagate()
       return self.getOutput()
    def sortByConfidence(self, output: np.ndarray) -> np.ndarray:
        return np.argsort(output)
 
    #Back Propagation
    def findRMSE(self, desired: np.ndarray) -> float:
        #Eric noises
        return np.sqrt(np.mean(np.square(self.getOutput() - desired)))
    def findError(self, desired: np.ndarray) -> float:
        return np.mean(np.square(self.getOutput() - desired))/2
    def backpropagateToOutputLayerZValues(self, desired: np.ndarray) -> None:
        self.ZValueDerivatives[-1] = self.outputActivationFunction[1](self.ZValues[-1]) * np.divide(self.activations[-1] - desired, self.heights[-1])
    def ZValueBackpropagation(self, desired: np.ndarray) -> None:
        self.backpropagateToOutputLayerZValues(desired)
        for layerNum in range(self.gaps - 1, 0, -1): #Back propagates by pushing from the layer in the front to the back
            self.ZValueDerivatives[layerNum - 1] = (np.transpose(self.weights[layerNum]) @ self.ZValueDerivatives[layerNum])*self.activationFunction[1](self.ZValues[layerNum - 1])
    def createWeightGradient(self) -> list[np.ndarray]:
        gradient = []
        for layerNum in range(0, self.gaps):
            gradient.append(np.outer(self.ZValueDerivatives[layerNum], self.activations[layerNum]))
        return gradient
    def trainOnExample(self, input: np.ndarray, expectedOutput: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray], float]:
        self.runNet(input)
        self.ZValueBackpropagation(expectedOutput)
        return (self.createWeightGradient(), self.ZValueDerivatives, self.findError(expectedOutput))      

    #Utilities for training
    def batchSplit(self, data: list[typ.Tuple[np.ndarray, np.ndarray]], batchSize: int) -> list[tuple]:
        return [data[batchNum*batchSize:(batchNum + 1)*batchSize] for batchNum in range(np.ceil(len(data)/batchSize).astype(int))]
    def addGradient(self, gradient1: list[np.ndarray], gradient2: list[np.ndarray]):
        return [tensor1 + tensor2 for tensor1, tensor2 in zip(gradient1, gradient2)]
    def applyGradient(self, tensors, gradients) -> None:
        return [tensor - gradient for tensor, gradient in zip(tensors, gradients)]
    
    #Logging neural net state
    def initializeLogDirectory(self) -> None:
        if not os.path.isdir(os.path.join(__file__, '..\\logs')):
            os.makedirs(os.path.join(__file__, '..\\logs'))
        
        weightDirPath = os.path.join(__file__, '..\\logs\\weights')
        biasDirPath = os.path.join(__file__, '..\\logs\\biases')
                        
        if not os.path.isdir(weightDirPath):
            os.makedirs(weightDirPath)
        
        if not os.path.isdir(biasDirPath):
            os.makedirs(biasDirPath)

        self.newestWeightLogPtr = len(os.listdir(weightDirPath))
        self.newestBiasLogPtr = len(os.listdir(biasDirPath))

        costsLogPath = os.path.join(__file__, '..\\logs\\costs.txt')
        if not os.path.exists(costsLogPath):
            with open(costsLogPath, 'w'):
                pass
        self.costOutputObj = open(costsLogPath, 'a')

    def logTensors(self, tensors, path):
        for tensorNum, tensor in enumerate(tensors):
            np.savetxt(os.path.join(path, str(tensorNum)), tensor, fmt='%f', delimiter=',', newline='\n')
    def logWeights(self):
        newWeightsFolder = os.path.join(os.path.join(__file__, '..\\logs\\weights\\'), str(self.newestWeightLogPtr))
        os.makedirs(newWeightsFolder)
        self.logTensors(self.weights, newWeightsFolder)
        self.newestWeightLogPtr += 1
    def logBiases(self):
        newBiasesFolder = os.path.join(os.path.join(__file__, '..\\logs\\biases\\'), str(self.newestBiasLogPtr))
        os.makedirs(newBiasesFolder)
        self.logTensors(self.biases, newBiasesFolder)
        self.newestBiasLogPtr += 1 
    def logCost(self, cost):
        self.costOutputObj.write(cost + '\n')
    
    #Training and testing
    def miniBatchTrain(self, data: list[tuple[np.ndarray]], batchSize: int, learningRate: float, printUpdates: bool, saveLogs: bool, repeats: int):
        if saveLogs:
            self.initializeLogDirectory()

        if repeats < 0:
            raise ValueError(f'{self} miniBatchTrain was passed \'{repeats}\' as repeats instead of positive integer (including 0).')

        batchNum = 1
        totalBatches = np.ceil((len(data)*(repeats + 1))/batchSize)
        batchTimes = []

        for run in range(repeats + 1):
            for batch in self.batchSplit(data, batchSize):
                batchStartTime = time.time()
                
                batchCostTotal = 0
                
                weightGradientSum = self.createWeights(lambda layerNum, inputs, outputs, params: np.zeros((outputs, inputs)), ())
                biasGradientSum = self.createBiases()
                
                for example in batch:
                    trainingOutput = self.trainOnExample(example[0], example[1])
                    weightGradientSum = self.addGradient(weightGradientSum, trainingOutput[0])
                    biasGradientSum = self.addGradient(biasGradientSum, trainingOutput[1])
                    batchCostTotal += trainingOutput[2]
                
                averageWeightGradient = [learningRate*(matrix/batchSize) for matrix in weightGradientSum] #Turning the sums into averages
                averageBiasGradient = [learningRate*(vector/batchSize) for vector in biasGradientSum]
                averageCost = batchCostTotal/batchSize
 
                batchTimes.append(time.time() - batchStartTime)
                completionTime = (sum(batchTimes)/len(batchTimes))*(totalBatches-batchNum)

                if printUpdates:
                    print(f"Batch #{batchNum} completed.\nCost: {averageCost}\nEst. time of completion: {np.floor(completionTime/3600)} Hr {np.ceil(np.mod(completionTime, 360)/60)} min")
                
                self.weights = self.applyGradient(self.weights, averageWeightGradient)
                self.biases = self.applyGradient(self.biases, averageBiasGradient)

                batchNum += 1
            self.logBiases()
            self.logWeights()
            self.logCost(str(averageCost))
            
    def testOnData(self, data: list[tuple[np.ndarray, any]], evaluationFunction: typ.Callable[[np.ndarray, any], typ.Union[bool, float]], evaluationsOutput: typ.Union[str, None]) -> float:
        evaluations = []
        for testExample in data:
            evaluations.append(evaluationFunction(self.runNet(testExample[0]), testExample[1]))

        if evaluationFunction != None:
            if not (self.checkValidFilePath(evaluationsOutput) and self.checkFilePathIsTxt(evaluationsOutput)):
                raise ValueError[f'{self} testOnData was passed {evaluationsOutput} for evaluationsOutput instead of the file path for/of an .txt file']
            
        with open(evaluationsOutput, 'a') as output:
            output.write(str(evaluations))
        evaluations = np.array(evaluations) #Freezes the evlauation array
        
        if evaluations.dtype == bool:
            return np.sum(evaluations)/np.size(evaluations)
        else:
            return np.mean(evaluations)



