from .Neuron import Neuron
import numpy as np
import random
import pickle

class Perceptron(object):

    def __init__(self, numberOfNeurons, imageWidth, imageHeight, weights=None):
        """Loads an existing neural network if weights points to a valid 
           weights file, otherwise, initialises a new neural network.
        
        Arguments:
            numberOfNeurons {int} -- The number of neurons in the network
            imageWidth {int} -- The width  of the input images in pixels
            imageHeight {int} -- The height of the input images in pixels
        
        Keyword Arguments:
            weights {string} -- The path to the weights file, or None to 
                                initialize a new network(default: {None})
        """

        if weights is None:
             # TODO Exercise 1 Initalize a new network containing 
             # numberOfNeurons neurons each with a different position chosen
             # at random
           positionList = self.generatePositionList(imageWidth,imageHeight)

           self.network = []
           for i in range(numberOfNeurons):
                x = np.random.random_integers(0, len(positionList)-1)
                self.network.append(
                Neuron([positionList[x][0],
                        positionList[x][1]], True))
                del positionList[x]




        else:
            # TODO Exercise ? Load an existing network
            self.load(weights)


    def __eq__(self, other):
        return (self.network == other.network)
    
    
    def generatePositionList(self, imageWidth, imageHeight):
        """Generate a list of all possible positions in an image of dimensions
            imageWidth x imageHeight
        
        Arguments:
            imageWidth {int} -- The width of the image
            imageHeight {int} -- The height of the image
        
        Returns:
            list((tuple(int,int))) -- The list of positions presented as (x, y)
        """

        positionList = []
        for x in range(imageWidth):
            for y in range(imageHeight):
                positionList.append((x ,y))
        
        return positionList

    # TODO Exercise 2: Implement the forward pass function
    def forwardPass(self, image):
        """Takes a binary image as input, computes the weighted sum of the
            of the products each neuron weight with the associated pixel value 
            and activates the neuron if the pixel value is 1.
            It then returns -1 if the weighted sum is negative, 0 if the 
            weighted sum is 0 and +1 if the weighted sum is positive.

        Arguments:
            image {numpy.array} -- The binary image input
        """

        sum = 0



        network = self.network
        for i in range(len(network)):
            pixel = image[network[i].yPos][network[i].xPos]
            if pixel == 1:
                network[i].active = True
            sum   +=  pixel*network[i].value;


        if sum < 0 :
            return -1

        if sum == 0:
            return 0

        if sum > 0:
            return 1















    # TODO Exercise 3: Implement the backpropagation function
    def backProp(self, expectedResult, result):
        """If the expected result does not match the actual result, add the 
           expected result to the value of all active neurons, then deactivate
           them.

        Arguments:
            expectedResult {int} -- The expected result for the forward pass
            result {int} -- The actual result for the forward pass
        """
        for i in range(len(self.network)):
            if expectedResult != result :
                if self.network[i].active:
                    self.network[i].value += expectedResult
                    self.network[i].active = False
            else:
                self.network[i].active = False






        pass

    # TODO Exercise 4: Implement the error calculation function
    def calcError(self, labels, results):
        """Compute the number of label/result pairs that are not equal and 
           return the result.

        Arguments:
            labels {list(int)} -- The list of labels
            results {list(int)} -- The list of results
        """
        error = 0


        goodResults = 0

        for i in range(len(results)):
            if labels[i] == results[i]:
                goodResults += 1
        error = len(labels) - goodResults

        return error


    # TODO Exercise 5: Implement the training function
    def train(self, images, labels, maxIterations):
        """Train the perceptron by computing the result of a forward pass then 
            by backpropagating that result through the neural network until the
            error reaches 0 or the max iteration number has been reached.

        Arguments:
            images {[type]} -- [description]
            labels {[type]} -- [description]
            numIterations {[type]} -- [description]
        """
        results = []
        for i in range(len(images)):
           results.append(self.forwardPass(images[i]))
           for j in range(0,maxIterations):
               self.backProp(labels[i],results[i])
               if self.calcError(labels,results) == 0:
                   break














            #pass

    # TODO Exercise 6: Implement the testing function
    def test(self, images):
        """Compute the forward pass results for a list of images
        
        Arguments:
            images {numpy.array} -- A list of binary images
        
        Returns:
            list(int) -- The forward pass results for each input image
        """

        return []

    # TODO Bonus: write the save and load functions that will allow you to save
    # the result of a training and reuse those weights later
    def save(self, file):
        """Save the current neural network to a file using 
            pickle
        
        Arguments:
            file {str} -- The path to the file the weights will be saved to
        """

        pass


    def load(self, file):
        """Load the neural network from a pickle file 

        Arguments:
            file {str} -- The path to the file containing the weights
        """

        self.network = []
