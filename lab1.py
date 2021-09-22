# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 23:35:34 2021

@author: rohan
"""


import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
import random
import math
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


# Setting random seeds to keep everything deterministic.
random.seed(1618)
np.random.seed(1618)
#tf.set_random_seed(1618)   # Uncomment for TF1.
tf.random.set_seed(1618)

# Disable some troublesome logging.
#tf.logging.set_verbosity(tf.logging.ERROR)   # Uncomment for TF1.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Information on dataset.
NUM_CLASSES = 10
IMAGE_SIZE = 784

# Use these to set the algorithm to use.
#ALGORITHM = "guesser"
ALGORITHM = "custom_net"
#ALGORITHM = "tf_net"





class NeuralNetwork_2Layer():
    def __init__(self, inputSize, outputSize, neuronsPerLayer, learningRate = 0.1):
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.neuronsPerLayer = neuronsPerLayer
        self.lr = learningRate
        self.W1 = np.random.randn(self.inputSize, self.neuronsPerLayer)
        self.W2 = np.random.randn(self.neuronsPerLayer, self.outputSize)

    # Activation function.
    def __sigmoid(self, x):
        #TODO: implement
        denom = np.add(1, np.exp(-x))
        result = np.divide(1, denom)
        return result

    # Activation prime function.
    def __sigmoidDerivative(self, x):
        #TODO: implement
    
        x1 = self.__sigmoid(x)
        x2 = np.subtract(1, self.__sigmoid(x))
        result = x1 * x2 
        return result

    # Batch generator for mini-batches. Not randomized.
    def __batchGenerator(self, l, n):
        for i in range(0, len(l), n):
            yield l[i : i + n]
    
    # Custom batch Generator
    def __doubleBatchGenerator(self, l, n, o):
        for i in range(0, len(l), n):
            yield l[i : i + n], o[i : i + n]

    # Training with backpropagation.
    def train(self, xVals, yVals, epochs = 30, minibatches = False, mbs = 100):
        #TODO: Implement backprop. allow minibatches. mbs should specify the size of each minibatch.

        for epoch in range(epochs):
            #make prediction on inputs
            start = 0
            print(epoch)
        
            
            if minibatches:
            
                w2_new = self.W2[:]
                w1_new = self.W1[:]
                for batch, batch2 in (self.__doubleBatchGenerator(xVals, mbs, yVals)):
                    model_output = self.predict(batch)
                    layer1, layer2 = self.__forward(batch)
                    loss = self.mse_loss(model_output, batch2)
                    derivative = self.loss_derivative(model_output, batch2) * self.__sigmoidDerivative(layer2)
                    back_layer1 = layer1[0][:]
                

                    term = np.dot(derivative.T, layer1) * self.lr
                    
                    w2_new = np.add(self.W2.T, term)
                    w2_new = w2_new.T
                    

                    back_layer1 = np.dot(self.W2, derivative.T)
                    
                    back_layer1 = (back_layer1.T * self.__sigmoidDerivative(layer1)).T
                    term2 = np.dot(back_layer1, batch) * self.lr
                    
                    w1_new = np.add(self.W1.T, term2)
                    w1_new = w1_new.T


                    self.W1 = w1_new[:]
                    self.W2 = w2_new[:]
            else :
                w2_new = self.W2[:]
                w1_new = self.W1[:]
                model_output = self.predict(xVals)
                layer1, layer2 = self.__forward(xVals)
                loss = self.mse_loss(layer2, yVals)
                derivative = self.loss_derivative(model_output, yVals) * self.__sigmoidDerivative(layer2)
                back_layer1 = layer1[0][:]

                
                term = np.dot(derivative.T, layer1) * self.lr
                w2_new = np.add(self.W2.T, term)
                w2_new = w2_new.T
                
    

                back_layer1 = np.dot(self.W2, derivative.T)
                back_layer1 = (back_layer1.T * self.__sigmoidDerivative(layer1)).T
                term2 = np.dot(back_layer1, xVals) * self.lr
                w1_new = np.add(self.W1.T, term2)
                w1_new = w1_new.T

                #print("w2_new shape : " + str(w2_new.shape))
                #print("w1_new shape : " + str(w1_new.shape))

                self.W1 = w1_new[:]
                self.W2 = w2_new[:]

            
            
            

    # Forward pass.
    def __forward(self, input):
        layer1 = self.__sigmoid(np.dot(input, self.W1))
        layer2 = self.__sigmoid(np.dot(layer1, self.W2))
        return layer1, layer2

    # Predict.
    def predict(self, xVals):
        _, layer2 = self.__forward(xVals)
        return layer2
    
    #Custom function
    def mse_loss(self, preds, labels):
        vector = np.subtract(preds, labels)
        vector = np.square(vector)
        squared_sum = np.sum(vector)
        loss = squared_sum / len(labels)
        return loss
    
    def loss_derivative(self, preds, labels):
        try:
            return (2/len(preds)) * (np.subtract(labels, preds))
        except:
            return (2) * (np.subtract(labels, preds))


# Classifier that just guesses the class label.
def guesserClassifier(xTest):
    ans = []
    for entry in xTest:
        pred = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        pred[random.randint(0, 9)] = 1
        ans.append(pred)
    return np.array(ans)


#==<TF_NET code>==#

def buildANN():
    model = keras.Sequential([keras.layers.Flatten(), keras.layers.Dense(512, activation=tf.nn.tanh),
     tf.keras.layers.Dense(10, activation=tf.nn.softmax)])
    #model = keras.Sequential([keras.layers.Dense(512, activation=tf.nn.relu),
     #tf.keras.layers.Dense(10, activation=tf.nn.softmax)])
    opt = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def trainANN(model, x, y, eps = 5):
    model.fit(x, y, epochs = eps, validation_split=0.2)

def runANN(model, x):
    preds = model.predict(x)
    return one_hot_encode(preds)


#=========================<Pipeline Functions>==================================

def getRawData():
    mnist = tf.keras.datasets.mnist
    (xTrain, yTrain), (xTest, yTest) = mnist.load_data()
    print("Shape of xTrain dataset: %s." % str(xTrain.shape))
    print("Shape of yTrain dataset: %s." % str(yTrain.shape))
    print("Shape of xTest dataset: %s." % str(xTest.shape))
    print("Shape of yTest dataset: %s." % str(yTest.shape))
    return ((xTrain, yTrain), (xTest, yTest))



def preprocessData(raw):
    ((xTrain, yTrain), (xTest, yTest)) = raw            #TODO: Add range reduction here (0-255 ==> 0.0-1.0).
    
    xTrain = np.divide(xTrain, 255)
    xTest = np.divide(xTest, 255)
    
    xTrain = np.reshape(xTrain, (len(xTrain), 784))
    xTest = np.reshape(xTest, (len(xTest), 784))

    yTrainP = to_categorical(yTrain, NUM_CLASSES)
    yTestP = to_categorical(yTest, NUM_CLASSES)
    print("New shape of xTrain dataset: %s." % str(xTrain.shape))
    print("New shape of xTest dataset: %s." % str(xTest.shape))
    print("New shape of yTrain dataset: %s." % str(yTrainP.shape))
    print("New shape of yTest dataset: %s." % str(yTestP.shape))
    return ((xTrain, yTrainP), (xTest, yTestP))



def trainModel(data):
    xTrain, yTrain = data
    if ALGORITHM == "guesser":
        return None   # Guesser has no model, as it is just guessing.
    elif ALGORITHM == "custom_net":
        print("Building and training Custom_NN.")
        print("Not yet implemented.")                   #TODO: Write code to build and train your custon neural net.
        network = NeuralNetwork_2Layer(784, 10, 512, 0.01)
        network.train(xTrain,  yTrain)
        return network
    elif ALGORITHM == "tf_net":
        print("Building and training TF_NN.")
        print("Not yet implemented.")                   #TODO: Write code to build and train your keras neural net.
        network = buildANN()
        trainANN(network, xTrain, yTrain)
        return network
    else:
        raise ValueError("Algorithm not recognized.")



def runModel(data, model):
    if ALGORITHM == "guesser":
        return guesserClassifier(data)
    elif ALGORITHM == "custom_net":
        print("Testing Custom_NN.")
        print("Not yet implemented.")                   #TODO: Write code to run your custon neural net.
        result = model.predict(data)
        return result
    elif ALGORITHM == "tf_net":
        print("Testing TF_NN.")
        print("Not yet implemented.")                   #TODO: Write code to run your keras neural net.
        result = runANN(model, data)
        return result
    else:
        raise ValueError("Algorithm not recognized.")



def evalResults(data, preds):   #TODO: Add F1 score confusion matrix here.
    xTest, yTest = data
    acc = 0
    for i in range(preds.shape[0]):
        if np.array_equal(preds[i], yTest[i]):   acc = acc + 1
    accuracy = acc / preds.shape[0]
    print("Classifier algorithm: %s" % ALGORITHM)
    print("Classifier accuracy: %f%%" % (accuracy * 100))
    print_confusion_matrix(preds, yTest)
    print()


def one_hot_encode(preds):
    index = np.argmax(preds, 1)
    preds_res = preds[:]
    for i, pred_res in enumerate(preds_res):
        for j, pred in enumerate(pred_res):
            if j == index[i]:
                preds_res[i][j] = 1.0
            else:
                preds_res[i][j] = 0.0

    return preds_res


def print_confusion_matrix(preds, actual):
    result = confusion_matrix(actual.argmax(axis=1), preds.argmax(axis=1))
    print(result)
    names = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    ax= plt.subplot()
    sns.heatmap(result, annot=True, ax = ax);
    ax.xaxis.set_ticklabels(names)
    ax.yaxis.set_ticklabels(names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()


#=========================<Main>================================================

def main():
    raw = getRawData()
    data = preprocessData(raw)
    model = trainModel(data[0])
    preds = runModel(data[1][0], model)
    evalResults(data[1], one_hot_encode(preds))



if __name__ == '__main__':
    main()
