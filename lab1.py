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
from sklearn.model_selection import train_test_split

import pandas as pd

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
    def train(self, xVals, yVals, epochs = 30, minibatches = True, mbs = 100):
        #TODO: Implement backprop. allow minibatches. mbs should specify the size of each minibatch.

        for epoch in range(epochs):
            #make prediction on inputs
            start = 0
            print(epoch)
        
            
            if minibatches:
            
                w2_new = self.W2[:]
                w1_new = self.W1[:]
                for batch, batch2 in (self.__doubleBatchGenerator(xVals, mbs, yVals)):
                    #model_output = self.predict(batch)
                    layer1, layer2 = self.__forward(batch)
                    #loss = self.mse_loss(layer2, batch2)

                    #The partial derivative of the error with respect to each of the weights leading into the output layer
                    #includes the partial derivative of the loss function wrt to each of the output neurons, multiplied by the
                    #partial derivative of the sigmoid function with respect to the net input, multiplied by the partial derivative of the
                    #net input w.r.t each of the weights heading into the output layer. The next variable contains the numpy array containing
                    #the activation function derivatives for all 10 output neurons for each of the 100 points in the batch. 
                    derivative = self.loss_derivative(layer2, batch2) * self.__sigmoidDerivative(layer2)

                    #print("derivative shape : " + str(derivative.shape))
                    #This 2d array stores the weight coefficients of weights from hidden to output layer
                    back_layer1 = layer1[0][:]
                
                    #This computes a matrix of the partial derivatives of the error with respect to each
                    #of the hidden layer weights. Taking the dot product accumulates the derivatives for all
                    #points in the batch. We then multiply that by the learning rate which gives us our 
                    #gradients for each of the weights.
                    term = np.dot(derivative.T, layer1) * self.lr
                    #print("term shape : " + str(term.shape))
                    
                    #We then update the weight vector
                    w2_new = np.add(self.W2.T, term)
                    w2_new = w2_new.T
                    

                    #The partial derivative of the net input to the output neurons with respect
                    #to the outputs from the hidden layers are the initial values of self.W2.
                    #For each hidden layer neuron, we need to total up the partial derivatives for each output neuron.
                    back_layer1 = np.dot(self.W2, derivative.T)
                    
                    #We then multiply this by the output function derivative with respect to each of the weights
                    #and add it up for all of the points. 
                    back_layer1 = (back_layer1.T * self.__sigmoidDerivative(layer1)).T
                    #print("layer 1 sigmoid derivative shape : " + str(self.__sigmoidDerivative(layer1).shape))
                    #We then multiply all this derivatives by the learning rate
                    term2 = np.dot(back_layer1, batch) * self.lr
                    #print("term2 shape : " + str(term2.shape))
                    
                    #We update the weight vector for self.W1
                    w1_new = np.add(self.W1.T, term2)
                    w1_new = w1_new.T


                    self.W1 = w1_new[:]
                    self.W2 = w2_new[:]
            else :
                w2_new = self.W2[:]
                w1_new = self.W1[:]
                #model_output = self.predict(xVals)
                layer1, layer2 = self.__forward(xVals)
                loss = self.mse_loss(layer2, yVals)
                derivative = self.loss_derivative(layer2, yVals) * self.__sigmoidDerivative(layer2)
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
        return one_hot_encode(layer2)
    
    #Custom function
    def mse_loss(self, preds, labels):
        vector = np.subtract(preds, labels)
        vector = np.square(vector)
        #squared_sum = np.sum(vector)
        loss = np.divide(vector, len(labels))
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
    model = keras.Sequential([keras.layers.Flatten(), keras.layers.Dense(512, activation=tf.nn.relu),
     tf.keras.layers.Dense(10, activation=tf.nn.softmax)])
    #model = keras.Sequential([keras.layers.Dense(512, activation=tf.nn.relu),
     #tf.keras.layers.Dense(10, activation=tf.nn.softmax)])
    opt = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def trainANN(model, x, y, eps = 5):
    model.fit(x, y, epochs = eps)

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
        #network = NeuralNetwork_2Layer(3, 3, 512, 0.01)
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



def evalResults(data, preds, confusion_matrix_size):   #TODO: Add F1 score confusion matrix here.
    xTest, yTest = data
    acc = 0
    for i in range(preds.shape[0]):
        if np.array_equal(preds[i], yTest[i]):   acc = acc + 1
    accuracy = acc / preds.shape[0]
    print("Classifier algorithm: %s" % ALGORITHM)
    print("Classifier accuracy: %f%%" % (accuracy * 100))
    print_confusion_matrix(preds, yTest, confusion_matrix_size)
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

def convert_to_tensor(preds, length):
    result = np.zeros((len(preds), length))
    for i, num in enumerate(preds):
        result[i][num] = 1.0

    return result


def print_confusion_matrix(preds, actual, last):
    result = confusion_matrix(actual.argmax(axis=1), preds.argmax(axis=1))
    for i, row in enumerate(result):
        false_pos = 0
        false_neg = 0
        
        true_pos = result[i][i]

        for j in range(0, last):
            if i == j:
                continue
            false_neg = false_neg + result[i][j]
            false_pos = false_pos + result[j][i]

        precision = true_pos / (false_pos + true_pos)
        recall = true_pos / (false_neg + true_pos)

        #print("false_pos : " + str(false_pos))
        #print("false_neg : " + str(false_neg))

        f1_score = (2 * precision * recall) / (precision + recall)
        print(" class", i, "f1 score : ", str(f1_score))

    print(result)
    """
    names = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    ax= plt.subplot()
    sns.heatmap(result, annot=True, ax = ax);
    ax.xaxis.set_ticklabels(names)
    ax.yaxis.set_ticklabels(names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()
    """

def get_iris_dataset():
    iris = pd.read_csv("https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv")
    print(iris.head())
    print(iris.dtypes)
    X_train, X_test, y_train, y_test = train_test_split(iris.iloc[:, 1:4], iris["virginica"], test_size=0.2)
    return (X_train, convert_to_tensor(y_train, 3)), (X_test, convert_to_tensor(y_test, 3))


#Extra credit for testing neural net on Iris Dataset
def train_and_test_on_iris():
    data = get_iris_dataset()
    network = NeuralNetwork_2Layer(3, 3, 512, 0.01)
    network.train(data[0][0], data[0][1], minibatches=False)
    preds = network.predict(data[1][0])
    evalResults(data[1], preds, 3)
#=========================<Main>================================================

def main():
    raw = getRawData()
    data = preprocessData(raw)
    model = trainModel(data[0])
    preds = runModel(data[1][0], model)
    evalResults(data[1], preds, 10)



if __name__ == '__main__':
    main()
