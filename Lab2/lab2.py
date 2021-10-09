import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers
from sklearn.metrics import confusion_matrix
import random

from tensorflow.keras.applications.vgg16 import VGG16


random.seed(1618)
np.random.seed(1618)
#tf.set_random_seed(1618)   # Uncomment for TF1.
tf.random.set_seed(1618)

#IH = input height
#IW = input width
#IZ = input depth
#IS = input size = height * width * depth

#tf.logging.set_verbosity(tf.logging.ERROR)   # Uncomment for TF1.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#ALGORITHM = "guesser"
#ALGORITHM = "tf_net"
ALGORITHM = "tf_conv"

#DATASET = "mnist_d"
DATASET = "mnist_f"
#DATASET = "cifar_10"
#DATASET = "cifar_100_f"
#DATASET = "cifar_100_c"

if DATASET == "mnist_d":
    NUM_CLASSES = 10
    IH = 28
    IW = 28
    IZ = 1
    IS = 784
elif DATASET == "mnist_f":
    NUM_CLASSES = 10
    IH = 28
    IW = 28
    IZ = 1
    IS = 784
elif DATASET == "cifar_10":
    #pass                                 # TODO: Add this case.
    NUM_CLASSES = 10
    IH = 32
    IW = 32
    IZ = 3
    IS = 3072

elif DATASET == "cifar_100_f":
    #pass                                 # TODO: Add this case.
    NUM_CLASSES = 100
    IH = 32
    IW = 32
    IZ = 3
    IS = 3072
elif DATASET == "cifar_100_c":
    #pass                                 # TODO: Add this case.
    NUM_CLASSES = 100
    IH = 32
    IW = 32
    IZ = 3
    IS = 3072


#=========================<Classifier Functions>================================

def guesserClassifier(xTest):
    ans = []
    for entry in xTest:
        pred = [0] * NUM_CLASSES
        pred[random.randint(0, 9)] = 1
        ans.append(pred)
    return np.array(ans)


def buildTFNeuralNet(x, y, eps = 5):
    #TODO: Implement a standard ANN here.
    model = keras.Sequential([keras.layers.Flatten(), keras.layers.Dense(512, activation=tf.nn.relu),
     tf.keras.layers.Dense(NUM_CLASSES, activation=tf.nn.softmax)])
    #model = keras.Sequential([keras.layers.Dense(512, activation=tf.nn.relu),
     #tf.keras.layers.Dense(10, activation=tf.nn.softmax)])
    opt = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x, y, epochs = eps)
    return model


def buildTFConvNet(x, y, eps = 15, dropout = True, dropRate = 0.25):
    #TODO: Implement a CNN here. dropout option is required.
    
    #VGG example

    lossType = keras.losses.categorical_crossentropy
    opt = keras.optimizers.Adam(learning_rate=0.001)
    if dropout == True:
        model = keras.Sequential(
            [
                keras.Input(shape=(IW, IH, IZ)),
                layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
                layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
                layers.BatchNormalization(),
                layers.Activation("relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Dropout(dropRate),
                layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
                layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
                layers.BatchNormalization(),
                layers.Activation("relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Dropout(dropRate),
                layers.Flatten(),
                layers.Dense(NUM_CLASSES, activation="softmax"),
            ]
        )
    else:
        model = keras.Sequential(
            [
                keras.Input(shape=(IW, IH, IZ)),
                layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
                layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
                layers.BatchNormalization(),
                layers.Activation("relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
                layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
                layers.BatchNormalization(),
                layers.Activation("relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Flatten(),
                layers.Dense(NUM_CLASSES, activation="softmax"),
            ]
        )

    model.compile(optimizer = opt, loss = lossType, metrics=['accuracy'])
    model.fit(x, y, epochs = eps)

    return model

    

#=========================<Pipeline Functions>==================================

def getRawData():
    if DATASET == "mnist_d":
        mnist = tf.keras.datasets.mnist
        (xTrain, yTrain), (xTest, yTest) = mnist.load_data()
    elif DATASET == "mnist_f":
        mnist = tf.keras.datasets.fashion_mnist
        (xTrain, yTrain), (xTest, yTest) = mnist.load_data()
    elif DATASET == "cifar_10":
        # TODO: Add this case.
        (xTrain, yTrain), (xTest, yTest) = tf.keras.datasets.cifar10.load_data()
    elif DATASET == "cifar_100_f":
        # TODO: Add this case.
        (xTrain, yTrain), (xTest, yTest) = (x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data(
            label_mode='fine'
        )
    elif DATASET == "cifar_100_c":
        # TODO: Add this case.
        (xTrain, yTrain), (xTest, yTest) = (x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data(
            label_mode='coarse'
        )
    else:
        raise ValueError("Dataset not recognized.")
    print("Dataset: %s" % DATASET)
    print("Shape of xTrain dataset: %s." % str(xTrain.shape))
    print("Shape of yTrain dataset: %s." % str(yTrain.shape))
    print("Shape of xTest dataset: %s." % str(xTest.shape))
    print("Shape of yTest dataset: %s." % str(yTest.shape))
    return ((xTrain, yTrain), (xTest, yTest))



def preprocessData(raw):
    ((xTrain, yTrain), (xTest, yTest)) = raw
    if ALGORITHM != "tf_conv":
        xTrainP = xTrain.reshape((xTrain.shape[0], IS))
        xTestP = xTest.reshape((xTest.shape[0], IS))
    else:
        xTrainP = xTrain.reshape((xTrain.shape[0], IH, IW, IZ))
        xTestP = xTest.reshape((xTest.shape[0], IH, IW, IZ))
    yTrainP = to_categorical(yTrain, NUM_CLASSES)
    yTestP = to_categorical(yTest, NUM_CLASSES)
    print("New shape of xTrain dataset: %s." % str(xTrainP.shape))
    print("New shape of xTest dataset: %s." % str(xTestP.shape))
    print("New shape of yTrain dataset: %s." % str(yTrainP.shape))
    print("New shape of yTest dataset: %s." % str(yTestP.shape))
    return ((xTrainP, yTrainP), (xTestP, yTestP))



def trainModel(data):
    xTrain, yTrain = data
    if ALGORITHM == "guesser":
        return None   # Guesser has no model, as it is just guessing.
    elif ALGORITHM == "tf_net":
        print("Building and training TF_NN.")
        return buildTFNeuralNet(xTrain, yTrain)
    elif ALGORITHM == "tf_conv":
        print("Building and training TF_CNN.")
        return buildTFConvNet(xTrain, yTrain)
    else:
        raise ValueError("Algorithm not recognized.")



def runModel(data, model):
    if ALGORITHM == "guesser":
        return guesserClassifier(data)
    elif ALGORITHM == "tf_net":
        print("Testing TF_NN.")
        preds = model.predict(data)
        for i in range(preds.shape[0]):
            oneHot = [0] * NUM_CLASSES
            oneHot[np.argmax(preds[i])] = 1
            preds[i] = oneHot
        return preds
    elif ALGORITHM == "tf_conv":
        print("Testing TF_CNN.")
        preds = model.predict(data)
        for i in range(preds.shape[0]):
            oneHot = [0] * NUM_CLASSES
            oneHot[np.argmax(preds[i])] = 1
            preds[i] = oneHot
        return preds
    else:
        raise ValueError("Algorithm not recognized.")



def evalResults(data, preds):
    xTest, yTest = data
    acc = 0
    for i in range(preds.shape[0]):
        if np.array_equal(preds[i], yTest[i]):   acc = acc + 1
    accuracy = acc / preds.shape[0]
    print("Classifier algorithm: %s" % ALGORITHM)
    print("Classifier accuracy: %f%%" % (accuracy * 100))
    print_confusion_matrix(preds, yTest, 10)
    print()

#code to plot bar graph

import matplotlib.pyplot as plt

def plot_bar_graph(data, vals, filename):
    fig = plt.figure(figsize=(10, 5))
    datasets = data[:]
    values = vals[:]
    plt.bar(datasets, values, width=0.5)
    for i in range(len(datasets)):
        plt.text(i,values[i],values[i])

    plt.savefig(filename)
    plt.clf()

#code to plot confusion matrix
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

#=========================<Main>================================================

def main():
    
    raw = getRawData()
    data = preprocessData(raw)
    model = trainModel(data[0])
    preds = runModel(data[1][0], model)
    evalResults(data[1], preds)
    
    """
    datasets = ["mnist_d", "mnist_f", "cifar_10", "cifar_100_f", "cifar_100_c"]
    ann_accs = [94.47, 81.93, 10.01, 5.02, 1]
    cnn_accs = [99.3, 92.3, 76.66, 45.6, 55]
    plot_bar_graph(datasets, ann_accs, "ANN_Accuracy_Plot.pdf")
    plot_bar_graph(datasets, cnn_accs, "CNN_Accuracy_Plot.pdf")
    """

if __name__ == '__main__':
    main()
