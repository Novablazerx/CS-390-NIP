import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
import random
#from scipy.misc import imsave, imresize
#import imageio
import cv2
from scipy.optimize import fmin_l_bfgs_b   # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin_l_bfgs_b.html
from tensorflow.keras.applications import vgg19
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import warnings
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from IPython.display import Image, display
from PIL import Image


random.seed(1618)
np.random.seed(1618)
#tf.set_random_seed(1618)   # Uncomment for TF1.
tf.random.set_seed(1618)

tf.compat.v1.disable_eager_execution()

#tf.logging.set_verbosity(tf.logging.ERROR)   # Uncomment for TF1.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

CONTENT_IMG_PATH = "tiger.jpg"           #TODO: Add this.
#CONTENT_IMG_PATH = tf.keras.utils.get_file('YellowLabradorLooking_new.jpg', 'https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg')
#STYLE_IMG_PATH = ""             #TODO: Add this.
STYLE_IMG_PATH = tf.keras.utils.get_file('kandinsky5.jpg','https://storage.googleapis.com/download.tensorflow.org/example_images/Vassily_Kandinsky%2C_1913_-_Composition_7.jpg')


CONTENT_IMG_H = 500
CONTENT_IMG_W = 500

STYLE_IMG_H = 500
STYLE_IMG_W = 500

CONTENT_WEIGHT = 0.4    # Alpha weight.
STYLE_WEIGHT = 95      # Beta weight.
TOTAL_WEIGHT = 1.0

TRANSFER_ROUNDS = 3

#Custom session
#tf.compat.v1.disable_v2_behavior()
#sess = tf.compat.v1.Session()


#=============================<Helper Fuctions>=================================
'''
TODO: implement this.
This function should take the tensor and re-convert it to an image.
'''
def deprocessImage(x):

    #Reshape to appropriate dimensions
    x1 = np.copy(x)
    x1 = x1.reshape((500, 500, 3))
    
    # Reverse vgg19 preprocessing
    x1[:, :, 0] += 103.939
    x1[:, :, 1] += 116.779
    x1[:, :, 2] += 123.68

    x1 = x1[:, :, ::-1]
    x1 = np.clip(x1, 0, 255).astype("uint8")
    
    return x1


def gramMatrix(x):
    features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    gram = K.dot(features, K.transpose(features))
    return gram



#========================<Loss Function Builder Functions>======================

def styleLoss(style, gen, numFilters):
    #TODO: implement.
    #Style loss formula
    result = K.sum(K.square(gramMatrix(style) - gramMatrix(gen))) / (4 * (numFilters^2) * ((STYLE_IMG_H * STYLE_IMG_W)^2))
    print("result of style loss shape : ", result)
    return result

def contentLoss(content, gen):
    return K.sum(K.square(gen - content)) 


def totalLoss(x, arg1, arg2, arg3, arg4):
    print("begin func")
    model = arg1
    styleLayerNames = arg2
    contentLayerName = arg3
    outputDict = arg4

    contentLayer = outputDict[contentLayerName]
    contentOutput = contentLayer[0, :, :, :]
    print("content output shape : ", contentOutput.shape)
    genOutput = contentLayer[2, :, :, :]
    #loss += None   #TODO: implement - loss.
    content_loss = contentLoss(contentOutput, genOutput)

    style_loss = 0.0

    for layerName in styleLayerNames:
        #loss += None   #TODO: implement - loss.
        styleLayer = outputDict[layerName]
        styleOutput = styleLayer[1, :, :, :]
        genOutput = styleLayer[2, :, :, :]
        numFilters = styleLayer.shape.as_list()[0]
        print("number of filters : ", numFilters)
        style_loss += styleLoss(styleOutput, genOutput, numFilters) * STYLE_WEIGHT

    tot_loss = style_loss * STYLE_WEIGHT + content_loss * CONTENT_WEIGHT

    return  tot_loss #TODO: implement.




#=========================<Pipeline Functions>==================================

def getRawData():
    print("   Loading images.")
    print("      Content image URL:  \"%s\"." % CONTENT_IMG_PATH)
    print("      Style image URL:    \"%s\"." % STYLE_IMG_PATH)
    cImg = load_img(CONTENT_IMG_PATH)
    tImg = cImg.copy()
    sImg = load_img(STYLE_IMG_PATH)
    print("      Images have been loaded.")
    return ((cImg, CONTENT_IMG_H, CONTENT_IMG_W), (sImg, STYLE_IMG_H, STYLE_IMG_W), (tImg, CONTENT_IMG_H, CONTENT_IMG_W))



def preprocessData(raw):
    img, ih, iw = raw
    img = img_to_array(img)
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        #img = imresize(img, (ih, iw, 3))
    """
    size = (iw, ih)
    img = cv2.resize(img, size)
    img = img.astype("float64")
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    return img

'''
TODO: Allot of stuff needs to be implemented in this function.
First, make sure the model is set up properly.
Then construct the loss function (from content and style loss).
Gradient functions will also need to be created, or you can use K.Gradients().
Finally, do the style transfer with gradient descent.
Save the newly generated and deprocessed images.
'''
def styleTransfer(cData, sData, tData):
    print("   Building transfer model.")
    

    contentTensor = K.variable(cData)
    styleTensor = K.variable(sData)
    genTensor = K.placeholder((1, CONTENT_IMG_H, CONTENT_IMG_W, 3))
    inputTensor = K.concatenate([contentTensor, styleTensor, genTensor], axis=0)

    
    """
    contentTensor = tf.convert_to_tensor(cData)
    styleTensor = tf.convert_to_tensor(sData)
    genTensor = tf.cast(tf.convert_to_tensor(tf.Variable((1.0, CONTENT_IMG_H, CONTENT_IMG_W, 3.0))), tf.float64)
    inputTensor = tf.concat([contentTensor, styleTensor, genTensor], axis=0)
    """
    
    #model = None   #TODO: implement - set up model.
    #vgg19 model setup
    model = vgg19.VGG19(include_top=False, weights="imagenet", input_tensor=inputTensor)
    outputDict = dict([(layer.name, layer.output) for layer in model.layers])
    print("   VGG19 model loaded.")
    loss = 0.0
    styleLayerNames = ["block1_conv1", "block2_conv1", "block3_conv1", "block4_conv1", "block5_conv1"]
    contentLayerName = "block5_conv2"
    print("   Calculating content loss.")
    contentLayer = outputDict[contentLayerName]
    contentOutput = contentLayer[0, :, :, :]
    genOutput = contentLayer[2, :, :, :]
    #loss += None   #TODO: implement - loss.
    #calculate content loss
    content_loss = contentLoss(contentOutput, genOutput)
    loss += CONTENT_WEIGHT * content_loss
    print("   Calculating style loss.")

    #calculate style loss for each style layer
    for layerName in styleLayerNames:
        #loss += None   #TODO: implement - loss.
        styleLayer = outputDict[layerName]
        styleOutput = styleLayer[1, :, :, :]
        genOutput = styleLayer[2, :, :, :]
        numFilters = styleLayer.shape.as_list()[0]
        print("number of filters : ", numFilters)
        loss += styleLoss(styleOutput, genOutput, numFilters) * STYLE_WEIGHT
    
    #loss += 2 * STYLE_WEIGHT
    #loss += None   #TODO: implement - loss.
    #loss = CONTENT_WEIGHT * content_loss + STYLE_WEIGHT * style_loss
    # TODO: Setup gradients or use K.gradients().
    grads = K.gradients(loss, genTensor)
    
    #initial guess image is the transfer image from raw
    x = tData
    
    
    print("   Beginning transfer.")
    for i in range(TRANSFER_ROUNDS):
        print("   Step %d." % i)
        #TODO: Implement.
        #TODO: perform gradient descent using fmin_l_bfgs_b.
        #Reshape images to be used by kfunction
        x = x.reshape(1, STYLE_IMG_H, STYLE_IMG_W, 3)
        outputs = [loss]
        outputs.append(grads)

        the_args = ([genTensor], outputs)
        #set up k function
        kfunction = K.function([genTensor], outputs)
        out_puts = kfunction(x)
        
        #Loss function to pass to optimizer. Outputs the first term of 
        #tuple returned by k function
        def loss_func(x):
            x = x.reshape(1, STYLE_IMG_H, STYLE_IMG_W, 3)
            out_puts = kfunction(x)
            
            return out_puts[0]
        
        #Gradient function to pass to optimizer. Outputs the second term of 
        #tuple returned by k function
        def grad_func(x):
            x = x.reshape(1, STYLE_IMG_H, STYLE_IMG_W, 3)
            out_puts = kfunction(x)
            grads1 = np.asarray(out_puts[1])
            grads1 = grads1.flatten().astype('float64')
            return grads1
        
        #Call optimizer
        x = fmin_l_bfgs_b(func=loss_func, x0=x, fprime=grad_func, maxfun=32, maxiter=1000)
        
    
        print("after return")
        
        tLoss = 10.0
        print("      Loss: %f." % tLoss)
        img = deprocessImage(x[0])

        
        #Save image
        saveFile = "output_image_tiger.jpg"   #TODO: Implement.
        img = Image.fromarray(img)
        img.save(saveFile)
        x = x[0]
        
        print("      Image saved to \"%s\"." % saveFile)
    print("   Transfer complete.")





#=========================<Main>================================================

def main():
    print("Starting style transfer program.")
    raw = getRawData()
    cData = preprocessData(raw[0])   # Content image.
    sData = preprocessData(raw[1])   # Style image.
    tData = preprocessData(raw[2])   # Transfer image.
    styleTransfer(cData, sData, tData)
    print("Done. Goodbye.")



if __name__ == "__main__":
    main()

#Questions
#1. Should loss be a tensor
#Answer : No loss should be a scalar that needs to be minimized.
#2. How should loss function be passed to optimizer function
#Answer : Loss should be passed as a function to the optimizer function. 
