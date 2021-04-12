#!/usr/bin/env python

#_____________# Packages #_____________#

import argparse
import os
import sys
import numpy as np
import pandas as pd
import glob
import cv2
import matplotlib.pyplot as plt
sys.path.append(os.path.join(".."))

# sklearn tools
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, 
                                     MaxPooling2D, 
                                     Activation, 
                                     Flatten, 
                                     Dense)
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import backend as K

#_____________# The script #_____________#

# FUNCTIONS

"""
image_list(): Creates two lists, one containing image objects extracted from folders given the arguments specified and one containing numeric values representing the painters of the images.
"""
def image_list(painter, folder, X, Y, painter_counter):
    for image_path in glob.glob(os.path.join("data", folder, f"{painter}", "*.jpg")):
        X.append(cv2.imread(image_path))   
        Y.append(painter_counter)
        
    return(X, Y)

"""
resize(): Takes an image list and predefined dimensions and resize all images in the list accordingly. Additionally, the function normalizes them (constraining numbers representing pixel itensities between 0-1)
"""
def resize(X, height, width):
    
    # Create 'dim' object from the height and width
    dim = (height, width) 
    
    output = []
    for image in X:    
        # Resize
        resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
        # Normalize the image
        resized = resized.astype("float") / 255.
        
        output.append(resized)
    
    return(output)

"""
plot_history(): Takes a model and a number of epochs, creates a plot of training loss and accuracy, and saves the plot in the folder 'out'.
"""
def plot_history(H, epochs):
    # visualize performance
    plt.style.use("fivethirtyeight")
    plt.figure()
    plt.plot(np.arange(0, epochs), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, epochs), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, epochs), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, epochs), H.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig("out/TrainingLossAndAccuracy_plot.jpg")


"""
dataset_generator(): Takes the list of painters and constructs the 4 necessary train and test objects to feed the model
"""
def dataset_generator(painters):
    
    # Empty containers
    trainX = []
    testX = []
    
    trainY = []
    testY = []

    # Create a counter
    painter_counter = 0
    
    for painter in painters:
    
        # Print message to terminal
        print(f"Importing images from {painter}")
        
        # Training data
        trainX, trainY = image_list(painter, "training", trainX, trainY, painter_counter) 

        # Test data
        testX, testY = image_list(painter, "validation", testX, testY, painter_counter)
        
        # Add 1 to the painter counter
        painter_counter = painter_counter + 1
    
    # Binarize the labels
    lb = LabelBinarizer() 
    trainY = lb.fit_transform(trainY)
    testY = lb.fit_transform(testY)
    
    return(trainX, trainY, testX, testY)
    
def model_initializer(painters, height, width):
    
    # initialise model
    model = Sequential()
    
    # define CONV => RELU layer
    model.add(Conv2D(32, (3, 3), # depth of convolutional layer - 32
                     padding="same", # means zeros around the outside of the image
                     input_shape=(height, width, 3))) # input shape of the data heightxwidth and 3 color channels
    # activation layer
    model.add(Activation("relu"))
    
    # softmax classifier
    model.add(Flatten())
    model.add(Dense(len(painters)))
    model.add(Activation("softmax"))
    
    
    # Compile model
    opt = SGD(lr =.01) # usually 0.001 --> 0.01
    model.compile(loss="categorical_crossentropy",
                  optimizer=opt,
                  metrics=["accuracy"])
    
    return(model)
    

# MAIN FUNCTION
def main(args):
    
    # Import parameters specified in the commanline
    height = args["resize_height"]
    width = args["resize_width"]

    # Create list of painters from one of the folders
    filepath = os.path.join("data", "validation")
    painters = os.listdir(filepath)
    
    # Create the 4 necessary list to feed the model
    trainX, trainY, testX, testY = dataset_generator(painters)
    
    # Resize the images according to the dimensions specified in the terminal  
    print("Currently resizing the images")
    trainX = resize(trainX, height, width)
    testX = resize(testX, height, width)

    # change image data into array and appropriately reshape
    trainX = np.array(trainX).reshape(len(trainX),height,width,3) # 3 represents number of color channels
    testX = np.array(testX).reshape(len(testX),height,width,3)
    
    # Initialise and compile model with predefined function
    model = model_initializer(painters, height, width)
    
    # Train model
    print("Currrently training the model")
    H = model.fit(trainX, trainY, 
                  validation_data=(testX, testY), 
                  batch_size=100,
                  epochs=40,
                  verbose=1)
    
    
    # THE MODEL OUTPUT
    
    # save 'training loss and accuracy plot' and 'model plot' in folder out
    plot_history(H, 40)
    plot_model(model, to_file='out/Model_plot.jpg', show_shapes=True, show_layer_names=True)
    
    # Save classification report in folder out
    predictions = model.predict(testX, batch_size=100)
    
    class_report = pd.DataFrame(classification_report(testY.argmax(axis=1),
                            predictions.argmax(axis=1),
                            target_names=painters,
                            output_dict = True))
    class_report.to_csv("out/ClassificationReport.csv")
    
    # print to terminal
    print("Classification Report:")
    print(class_report) 
    print("A plot of the training loss and accuracy, a model plot, and the classification report have successfully been saved in the folder 'out'")
    

# RUN THE SCRIPT
if __name__=="__main__":
    
    # Argument parser
    ap = argparse.ArgumentParser()
    ap.add_argument("-a", "--resize_height", default = 30, type = int,
                    help = "The default is 30")
    ap.add_argument("-b", "--resize_width", default = 30, type = int,
                    help = "The default is 30")
    
    # Parse arguments. args is now an object containing all arguments added through the terminal. 
    argument_parser = vars(ap.parse_args())
    
    # run main() function
    main(argument_parser)




















