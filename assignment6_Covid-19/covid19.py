#!/usr/bin/python

#___________ Import packages ___________#

# System tools, plotting etc.
import sys, os
sys.path.append(os.path.join(".."))
import argparse
import numpy as np
import glob
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import pydot, graphviz

# sklearn tools
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# VGG16 model
from tensorflow.keras.applications.vgg16 import VGG16 # preprocess_input,decode_predictions,

# layers
from tensorflow.keras.layers import (Flatten, 
                                     Dense, 
                                     Dropout)

# generic model object
from tensorflow.keras.models import Model
# optimizer
from tensorflow.keras.optimizers import Adam
# tensorflow plot function
from tensorflow.keras.utils import plot_model


#___________ FUNCTIONS ___________#

def resizer(X, height, width):
    """
    resizer(): Takes a list of images as well as predefined height and width. 
    It returns all images normalized (pixel intensities are constrained between 0-1) and resized.
    """
    # Create 'dim' object from the height and width
    dim = (height, width) 
    
    # Loop over every image object in X and resize them
    output = []
    for image in X:    
        # Resize
        resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
        # Normalize the image
        resized = resized.astype("float") / 255.
        output.append(resized)
    
    return(output)


def plot_history(H, epochs, filename):
    """
    plot_history(): Takes a model and a number of epochs, creates a plot of 
    training loss and accuracy, and saves the plot in the folder 'out'. 
    This function has been made by Ross 
    (https://github.com/CDS-AU-DK/cds-visual/blob/main/notebooks/session9.ipynb) 
    and only slightly motified to save the output.)
    """
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
    plt.savefig(filename)
       

def get_data():
    """
    get_data(): Function that import data from 6 different folders, gather the images in a list X 
    while saving the corresponding labels in y. The function returns X and y.
    """
    # Get conditions (names of the folders) to loop over all three and import images
    filepath = os.path.join("data", "train")
    conditions = os.listdir(filepath)
    
    # Create empty lists 
    X = []
    y = []
    
    # Loop over all three conditions, extract every image, save it in X and the corresponding condition in y.
    for condition in conditions:
        for image in glob.glob(os.path.join("data", "*",f"{condition}", "*")):
            # Append the image to X
            X.append(cv2.imread(image))
            # Append the condition label to y
            if condition == 'Viral Pneumonia':
                y.append("pneumonia")
            elif condition == 'Normal':
                y.append("normal")
            elif condition == 'Covid':
                y.append("covid")
    
    return(X, y, conditions)


def save_report(y_test, predictions, conditions, filename):
    """   
    Takes y_test, predictions and a filename.
    Create report object, change into df with pandas, 
    save in folder 'out' with the filename specified.
    """
    # Create report object
    report = classification_report(y_test.argmax(axis=1),
                                   predictions.argmax(axis=1),
                                   target_names=conditions,
                                   output_dict=True)
    # Save it as df, create output filepath and save it
    df = pd.DataFrame(report).transpose()
    output_path = 'out/{}'.format(filename)
    df.to_csv(output_path, index=True)
    
    
#___________ MAIN FUNCTION ___________#

def main(args):
    
    # import data
    print("Importing images...")
    X, y, conditions = get_data()

    # Parameters that can be specified in the command line
    height = args["height"]
    width = args["width"]
    epochs = args["epochs"]
    test_size = args["test_size"]
    
    # Resize the images according to values specified in the command line
    print(f"The images are being resized to {height}x{width}...")
    X_resized = resizer(X, height, width)
    
    # change image data into array and appropriately reshape
    X_resized = np.array(X_resized).reshape(len(X_resized), height, width, 3)
    
    # Split X and y in train and test datasets given the test_size specified in the command line
    X_train, X_test, y_train, y_test = train_test_split(X_resized,
                                                   y,
                                                   random_state = 9,
                                                   test_size = test_size)
    
    # Binarize the labels
    lb = LabelBinarizer() 
    y_train = lb.fit_transform(y_train)
    y_test = lb.fit_transform(y_test)
    
    # Initialize the VGG16 model
    model = VGG16()
    
    # loading the model (without classifier layers)
    model = VGG16(include_top=False,
                  pooling='avg',
                  input_shape=(height, width, 3))
    
    # disable training of loaded layers
    for layer in model.layers:
        layer.trainable = False
    
    # Flatten and dense layers
    flat1 = Flatten()(model.layers[-1].output)
    class1 = Dense(256,
                   activation='relu')(flat1)
    # drop out layer (there is a 20% probability of a node being randomnly dropped)
    drop1 = Dropout(0.2)(class1)
    # last classifiction layer with 3 possible outcomes
    output = Dense(3, 
                   activation='softmax')(drop1)
    
    # define new model
    model = Model(inputs=model.inputs, 
                  outputs=output)
    
    # Compile model using Adam
    model.compile(optimizer=Adam(lr=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    # fit the model
    print("The data is fed to the model...")
    H = model.fit(X_train, y_train, 
              validation_data=(X_test, y_test), 
              batch_size=128,
              epochs=epochs,
              verbose=1)

    # Get predictions and print them in the terminal
    predictions = model.predict(X_test, batch_size=32)
    print("Classification report:")
    print(classification_report(y_test.argmax(axis=1),
                                predictions.argmax(axis=1),
                                target_names=conditions))
    #Output filenames
    report_filename = 'Covid_ClassifierReport.csv'
    history_filename = 'Covid_TrainingLossAndAccuracy.jpg'
    model_filename = "Covid_ClassificationModel.jpg"
    
    # Save report, training loss/accuracy plot and an illustration of the model to the folder 'out'
    save_report(y_test, predictions, conditions, report_filename)
    plot_history(H, epochs, f"out/{history_filename}")
    plot_model(model, to_file=f'out/{model_filename}', show_shapes=True, show_layer_names=True)
    print(f"The classification report, a plot illustrating the training loss and accuracy, and a plot of the deep learning model have been saved in the folder 'out' as '{report_filename}', '{history_filename}', and '{model_filename}'")
    
#___________ RUN MAIN FUNCTION ___________#
if __name__=="__main__":
    
    # Argument parser
    ap = argparse.ArgumentParser()
    ap.add_argument("-a", "--height", default = 32, type = int,
                    help = "The default is 32")
    ap.add_argument("-b", "--width", default = 32, type = int,
                    help = "The default is 32")
    ap.add_argument("-e", "--epochs", default = 10, type = int,
                    help = "The default is 10")
    ap.add_argument("-s", "--test_size", default = 0.25, type = int,
                    help = "The default is 0.25")
    
    # Parse arguments. args is now an object containing all arguments added through the terminal. 
    argument_parser = vars(ap.parse_args())
    
    # run main() function
    main(argument_parser)














