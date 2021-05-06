#!/user/bin/python

#___________ Import packages ___________#

# System tools etc.
import sys,os
sys.path.append(os.path.join(".."))
import argparse
import pandas as pd

# Functions from utils
from utils.neuralnetwork import NeuralNetwork
import utils.classifier_utils as clf_util

# sklearn tools
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import datasets
from sklearn import metrics


#___________ FUNCTIONS ___________#

def get_data():
    """
    Gets the MNIST data with 'datasets', converts to float and constrains the pixel intensities between 0 and 1.
    """
    # Load the data
    digits = datasets.load_digits()
    # Convert to floats
    data = digits.data.astype("float")
    # MinMax regularization. constrain between 0 and 1
    data = (data - data.min())/(data.max() - data.min())

    return(data, digits)

def network_trainer(layers, X_train):
    """
    Function that takes layers specified in the terminal and the training data, trains a model based on the layers and returns it. 
    The model relies on the utils script neuralnetworks.py
    """
    # change the list of hidden layers specified in the terminal to integers
    for i in range(0, len(layers)): 
        layers[i] = int(layers[i])
    
    #train network (while printing information in the terminal)
    print("[INFO] training network...")
    
    # Depending on the number of hidden layers (0-3) specified at the commandline, train the network. 
    if (len(layers)==1):
        nn = NeuralNetwork([X_train.shape[1], layers[0], 10])
    elif (len(layers)==2):
        nn = NeuralNetwork([X_train.shape[1], layers[0], layers[1], 10])
    elif (len(layers)==3):
        nn = NeuralNetwork([X_train.shape[1], layers[0], layers[1], layers[2], 10])
    else:
        nn = NeuralNetwork([X_train.shape[1], 10])
            
    return(nn)

def save_report(y_test, predictions, filename):
    """   
    Takes y_test, predictions and a filename.
    Create report object, change into df with pandas, 
    save in folder 'out' with the filename specified.
    """
    report = metrics.classification_report(y_test.argmax(axis=1), predictions, output_dict=True)
    df = pd.DataFrame(report).transpose()
    output_path = '../out/{}'.format(filename)
    df.to_csv(output_path, index=True)

    
#___________ MAIN FUNCTION ___________#

def main(args):
    
    # Import parameters specified in the commandline
    test_size = args["test_size"]
    layers = args["layers"]
    epochs = args["epochs"]
    
    # get data regularized
    data, digits = get_data()
    
    # split data in train and test data given the test size specified at the commandline
    X_train, X_test, y_train, y_test = train_test_split(data, 
                                                      digits.target, 
                                                      test_size=test_size)
    
    # convert labels in y from integers to vectors 
    y_test_int = y_test # (keep y_test labels in another object as integers to plot confusion matrix)
    y_train = LabelBinarizer().fit_transform(y_train)
    y_test = LabelBinarizer().fit_transform(y_test)
    
    # Train the neural network based on the number of layers specified in the terminal
    nn = network_trainer(layers, X_train)
    
    # Print information nn (number of epochs corresponding loss)
    print("[INFO] {}".format(nn))
    nn.fit(X_train, y_train, epochs=epochs)
    
    # evaluate network (and print classification report to terminal)
    print(["[INFO] evaluating network..."])
    predictions = nn.predict(X_test)
    predictions = predictions.argmax(axis=1)
    print(classification_report(y_test.argmax(axis=1), predictions))
    
    # Save report with predefined function
    save_report(y_test, predictions, 'nn_ClassifierReport.csv')
    # Save confusion matrix
    clf_util.save_cm(y_test_int, predictions, "../out/nn_ConfusionMatrix.png", normalized=True)
      
    
if __name__=="__main__":
    
    # Create an argparse argument
    ap = argparse.ArgumentParser()
    
    # Assign argument specified in the terminal to 'ap'
    ap.add_argument("-s", "--test_size", default = 0.25, type = float,
                    help = "Specify test data proportion from 0 to 1.")
    ap.add_argument("-l", "--layers", default = [], nargs = "+",
                help = "Specify up to 3 hidden layers seperated by space. Default is none")
    ap.add_argument("-e", "--epochs", default = 1000, type = int,
                help = "Specify number of epochs. Default is 1000")

    # Parse arguments. args is now an object containing all arguments added through the terminal. 
    argument_parser = vars(ap.parse_args())
    
    # run main() function
    main(argument_parser)




