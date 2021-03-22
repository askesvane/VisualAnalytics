#!/user/bin/python

# Import packages 
import sys,os
sys.path.append(os.path.join(".."))
import argparse
from utils.neuralnetwork import NeuralNetwork
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import datasets
from sklearn import metrics
import pandas as pd


# Defining the main function

def main(args):
    
    # Import parameters specified in the commandline
    test_size = args["test_size"]
    layers = args["layers"]
    epochs = args["epochs"]
    filename = args["filename_out"]
    
    # Load the data
    digits = datasets.load_digits()
    # Convert to floats
    data = digits.data.astype("float")
    
    # MinMax regularization. constrain between 0 and 1
    data = (data - data.min())/(data.max() - data.min())
    
    # split data
    X_train, X_test, y_train, y_test = train_test_split(data, 
                                                      digits.target, 
                                                      test_size=test_size)
    
    # convert labels in y from integers to vectors
    y_train = LabelBinarizer().fit_transform(y_train)
    y_test = LabelBinarizer().fit_transform(y_test)
    
    # change the list of hidden layers specified in the terminal to integers
    for i in range(0, len(layers)): 
        layers[i] = int(layers[i])
    
    #train network (while printing information in the terminal)
    print("[INFO] training network...")
    
    if (len(layers)==1):
        nn = NeuralNetwork([X_train.shape[1], layers[0], 10])
    elif (len(layers)==2):
        nn = NeuralNetwork([X_train.shape[1], layers[0], layers[1], 10])
    elif (len(layers)==3):
        nn = NeuralNetwork([X_train.shape[1], layers[0], layers[1], layers[2], 10])
    else:
        nn = NeuralNetwork([X_train.shape[1], 10])

    print("[INFO] {}".format(nn))
    nn.fit(X_train, y_train, epochs=epochs)
    
    # evaluate network (and print classification report to terminal)
    print(["[INFO] evaluating network..."])
    predictions = nn.predict(X_test)
    predictions = predictions.argmax(axis=1)
    print(classification_report(y_test.argmax(axis=1), predictions))
    
    
    
    """   
    - Create report object
    - change into df with pandas
    - save in folder 'out' with the filename specified in the commandline.
    """
    report = metrics.classification_report(y_test.argmax(axis=1), predictions, output_dict=True)
    df = pd.DataFrame(report).transpose()
    output_path = '../out/{}'.format(filename)
    df.to_csv(output_path, index=True)
    
    
if __name__=="__main__":
    
    # Create an argparse argument
    ap = argparse.ArgumentParser()
    
    # Assign argument specified in the terminal to 'ap'
    ap.add_argument("-s", "--test_size", required = True, type = float,
                    help = "Specify test data proportion from 0 to 1.")
    ap.add_argument("-l", "--layers", default = [], nargs = "+",
                help = "Specify up to 3 hidden layers seperated by space. Default is none")
    ap.add_argument("-e", "--epochs", default = 1000, type = int,
                help = "Specify number of epochs. Default is 1000")
    ap.add_argument("-n", "--filename_out", default = "nn_ClassifierReport.csv", 
                    help = "Specify a filename for the classifier report. Default is 'nn_ClassifierReport.csv'.")


    # Parse arguments. args is now an object containing all arguments added through the terminal. 
    argument_parser = vars(ap.parse_args())
    
    # run main() function
    main(argument_parser)




