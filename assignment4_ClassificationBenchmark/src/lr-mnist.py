#!/user/bin/python

# IMPORT PACKAGES

# Import packages
import argparse
import os
import sys
import pandas as pd
sys.path.append(os.path.join(".."))

# Import teaching utils
import numpy as np
import utils.classifier_utils as clf_util

# Import sklearn metrics
from sklearn import metrics
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# THE MAIN FUNCTION

def main(args):
    
    # Import parameters specified in the commanline
    test_size = args["test_size"]
    filename = args["filename_out"]
    
    
    """
    The MNIST_784 data is imported with fetch_openml. The images are saved as lists of pixel intensities in 
    the object X; their labels (in terms of number 1-9) are saved as a list in y.
    """
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
    
    # I make sure that X and y are numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    # Creating test and training dataset
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y, 
                                                        random_state=9, # I am not sure what this one does
                                                        test_size=test_size)
    
    #scaling the features (constrain intensities between 0 and 1)
    X_train_scaled = X_train/255.0
    X_test_scaled = X_test/255.0
    
    # Training the logistic regression model on the training data
    ClassificationModel = LogisticRegression(penalty='none', 
                         tol=0.1, 
                         solver='saga',
                         multi_class='multinomial').fit(X_train_scaled, y_train)
    
    
   
    """
    Prediction for all test data
    'y_pred' is a long array of predicted labels (from 0-9)
    """
    y_pred = ClassificationModel.predict(X_test_scaled)

    # Print accuracy score to the terminal
    accuracy = accuracy_score(y_test, y_pred)
    print(f"The accuracy score is {accuracy}.")
    
    # Print the classification metrics to the terminal
    ClassificationMetrics = metrics.classification_report(y_test, y_pred)
    print(ClassificationMetrics)
    
    
    
    """   
    - Create report object
    - change into df with pandas
    - save in folder 'out' with the filename specified in the commandline.
    """
    report = metrics.classification_report(y_test, y_pred, output_dict=True)
    df = pd.DataFrame(report).transpose()
    output_path = '../out/{}'.format(filename)
    df.to_csv(output_path, index=True)
    

if __name__=="__main__":
    
    # Argument parser
    ap = argparse.ArgumentParser()
    ap.add_argument("-s", "--test_size", required = True, type = float,
                    help = "Specify test data proportion from 0 to 1.")
    ap.add_argument("-n", "--filename_out", default = "lr_ClassifierReport.csv", 
                    help = "Specify a filename for the classifier report. Default is 'lr_ClassifierReport.csv'.")
    
    # Parse arguments. args is now an object containing all arguments added through the terminal. 
    argument_parser = vars(ap.parse_args())
    
    # run main() function
    main(argument_parser) 
