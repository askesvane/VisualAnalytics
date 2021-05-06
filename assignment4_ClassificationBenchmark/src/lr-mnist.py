#!/user/bin/python

#____________ IMPORT PACKAGES ____________#

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


#____________ THE MAIN FUNCTION ____________#

def main(args):
    
    # Import parameters specified in the commandline + calling main class
    ClassificationAnalysis(test_size = args["test_size"])

    

#____________ CLASS ____________#

class ClassificationAnalysis:
    
    def __init__(self, test_size):
        
        # setting test_size according to the provided information.
        self.test_size = test_size
        
        # Get data devided in data and labels
        print("Importing data...")
        X, y = self.get_data()
        
        # Split into test and train datasets 
        X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled = self.split_data(X = X, 
                                                                                          y = y, 
                                                                                          test_size = self.test_size)
        # Get predicted labels with the logistic classification model
        print("Training the logistic regression model on the training data...")
        y_pred = self.ClassificationModel(X_train_scaled = X_train_scaled, 
                                          X_test_scaled = X_test_scaled, 
                                          y_train = y_train)
    
        # Print result to terminal
        self.printer(y_test = y_test, y_pred = y_pred)
        
        # save as csv
        self.save_report(y_test = y_test, y_pred = y_pred, filename = "lr_ClassifierReport.csv")
        
        # save confusion matrix plot
        clf_util.save_cm(y_test, y_pred, "../out/lr_ConfusionMatrix.png", normalized=True)
        
        # Print status to terminal
        print("A classification report and a confusion matrix have successfully been saved in the folder 'out'.")
        
        
    def get_data(self):
        
        """
        The MNIST_784 data is imported with fetch_openml. The images are saved as lists of pixel intensities in 
        the object X; their labels (in terms of number 1-9) are saved as a list in y.
        """
        X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
    
        # I make sure that X and y are numpy arrays
        X = np.array(X)
        y = np.array(y)
        
        return(X, y)


    def split_data(self, X, y, test_size):
        """
        Function that splits and normalises the data.
        """
        # Creating test and training dataset
        X_train, X_test, y_train, y_test = train_test_split(X,
                                                            y, 
                                                            random_state=9, 
                                                            test_size=test_size)
    
        #scaling the features (constrain intensities between 0 and 1)
        X_train_scaled = X_train/255.0
        X_test_scaled = X_test/255.0
        
        return(X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled)
    
    
    def ClassificationModel(self, X_train_scaled, X_test_scaled, y_train):
        """
        Function that trains a classification model and provides label predictions as the output.
        """
        # Training the logistic regression model on the training data
        ClassificationModel = LogisticRegression(penalty='none', 
                             tol=0.1, 
                             solver='saga',
                             multi_class='multinomial').fit(X_train_scaled, y_train)
        y_pred = ClassificationModel.predict(X_test_scaled)
        
        return(y_pred)

    
    def printer(self, y_test, y_pred):
        """
        Print the classification metrics to the terminal
        """
        # Print accuracy score to the terminal
        accuracy = accuracy_score(y_test, y_pred)
        print(f"The accuracy score is {accuracy}.")
    
        # Print
        ClassificationMetrics = metrics.classification_report(y_test, y_pred)
        print(ClassificationMetrics)
    
    
    def save_report(self, y_test, y_pred, filename):
        """   
        Create report object, change into df with pandas, 
        save in folder 'out' with the filename specified in the commandline.
        """
        report = metrics.classification_report(y_test, y_pred, output_dict=True)
        df = pd.DataFrame(report).transpose()
        output_path = '../out/{}'.format(filename)
        df.to_csv(output_path, index=True)


#____________ RUN MAIN FUNCTION ____________#

if __name__=="__main__":
    
    # Argument parser
    ap = argparse.ArgumentParser()
    ap.add_argument("-s", "--test_size", default = 0.25, type = float,
                    help = "Specify test data proportion from 0 to 1. The default is 0.25")
    
    # Parse arguments. args is now an object containing all arguments added through the terminal. 
    argument_parser = vars(ap.parse_args())
    
    # run main() function
    main(argument_parser) 
