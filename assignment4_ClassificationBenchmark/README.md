# Assignment 4 - Classification Benchmark

## Assignment description

__Classifier benchmarks using Logistic Regression and a Neural Network__

This assignment builds on the work we did in class and from session 6.

You'll use your new knowledge and skills to create two command-line tools which can be used to perform a simple classification task on the MNIST data and print the output to the terminal. These scripts can then be used to provide easy-to-understand benchmark scores for evaluating these models.

You should create two Python scripts. One takes the full MNIST data set, trains a Logistic Regression Classifier, and prints the evaluation metrics to the terminal. The other should take the full MNIST dataset, train a neural network classifier, and print the evaluation metrics to the terminal.

## Peer-review instructions

__Setup__
- Clone the whole repository to a chosen location on your computer by executing ```git clone https://github.com/askesvane/VisualAnalytics.git``` in the terminal.
- Through the terminal, navigate to the folder for assignment 4 by executing ```cd VisualAnalytics/assignment4_ClassificationBenchmark```
- Set up the virtual environment by executing ```bash create_visual_venv.sh``` in the terminal. A virtual environment called 'classification_env' should now appear in the folder. The virtual environment automatically installs all required packages provided in the 'requirements.txt' file.
- Activate the virtual environment by executing ```./source classification_env/bin/activate``` in the terminal. You should now see the virtual environment in parentheses at the commandline.

__Run the scripts__
- There are two scripts to be executed - one using a logistic regression model and one using a neural network to classify.
- First, run either the python script 'lr-mnist.py' or 'lr-mnist_classes.py'. They do exactly the same - the only difference is that I tried to work with classes in the second one. You have to specify the size of the test dataset as a numric value between 0 and 1. Additionally, you can specify the filename of the classification report to be saved (remember '.csv' at the end). Execute ```python [either 'lr-mnist.py' or 'lr-mnist_classes.py'] --test_size --filename_out```.
- Second, run the python script 'nn-mnist.py'. You have to specify the size of the test dataset again. Additionally, you can specify up to three hidden layers (seperated by space e.g. '40 30 20'). The input layer has 64 and the output layer 10. These should not be specified in the commandline. Number of epochs as well as the output filename can also be specified. Execute ```python nn-mnist.py --test_size --layers --epochs --filename_out```.
- The two classification reports can be found in the folder 'out'.




















