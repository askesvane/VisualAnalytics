# Assignment 4 - Classification Benchmark

## Assignment description

__Classifier benchmarks using Logistic Regression and a Neural Network__

The assignment is to create two commandline tools which can be used to perform a simple classification task on the MNIST data and print the output to the terminal. These scripts can then be used to provide easy-to-understand benchmark scores for evaluating these models.

You should create two Python scripts. One takes the full MNIST data set, trains a Logistic Regression Classifier, and prints the evaluation metrics to the terminal. The other should take the full MNIST dataset, train a neural network classifier, and print the evaluation metrics to the terminal.

## Logistic regression classifier

### Methods 
The first script 'lr_mnist.py' employs a simple multiclass logistic regression model to solve the classification task. The performance of this classification model establishes a baseline comparable to the performance of the neural network model.

### Results and evaluation
The accuracy score when running the script with the default data test size (0.25) is 92%. A detailed classification report is printed in the terminal and saved in the folder 'out' as 'lr_ClassifierReport.csv'. Additionally, a confusion matrix 'lr_ConfusionMatrix.png' is saved in the folder. The confusion matrix shows that the model performs extremely well (close to 100% accuracy) when classifying certain numbers (including 1,2, and 6) while performing more purely with other numbers (3, 5, and 8). The matrix indicates that the model tends to confuse these numbers with each other. This is probably due to the nature of these numbers in terms of structural similarities.

## Neural network classifier 

### Methods
The second script utilizes the predefined class NeuralNetwork() which can be found in the the script ```utils/neuralnetwork.py```. The neural network is given the size of the input layer, output layer as well as potential hidden layers. 

### Results and evaluation
The accuracy of running the script with the default test size (0.25) and two hidden layers (64-32-16-10) was 97% thus outperforming the simple logistic regression model. A classification report and a confusion matrix can be found in 'out' as 'nn_ClassifierReport.csv' and 'nn_ConfusionMatrix.png'. The confusion matrix indicates that, although generally performing with high accuracy, certain numbers tend to be harder to distinguish from each other than others. For instance, in 9% of the cases 5 is predicted as a 9. This is likely due to their structural similarities.

Employing a neural network seems to improve performance over simpler logistic regression classifier tools. The model manages to significantly better learn and predict the different numbers. However, it is crucial to keep in mind that this dataset is not representative for 'real world' classification tasks. All images were black/white, they were of the same size, and the numbers were written approximately at the center of each image etc. These factors undoubtably facilitated the very high accuracy.

## Repository structure and files
This repository has the following directory structure:

| Column | Description|
|--------|:-----------|
```out``` | Folder containing the output classification reports as well as confusion matrices from running the two scripts.
```src``` | Folder containing the scripts to be executed from the commandline.
```utils``` | Folder containing scripts with additional function required to run the classification scripts.
```create_lang_venv.sh``` | A bash script which automatically generates a new virtual environment 'classification_env', and install all the packages contained within 'requirements.txt'
```requirements.txt``` | A list of packages along with the versions that are required.
```README.md``` | This readme file.


## Usage (reproducing the results)

### Virtual environment
In order to run the script, one is required to set up the virtual environment with all necessary packages installed specified in 'requirements.txt'. Please clone the repo, navigate to the folder for this assignment, run the bash script to set up the environment, and lastly activate it. The following code should be executed from the terminal:

```bash
git clone https://github.com/askesvane/VisualAnalytics.git
cd assignment4_ClassificationBenchmark
bash ./create_vision_venv.sh
source ./classification_env/bin/activate
```

### Get the data
The MNIST dataset will automatically be imported when running the scripts.

### Execute the scripts
First, navigate to the folder containing the scripts:
```bash
cd src
```
__Logistic regression classifier__ <br>
Run the script 'lr-mnist.py'. You can specify the size of the test dataset relative to the full dataset. The script takes any value between 0 and 1. The default is 0.25.

```bash
python lr-mnist.py --test_size 0.25
```
__Neural network classifier__ <br>
Run the script 'nn-mnist.py'. As in the previous script, you can specify the size of the test dataset (the default is 0.25). Additionally, you can specify up to three hidden layers (seperated by space e.g. '40 30 20'). The input layer has 64 and the output layer 10. These should not be specified at the commandline. The default is no hidden layers. However, I will recommend two hidden layers, 32 and 16. The number of epochs can also be specified (the default is 1000).

```bash
python nn-mnist.py --test_size 0.25 --layers 32 16 --epochs 1000
```