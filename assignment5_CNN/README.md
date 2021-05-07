# Assignment 5 - CNN

## Description of the assignment
__Multi-class classification of impressionist painters__
The assignment is to build a classifier which can predict artists from paintings. The data for the assignment can be found here: https://www.kaggle.com/delayedkarma/impressionist-classifier-data
Using this data, you should build a deep learning model using convolutional neural networks which classify paintings by their respective artists. Why might we want to do this? Well, consider the scenario where we have found a new, never-before-seen painting which is claimed to be the artist Renoir. An accurate predictive model could be useful here for art historians and archivists!

## The method
To solve this multiple classification problem, I have built a deep learning model using convolutional neural networks (CNN). 
I constructed a model with a ShallowNet architecture (see Model_plot.jpg in the folder 'out'). It takes the height and width specified at the command line as well as 3 (number of colour channels) as the input layer. The images are then fed into a single convolutional layer with a kernel size of 3x3 and a depth of 32. The images are ‘padded’ with an extra set of columns and rows only containing zeros to get around potential conflicts between the size of the image and the kernel. The model feeds the output of the convolutional layer into a ‘ReLU’ activation layer and subsequently flattens the image into a single dimension. Lastly, the flattened image is fed into a fully connected network with the number of painters as the number of potential outcomes using the ‘softmax’ activation function.

## Results and evaluation
The weighted accuracy of the ShallowNet model (f1-score) is 35% running with the default parameters (a detailed classification report with the results can be found in 'out'). Increasing the size of the resized images did not seem to improve the results. Although an accuracy of 35% is fairly low, it is important to keep in mind that the model had 10 potential outcomes (in terms of painters) and is thus performing significantly above change.

TrainingLossAndAccuracy_plot.jpg (in 'out') illustrates the training loss and accuracy of the model evolving over 40 epochs. The graphs representing the training and validation accuracy start to diverge after approximately 10 epochs where the training accuracy keeps improving while the validation accuracy stagnates. This clearly shows that the model is overfitting to the training data. The training loss graphs indicate the same tendency as they quickly start to diverge (training loss keeps decreasing while validation loss stagnates). To come about the issue of overfitting, more data would probably improve the results. Since that is, in this case, not possible, other preprocessing steps like removing outliers, duplicates, hand drawn sketches etc. could potentially diminish the overfitting. 

## Repository structure and files
This repository has the following directory structure:

| Column | Description|
|--------|:-----------|
```out``` | Contains the outputs from running the script.
```cnn_artists.py```| The script to be executed from the terminal.
```create_visual_venv.sh``` | A bash script which automatically generates a new virtual environment 'CNN_env', and install all the packages contained within 'requirements.txt'
```requirements.txt``` | A list of packages along with the versions that are required.
```README.md``` | This readme file.


## Usage (reproducing the results)

### Virtual environment
In order to run the script, one is required to set up the virtual environment with all necessary packages installed. Please clone the repo, navigate to the folder for this assignment, run the bash script to set up the environment, and lastly activate it. The following code should be executed from the terminal:

```bash
git clone https://github.com/askesvane/VisualAnalytics.git
cd assignment5_CNN
bash ./create_visual_venv.sh
source ./CNN_env/bin/activate
```

### Get the data
The data occupies approximately 2GB and thus exceeds Github's file size limit. To run the script, the data has to be downloaded from kaggle.com: https://www.kaggle.com/delayedkarma/impressionist-classifier-data
- Download the zip-file and unzip it.
- Create a new folder in ```.../VisualAnalytics/assignment5_CNN/```called 'data'.
- Move the folders 'training' and 'validation' both containing several additional folders (named after each painter) to the new data folder. 

### Execute the script 
Now, the script can be executed. You can specify the height and width of the resized images. In both cases the default is 50. Additionally, you can specify the number of epochs. The default is 40.

```bash
python cnn_artists.py --resize_height 50 --resize_width 50 --epochs 40 
```
While running, status updates will be printed to the terminal. Afterwards, the classification report and plots can be found in the folder called 'out'. It takes approximately 4 minutes.