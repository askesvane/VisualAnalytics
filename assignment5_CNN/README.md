# Assignment 5 - CNN

## Description of the assignment
__Multi-class classification of impressionist painters__

So far in class, we've been working with 'toy' datasets - handwriting, cats, dogs, and so on. However, this course is on the application of computer vision and deep learning to cultural data. This week, your assignment is to use what you've learned so far to build a classifier which can predict artists from paintings.

You can find the data for the assignment here: https://www.kaggle.com/delayedkarma/impressionist-classifier-data

Using this data, you should build a deep learning model using convolutional neural networks which classify paintings by their respective artists. Why might we want to do this? Well, consider the scenario where we have found a new, never-before-seen painting which is claimed to be the artist Renoir. An accurate predictive model could be useful here for art historians and archivists!

For this assignment, you can use the CNN code we looked at in class, such as the ShallowNet architecture or LeNet. You are also welcome to build your own model, if you dare - I recommend against doing this.

Perhaps the most challenging aspect of this assignment will be to get all of the images into format that can be fed into the CNN model. All of the images are of different shapes and sizes, so the first task will be to resize the images to have them be a uniform (smaller) shape.

You'll also need to think about how to get the images into an array for the model and how to extract 'labels' from filenames for use in the classification report


## Repository structure and files

This repository has the following directory structure:

| Column | Description|
|--------|:-----------|
```out``` | Contains the outputs from running the script.
```cnn_artists.py```| The script to be executed from the terminal.
```create_lang_venv.sh``` | A bash script which automatically generates a new virtual environment 'CNN_env', and install all the packages contained within 'requirements.txt'
```requirements.txt``` | A list of packages along with the versions that are required.
```README.md``` | This readme file.


## Run the script

### Virtual environment
In order to run the script, one is required to set up the virtual environment with all necessary packages installed. Please clone the repo, navigate to the folder for this assignment, run the bash script to set up the environment, and lastly activate it. The following code should be executed from the terminal:

```bash
git clone https://github.com/askesvane/VisualAnalytics.git
cd assignment5_CNN
bash ./create_vision_venv.sh
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

## Results

The classification accuracy of the model (f1-score) is 0.36 running with the default parameters. A detailed csv file with the results can be found in 'out'. Increasing the size of the images did not seem to improve the results.

