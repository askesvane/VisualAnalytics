# Assignment 6  - Corona Classification from X-rays
__Self-assigned project__

## Description of the assignment

__Assignment__

The assignment is set to build a classifier which can predict whether or not the patient was affected with covid or viral pneumonia based on the x-ray chest scans. The data for the assignment can be found [here](https://www.kaggle.com/khoongweihao/covid19-xray-dataset-train-test-sets).

__Contribution__

This assignment was written as a group project between Hanna Janina Matera (au603273) and Aske Svane Qvist (au613522), where: 

> “Both students contributed equally to every stage of this project from initial conception and implementation, through the production of the final output and structuring of the repository. (50/50%)”

__Background__

The X-ray scanning technique is one of the methods used to detect the effect of several respiratory diseases, COVID-19 being one of them. While the diagnosis is most often confirmed using polymerase chain reaction (PCR) or antygene tests, infected patients with Covid may present on chest X-ray images with apparent patterns of abnormality detectable with a naked eye.
These abnormalities comprise things like bilateral multiple lobular and subsegmental areas of consolidation, bilateral ground-glass opacity and subsegmental areas of consolidation in the chest. 
An accurate and timely classification of Covid (or other respiratory diseases) would enable a swift implementation of all the supportive care required by patients affected by COVID-19 and therefore buliding an accurate predictive model is of a great importance and relevance. 

With that in mind, the aim of this assignment is to build a model classifier, trained on chest X-ray images of patients diagnosed with COVID-19, Viral pneumonia and control patients without any medical condition. A construction of an accurate predictive model would enable us to predict the presence of the infection and differentiate between the COVID-19 disease and other viral infections, such as e.g., Viral pneumonia. 
Ultimately, with this assignment we will address the research question:

> Can COVID-19 infection be detected as well as differentiated from Viral pneumonia using only X-ray scan images?



## The method

To solve this multiple classification problem, we used a pretrained convolutional neural network model: VGG16, known for its significantly better performance compared to the previous generations of classification models. In this assignment, the default complex model architecture was enriched with additional dropout layers in order to prevent potential overfitting of the model. 

Upon importing and reshaping the image data into a numpy array object we proceeded with constructing the model. First of all, we have explicitly disabled training of the convolutional layers to use the already existing weights of the model and prevent the model overfitting to the data. Then we added new classification layers: a flattening layer, a dense layer with the 'relu' activation function, and lastly the output classification layer with the 'softmax' activation function with 3 possible diagnostic outcomes: Viral Pneumonia, Covid and Normal. The model was then compiled using 'Adam' as an optimizer and 'categorical crossentropy' as the loss function parameter. Lastly, we trained the model on the data with the number of epochs set to 10 and a batch size of 128.


## Results and evaluation
The weighted accuracy of the VGG16 model (f1-score) was estimated to very high 94% running with the default parameters (a detailed classification report with the results can be found in the 'out' folder in the GitHub repository). Manipulating the size of the resized images did not seem to improve the results significantly

Covid_TrainingLossAndAccuracy.jpg (see the graph below) illustrates the training loss and accuracy of the model evolving over 10 epochs. The graphs representing the training and validation accuracy start to align after approximately 2 epochs which indicated no issue of overfitting to the data, which is a very desirable outcome, and which was our intention when adding the dropout layers to the model. The training loss graphs indicate the same tendency. What it means is that the model performs exceptionally well at predicting the diagnosis. However, the relatively small amount of data used in the study was definitely a limitation and should be taken into account when interpreting the results. For further research, including other categories of respiratory diseases such asthma or lung cancer in the model could make the results more informative.

## Repository structure and files
This repository has the following directory structure:

| Column | Description|
|--------|:-----------|
```out``` | Contains the outputs from running the script.
```covid19.py```| The script to be executed from the terminal.
```create_visual_venv.sh``` | A bash script which automatically generates a new virtual environment 'Covid_env', and install all the packages contained within 'requirements.txt'
```requirements.txt``` | A list of packages along with the versions that are required.
```README.md``` | This readme file.
```data```| A folder with images of chest X-rays scans.


## Usage (reproducing the results)

### Virtual environment
In order to run the script, one is required to set up the virtual environment with all necessary packages installed. Please clone the repo, navigate to the folder for this assignment, run the bash script to set up the environment, and lastly activate it. The following code should be executed from the terminal:

```bash
git clone https://github.com/askesvane/VisualAnalytics.git
cd assignment6_Covid19
bash ./create_visual_venv.sh
source ./Covid_env/bin/activate
```

### Execute the script 
Now, the script can be executed. You can specify the height (-a) and width (-b) of the resized images. In both cases the default is 32. Additionally, you can specify the number of epochs (-e) with a default of 10 and the test text_size (-s) with a default of 0.25. 

```bash
python covid19.py -a 32 -b 32 -e 10 -s 0.25 
```
While running, status updates will be printed to the terminal. Afterwards, the classification report and plots can be found in the folder called 'out'. It takes approximately 3 minutes.