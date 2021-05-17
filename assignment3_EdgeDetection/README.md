# Assignment 3 - Edge detection

## Assignment description

__Finding text using edge detection__

The purpose of this assignment is to use computer vision to extract specific features from images. In particular, we're going to see if we can find text. We are not interested in finding whole words right now; we only want to find language-like objects, such as letters and punctuation.

The image can be found here:
https://upload.wikimedia.org/wikipedia/commons/f/f4/%22We_Hold_These_Truths%22_at_Jefferson_Memorial_IMG_4729.JPG

Do the following tasks:
- Draw a green rectangular box to show a region of interest (ROI) around the main body of text in the middle of the image. Save this as image_with_ROI.jpg.
- Crop the original image to create a new image containing only the ROI in the rectangle. Save this as image_cropped.jpg.
- Using this cropped image, use Canny edge detection to 'find' every letter in the image
- Draw a green contour around each letter in the cropped image. Save this as image_letters.jpg

## The method
To solve this task, I have created 3 function: one function draws a green rectangle on the image, one function crops the image, and one function draws green contours around every text object (letters, puncturations etc.).

To draw draw the green contours, I have used openCV to change the image into greyscale and subsequently blurred the image (gaussian) with a 5x5 kernel. I found the letters with canny edge detection defining the lower and upper boundaries as 75 and 140. I used openCV's findContours() and drawContours() to apply green contours around the letters.

## Results and evaluation
The output images can be found in the folder 'img'. I successfully managed to draw a rectangle as well as crop the image. I managed to capture every letter in the image and draw green contours around them. The three images are saved in the folder 'img' as 'image_with_ROI.jpg', 'image_cropped.jpg', and 'image_letters.jpg'. However, I did not manage to exclude everything but the letters because some darker fissures were within the threshold to be detected. By increasing the threshold to overcome this problem, I would augment the risk of not detecting less pronounced letters.

## Repository structure and files
This repository has the following directory structure:

| Column | Description|
|--------|:-----------|
```img``` | Folder containing the image to be processed as well as the three output images from running the script.
```create_visual_venv.sh``` | A bash script which automatically generates a new virtual environment 'edgedetection_env', and install all the packages contained within 'requirements.txt'
```edge_detection.py```| The script to be executed from the terminal.
```README.md``` | This readme file.
```requirements.txt``` | A list of packages along with the versions that are required.

## Usage (reproducing the results)

### Virtual environment
In order to run the script, one is required to set up the virtual environment with all necessary packages installed. Please clone the repo, navigate to the folder for this assignment, run the bash script to set up the environment, and lastly activate it. The following code should be executed from the terminal:

```bash
git clone https://github.com/askesvane/VisualAnalytics.git
cd assignment3_EdgeDetection
bash ./create_visual_venv.sh
source ./edgedetection_env/bin/activate
``` 

### Execute the script 
Now, the script can be executed. You can specify the left width, the upper height, right width, and bottom height of the rectangle around the letters. The defaults are 1385, 880, 2890, and 2800.
```bash
python edge_detection.py --width_left 1385 --height_upper 880 --width_right 2890 --height_bottom 2800 
```
The outputs will be saved in 'img'.


