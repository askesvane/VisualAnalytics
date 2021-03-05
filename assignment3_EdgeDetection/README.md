# Assignment 3 - Edge detection

## Assignment description

The purpose of this assignment is to use computer vision to extract specific features from images. In particular, we're going to see if we can find text. We are not interested in finding whole words right now; we'll look at how to find whole words in a coming class. For now, we only want to find language-like objects, such as letters and punctuation.

Download and save the image at the link below:
https://upload.wikimedia.org/wikipedia/commons/f/f4/%22We_Hold_These_Truths%22_at_Jefferson_Memorial_IMG_4729.JPG

Using the skills you have learned up to now, do the following tasks:

- Draw a green rectangular box to show a region of interest (ROI) around the main body of text in the middle of the image. Save this as image_with_ROI.jpg.
- Crop the original image to create a new image containing only the ROI in the rectangle. Save this as image_cropped.jpg.
- Using this cropped image, use Canny edge detection to 'find' every letter in the image
- Draw a green contour around each letter in the cropped image. Save this as image_letters.jpg

## Peer-review instructions

- Clone the whole repository to a chosen location on your computer by executing ```git clone https://github.com/askesvane/VisualAnalytics.git``` in the terminal.
- Through the terminal, navigate to the folder for assignment 3 by ```cd assignment3_EdgeDetection```
- Set up the virtual environment by executing ```bash create_visual_venv.sh``` in the terminal. A virtual environment called 'edgedetection_environment' should now appear in the folder. The virtual environment automatically installs all required packages provided in the 'requirements.txt' file.
- Activate the virtual environment by executing ```./source edgedetection_environment/bin/activate``` in the terminal. You should now see the virtual environment in parentheses at the commandline.
- Run the python script 'edge_detection.py' by executing ```python edge_detection.py```.
- If you navigate to the folder 'img' you will find the 3 generated images.




















