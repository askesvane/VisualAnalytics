#!/usr/bin/env python

"""
Instructions:
- Draw a green rectangular box to show a region of interest (ROI) around the main body of text in the middle of the image. Save this as image_with_ROI.jpg.

- Crop the original image to create a new image containing only the ROI in the rectangle. Save this as image_cropped.jpg.

- Using this cropped image, use Canny edge detection to 'find' every letter in the image
Draw a green contour around each letter in the cropped image. Save this as image_letters.jpg

"""

#_______________# Import packages #_______________# 

import os
import sys
sys.path.append(os.path.join(".."))
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path

#_______________# The script #_______________#

def main():
    
    # import image from the 'img' folder
    fname = os.path.join("img","We_Hold_These_Truths.JPG")
    image = cv2.imread(fname)

    # define properties
    upper_left = (1385, 880)
    bottom_right = (2890, 2800)
    color_green = (0, 255, 0)
    thickness = 5

    # draw rectangle on the picture
    image_with_ROI = cv2.rectangle(image.copy(), upper_left, bottom_right, color_green, thickness)

    # create a filename with img/ path
    filename_ROI = 'img/image_with_ROI.jpg'

    # Using cv2.imwrite() method to save the image 
    cv2.imwrite(filename_ROI, image_with_ROI)


    # Cropped the image based on the defined rectangle properties
    image_cropped = image[upper_left[1]: bottom_right[1], upper_left[0]:bottom_right[0]]

    # create a filename with img/ path
    filename_crop = 'img/image_cropped.jpg'

    # Using cv2.imwrite() method to save the image 
    cv2.imwrite(filename_crop, image_cropped)


    # Using canny edge detection to 'find' every letter in the image

    # From BGR to grey
    grey_image = cv2.cvtColor(image_cropped, cv2.COLOR_BGR2GRAY)

    # Gaussian blurring of the image
    blurred = cv2.GaussianBlur(grey_image, (5,5), 0) 

    canny = cv2.Canny(blurred, 75, 140) # the threshold has been adjusted with trial and error

    # Get contours
    (contours, _) = cv2.findContours(canny.copy(), 
                
                    # hierachy, if there is an outer structure, this one will be kept - the inner removed
                    cv2.RETR_EXTERNAL, 
                    cv2.CHAIN_APPROX_SIMPLE)

    # Create cropped image with green contours ontop
    image_letters = cv2.drawContours(image_cropped.copy(), 
                             contours, 
                             -1, # -1 is all coins, 0,1,2 etc. is one coin at a time
                             (0,255,0), # Adding a green color
                             2)
    
    # create a filename with img/ path
    filename_letters = 'img/image_letters.jpg'
    
    # Using cv2.imwrite() method to save the image 
    cv2.imwrite(filename_letters, image_letters)

    # Print text in terminal
    print("The images 'image_with_ROI.jpg', 'image_cropped.jpg', and 'image_letters.jpg' have been successfully saved in the folder 'img'.")

#_______________# The end #_______________#
    
if __name__=="__main__":
    main()
    




















