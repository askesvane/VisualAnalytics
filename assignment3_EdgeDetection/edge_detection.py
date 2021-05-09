#!/usr/bin/env python

#_______________# Import packages #_______________# 

# tools
import os
import sys
sys.path.append(os.path.join(".."))
import argparse
import numpy as np

# Image tool
import cv2

#_______________# Functions #_______________# 

# Rectangle on image function
def ROI(image, upper_left, bottom_right, filename):
    """
    The function takes an image, upper left corner, bottom right corner, 
    and the filename of the output as parameters. The corners are used to draw a green rectangle on the image.
    The image will be saved to the folder 'img' along with a message printed in the terminal.
    """
    # additional paramters
    color_green = (0, 255, 0)
    thickness = 5

    # draw rectangle on the picture
    image_with_ROI = cv2.rectangle(image.copy(), upper_left, bottom_right, color_green, thickness)

    # create a filename with img/ path
    filename_ROI = f'img/{filename}'

    # Using cv2.imwrite() method to save the image ans print message to terminal
    cv2.imwrite(filename_ROI, image_with_ROI)
    msg_print(filename)

def crop_me(image, upper_left, bottom_right, filename):
    """
    The function takes an image, upper left corner, bottom right corner, 
    and the filename of the output as parameters.
    An image will be saved to the folder 'img' along with a message printed in the terminal.
    """
    # Cropped the image based on the defined rectangle properties
    image_cropped = image[upper_left[1]: bottom_right[1], upper_left[0]:bottom_right[0]]

    # create a filename with img/ path
    filename_crop = f'img/{filename}'

    # Using cv2.imwrite() method to save the image 
    cv2.imwrite(filename_crop, image_cropped)
    msg_print(filename)
    
    return(image_cropped)

# Edge_detecter function
def edge_detecter(image_cropped, filename):    
    """
    The function is using canny edge detection to 'find' every letter in the image. 
    They will we drawn as green contours on the cropped image.
    The function takes 2 parameters, the cropped image and the filename of the output.
    """
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
    filename_letters = f'img/{filename}'

    # Using cv2.imwrite() method to save the image + print message in the terminal.
    cv2.imwrite(filename_letters, image_letters)
    msg_print(filename)
    

# Printer function
def msg_print(filename):
    # Takes filename and print status message to the terminal
    print(f"The image '{filename}' has been saved in the folder 'img'.")


#_______________# MAIN FUNCTION #_______________#

def main(args):

    # import image from the 'img' folder
    fname = os.path.join("img","We_Hold_These_Truths.JPG")
    image = cv2.imread(fname)

    # take parameters defined in the command line.
    upper_left = (args["width_left"], args["height_upper"])
    bottom_right = (args["width_right"], args["height_bottom"])
    
    # Run ROI function
    ROI(image, upper_left, bottom_right, "image_with_ROI.jpg")
    
    # Run crop_me function
    image_cropped = crop_me(image, upper_left, bottom_right, "image_cropped.jpg")

    # Run edge_detecter function
    edge_detecter(image_cropped, "image_letters.jpg")



#_______________# RUN MAIN #_______________#

if __name__=="__main__":
    
    # Argument parser
    ap = argparse.ArgumentParser()
    ap.add_argument("-a", "--width_left", default = 1385, type = int, help = "Rectangle corners. Width left. Default is 1385")
    ap.add_argument("-b", "--height_upper", default = 880, type = int, help = "Rectangle corners. Height upper. Default is 880")
    ap.add_argument("-c", "--width_right", default = 2890, type = int, help = "Rectangle corners. Width right. Default is 2890")
    ap.add_argument("-d", "--height_bottom", default = 2800, type = int, help = "Rectangle corners. Height bottom. Default is 2800")
                    
    # Parse arguments. args is now an object containing all arguments added through the terminal. 
    argument_parser = vars(ap.parse_args())
    
    main(argument_parser)