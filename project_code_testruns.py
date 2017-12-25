#Combine 2 and 3 to run
import numpy as np
#Using the trained model, iterate over patched of the image. The input images
#are of size (720,1280). Crop the top half. Now, there is a (360,1280) image.
#For the first half of the image, (0:180,1280), find the cars using 64x64
#blocks of the image. Slide a window with strides of 1 left and 1 right.
#For the middle half of the image (90:270,1280), find the cars using 128x128
#blocks of the image. For the bottom half of the image, (180:360,1280), find
#cars using 256x256 blocks of the image

#First, create a function that gathers all of the windows
def slide_window(img, x_start_stop=[0, 1280], y_start_stop=[360, 720], 
                    xy_window=(64, 64), xy_stride=(2,2)):
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]
    # Compute the span of the region to be searched    
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = xy_stride[0]
    ny_pix_per_step = xy_stride[1]
    # Compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0]-xy_stride[0])
    ny_buffer = np.int(xy_window[1]-xy_stride[1])
    nx_windows = np.int((xspan-nx_buffer)/nx_pix_per_step) 
    ny_windows = np.int((yspan-ny_buffer)/ny_pix_per_step) 
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs*nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys*ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]
            # Append window position to list
            window_list.append(((np.int(startx), np.int(starty)),
                                (np.int(endx), np.int(endy))))
    # Return the list of windows
    return window_list

#Set the number of windows
n_window_sizes = 3
#set the percentage size increase for each different window;
#With increase of 2, 64x64 becomes 128x128 then 256x256, and so on
perc_increase = 1.5
#Set the beginning size
xy_begin = (64,64)
#obtain the y_window to gather images from
y_window = [360,720]
#with the y_window and the n_window_sizes, obtain the y_size
y_size = (y_window[1]-y_window[0])/(n_window_sizes-1)
#obtain the amount of the window to overlap
y_overlap = y_size/(n_window_sizes-1)
#create a list to put all of the windows
all_windows = []
#obtain a dummy image of the size of actual images
img_dummy = np.zeros((720,1280))
#for each different window
for i in range(n_window_sizes):
    #find where the y_window starts and stops
    y_start = y_window[0]+(y_size-y_overlap)*i
    y_stop = min(y_start+y_size,img_dummy.shape[0])
    #find what the window sizes will be
    xwindow = xy_begin[0]*(perc_increase**i)
    ywindow = xy_begin[1]*(perc_increase**i)
    window_list = slide_window(img_dummy, x_start_stop=[0, 1280],
                               y_start_stop=[y_start, y_stop],
                               xy_window=(xwindow, ywindow),
                               xy_stride=(4*(i+1),4*(i+1)))
    all_windows.append(window_list)
    #print(i,y_start,y_stop,xwindow,ywindow)
windows = [item for sublist in all_windows for item in sublist]



import os
import matplotlib.pyplot as plt
import cv2
from keras.models import load_model
from PIL import Image
from scipy.ndimage.measurements import label
import numpy as np

# Define a function to draw bounding boxes
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy
#Define a function that creates a heat map
def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
    # Return updated heatmap
    return heatmap
#Define a function that thresholds the false positives
def apply_threshold(heatmap, threshold):
    #Make a copy of the image
    heatmapcopy = np.copy(heatmap)
    # Zero out pixels below the threshold
    heatmapcopy[heatmapcopy <= threshold] = 0
    # Return thresholded map
    return heatmapcopy
#Define a function that draws boxes around the cars, given the labels
def draw_labeled_bboxes(img, labels):
    # Make a copy of the image
    imcopy = np.copy(img)
    #make an empty list for collected bbox points
    bbox_list = []
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # if the box is a subbox of an existing box, then do not draw the box
        subbox = False
        for car in bbox_list:
            if car[0][0]<bbox[0][0] and car[0][1]<bbox[0][1]:
                if car[1][0]>bbox[1][0] and car[1][1]>bbox[1][1]:
                    subbox = True
        if subbox:
            continue
        #if a car in the bbox_list is a subbox of bbox, remove the entry from the list
        for index,car in enumerate(bbox_list):
            if car[0][0]>bbox[0][0] and car[0][1]>bbox[0][1]:
                if car[1][0]<bbox[1][0] and car[1][1]<bbox[1][1]:
                    del bbox_list[index]
        #Append the bbox to the list
        bbox_list.append(bbox)
    for bbox in bbox_list:
        # Draw the box on the image
        cv2.rectangle(imcopy, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return imcopy


#using the windows, fit the classifier onto each window image. If car is
#detected, append to on_windows list
#import model
model = load_model('C:/Users/mdf30/src/CarND-Vehicle-Detection/AWS_output/model_nvidia1.h5')
#read in images
input_dir = 'C:/Users/mdf30/src/CarND-Vehicle-Detection/test_images/'
output_dir = 'C:/Users/mdf30/src/CarND-Vehicle-Detection/output_images/'

#for each image
"""
for filename in os.listdir(input_dir):
    if filename.endswith('.jpg'):
        #read in the image
        test_img = plt.imread(input_dir+filename)
        #since the jpg image was read in with matplot, the format it RGB
        #change to BGR
        test_img = test_img[...,::-1].copy()
        #gray the image
        gray_img = cv2.cvtColor(test_img,cv2.COLOR_BGR2GRAY)
        #empty list for windows identified with cars
        on_windows = []
        #for each window in the windows list
        for window in windows:
            #obtain the image to be predicted on. Resize to 64,64,3
            pred_img = cv2.resize(gray_img[window[0][1]:window[1][1],
                                       window[0][0]:window[1][0]].copy(),
                                   (64, 64))
            #reshape the image
            pred_img = pred_img.reshape((1,pred_img.shape[0],pred_img.shape[1],1))
            #predict with model
            prediction = model.predict(pred_img)
            #If the prediction is equal to 1, then it is a vehicle.
            #append the window to on_windows
            if prediction[0][1] > 0.99:
                on_windows.append(window)
        boxed_img = draw_boxes(test_img,on_windows)
        output_img = Image.fromarray(boxed_img)
        output_img.save(output_dir+filename)
        #create heatmap
        heatmap = add_heat(np.zeros_like(test_img), on_windows)
        #apply threshold
        thresh_heatmap = apply_threshold(heatmap, threshold=2)
        #Find and label the cars in the image
        labels = label(thresh_heatmap)
        #Draw boxes around the cars using the labels
        final_img = draw_labeled_bboxes(test_img, labels)
        output_img = Image.fromarray(final_img)
        output_img.save(output_dir+'final_'+filename)
        break
"""
