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
