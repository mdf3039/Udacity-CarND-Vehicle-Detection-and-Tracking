##Writeup Template
###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images1/output_image1.png
[image2]: ./output_images1/output_image2.png
[image3]: ./output_images1/output_image3.png
[image4]: ./output_images1/output_image4.png
[image5]: ./output_images1/output_image5.png
[image6]: ./output_images1/output_image6.png
[video1]: ./final_video_output.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the first 66 lines of the .py file. I did not use HOG to extract features. Since all of the previous lessons have illustrated how effective convolutional neural networks can be, I decided to use CNNs. I read in all images from the 'vehicles' and 'non-vehicles' classes.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

Similar to the Behavioral Cloning project, I then grayscaled the image. I also obtain a one hot encoding oof the two labels. Here is an example of the grayscaled image:

![alt text][image2]

####2. Explain how you settled on your final choice of HOG parameters.

I knew from the moment I read the project description that I wanted to attempt a Deep Learning approach instead of the suggested Machine Learning approach. After reading about some of the ways other have attempted a Deep Learning approach, I set out on my own. I did not like the YOLO atchitecture, as it would seem to lead to possible undetections if many objects are in the same frame. I am also not quite sure if I fully understood it. I chose to use the NVIDIA architecture, from the Behavioral Cloning project, and the architecture used in my Traffic Sign Classifier project, which fed unrefined features.

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

Training the CNN was done in lines 41-66. To train the network, I began by using a modeling achitecture similar to the NVIDIA I used in the Behavioral Cloning project. The model first normalizes by subtacting 128, then dividing 128. Next, 5 convolutional layers with filters of 5x5, 5x5, 5x5, 3x3, and 3x3, sequentially, were added to the model. The number of feature maps obtained from the layers were 24, 36, 48, 64, and 64, sequentially. All of the convolutional layers were ReLU activated with 'same' border padding. A 50% dropout layer was added after the convolutional layers, then the model was flattened to be fed into the fully connected layer. Three fully connected, ReLU activated, layers were added to the model of sizes 100, 50, and 10, sequentially. A final sigmoid activated layer of size 2 was added to the model for the two categories of vehicle and non-vehicle. The model was compiled using the categorical_crossentropy loss function and the Adam optimizer. 
The model was fit using a 20% validation set and 5 epochs. Multiple values of epochs (3,4,5,6,7) were tested to observe where possible overfitting would occur. Overfitting occured after more than 5 epochs. 5 epochs gave a validation loss of 2.64%.

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

This was done in lines 71-151. I created a function that slid a window of size (a,b) across a subspace of the image, with strides (s1,s2) reflective of the x and y direction. I used only the right half of the image; when I used both halves, cars in the other lane kept getting picked up. I then created a loop: using 3 different window sizes, I obtained 3 overlapping subspaces of the bottom half of the image. I only used the bottom half of the image, because cars won't appear in the sky. The first window of size (100,80) slid across the top half of the used image space(The used image space is the bottom half of the image) with a stride of (4,4). The next window of size (100,80)*1.5 slid across the interquartile of the used image space with a stride of (4,4)*2. The next window of size (100,80)*1.5*1.5 slid across the bottom half of the used image space with a stride of (4,4)*3. I increased the window size as the sliding window lowered, because objects farther from the horizon will be bigger. Since the objects are bigger, there is less need to search as many windows, so the stride increased as well. The overlap was 50% with just 3 windows. Here is a picture of my windows. It is jumbled together, so it is difficult to see; a lot of windows are taken.

![alt text][image3]

The trained CNN is used to predict whether the window contains a vehicle. Based upon the likelihood of a vehicle in the window, I decide whether to append the window to a list of vehicle windows. I then obtain a heatmap using the vehicle windows. I added heat to the heatmap. This helps detect the entire car, when the signal is too weak on the side of the car. I got this idea from the forum. I set up 5 deques. Deques allow me to append images to a list, but removes the last image if the list gets too big. Since the heatmap from any particular image has proven to be noisy, I decided to use the 5 deques to level the noise. The last 8 images, the last 16 images, the last 25 images.  Averages of each deque were taken, then the average of those averages were taken. This produced the final heatmap, which was then thresholded to avoid weak signals. The cars were labeled and found using the thresholded final heatmap. The labels were drawn onto the image and saved.

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

I tried many things to get my pipeline to work. From each window, I obtained a confidence prediction percentage whether it is a vehicle or not. I also applied a threshold to the heatmap. The threshold removes weak heat signals from the heatmap. I also added heat to the heatmaps, which heightens the signal. It helps detect the sides of the car; my model was not capturing the entire car initially and this came as a tip from the forum. Deques were used to decrease the noise around the encapsulating window of the car. Based upon constant rerunning of the code, I found the prediction percentage at 97.5% to be the best, but it has some false positives. A prediction percentage at 99% does not have any false positives, but sometimes it fails to detect the car. The best threshold at 99% is 4 with an added heat of 2. The best threshold at 97.5% is 6 with an added heat of 2. Since the project rubric says minimal false positives, I went with the 97.5%. 

![alt text][image4]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./final_video_output.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the windows of positive detections in each frame of the video (line 273).  From the positive detections I created a heatmap and then added heat to the map, which helped with side car detections (lines 278-285). I also took an average of deque averages (lines 280-295). I applied a final threshold to the heatmap to remove false positives (line 297). I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap (line 299).  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected (line 301).  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames. Here the resulting bounding boxes are drawn onto the last frame in the series. I put it on one plot.
![alt text][image6]



---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further. 

I believe the use of CNNs instead of HOG proved to be successful. CNNs can be purposed for many similar tasks, so finding a optimal model is easier. For example, two CNNs were used to detect Traffic Signs and Behavior Cloning. Those architectures were repurposed and reused to help detect cars in this problem. What worked well was the sliding window approach. Increasing the stride with the size of the window proved to save a lot of computational time. What also worked well is the deque averaging. It helped stabilize the box. Another side effect it had was removing some false positives and helping identify the vehicle when some frames did not properly identify the vehicle. Adding heat to the heatmap helped find the sides of the car. 

The biggest problem I had was adjusting the parameters so there are no false positives while still obtaining a bounding box that fully encapsulates the car. This proved to be really difficult. I believe one of the reasons for this is most of the vehicle images used to train the model appear to be the backside of the vehicle. This is the most important part, and will usually be the most observed by a automated vehicle, but it made finding the entire car a little difficult. In hindsight, instead of searching for optimal parameters, I believe a better approach would have been to incorporate the third dataset provided in the README that contained images of traffic signals, people, cars, and trucks. This could have possibly provided a more robust CNN that better detects sides of cars. 

I also had a big problem with the noise from the rectangle drawn around the box. I found that using an average of deques instead of just one deque eliminated a lot of noise, as well as some false positives.

I do believe that some of the false positives occur because the CNN is picking up cars in the background. CNNs work well with detecting objects in images, because it can extract features from big or small images. 

My pipeline would likely fail if cars were close together. My use of labels does not include overlapping boxes. This means that when two hotspots of cars somewhat overlap, the result is a large box encapsulating both cars. Ideally, I would prefer the labels to overlap.  

What could be improved upon is a better way to detect the frame of the vehicle.  A rectangle capturing the position of the vehicle is not optimal; the vehicle placement, with respect to the camera, becomes more slanted for cars in farther lanes. A rectangle captures more than what is needed. Perhaps a better solution is a parallelogram or an oval. Also, looking back, I did not use the architecture from the Traffic Sign Classifier project, becuase I was pleased with the accuracy from the NVIDIA-based architecture. I should have compared them.

 

