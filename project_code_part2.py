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
                               xy_stride=(2*(i+1),2*(i+1)))
    all_windows.append(window_list)
    print(i,y_start,y_stop,xwindow,ywindow)
windows = [item for sublist in all_windows for item in sublist]




