import cv2
from datetime import datetime
from PIL import Image

#video input
"""cap = cv2.VideoCapture('C:/Users/mdf30/src/CarND-Vehicle-Detection/project_video.mp4')
ret, frame = cap.read()
while ret:
    #vid_pic = projected_path(frame,mtx=mtx,dist=dist,src=src,dst=dst)
    #Save the pic in the correct folder with the correct timestamp
    time = datetime.now().strftime('%Y_%m_%d_%H_%M_%S_%f')
    im = Image.fromarray(frame)
    im.save('video_input/'+time+'.jpg')
    ret, frame = cap.read()
"""
#test video input
cap = cv2.VideoCapture('C:/Users/mdf30/src/CarND-Vehicle-Detection/test_video.mp4')
ret, frame = cap.read()
while ret:
    #vid_pic = projected_path(frame,mtx=mtx,dist=dist,src=src,dst=dst)
    #Save the pic in the correct folder with the correct timestamp
    time = datetime.now().strftime('%Y_%m_%d_%H_%M_%S_%f')
    im = Image.fromarray(frame)
    im.save('test_video_input/'+time+'.jpg')
    ret, frame = cap.read()
