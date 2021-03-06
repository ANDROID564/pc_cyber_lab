# -*- coding: utf-8 -*-
"""
Created on Thu Aug 30 18:28:45 2018

@author: acer
"""

######## Video Object Detection Using Tensorflow-trained Classifier #########
#
# Author: Evan Juras
# Date: 1/16/18
# Description: 
# This program uses a TensorFlow-trained classifier to perform object detection.
# It loads the classifier uses it to perform object detection on a video.
# It draws boxes and scores around the objects of interest in each frame
# of the video.

## Some of the code is copied from Google's example at
## https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb

## and some is copied from Dat Tran's example at
## https://github.com/datitran/object_detector_app/blob/master/object_detection_app.py

## but I changed it to make it more understandable to me.

# Import packages
import os
clear = lambda: os.system('cls')
clear()


import cv2
import numpy as np
import tensorflow as tf
import sys
from test_frames1 import analyse_frame
from termcolor import colored
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style


style.use('fivethirtyeight')


from PIL import Image
import matplotlib.pyplot as plt
from os.path import isfile, join

image_array=[] 

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util
#from drawnow import drawnow
# Name of the directory containing the object detection module we're using
MODEL_NAME = 'inference_graph'
VIDEO_NAME = '11.avi'

# Grab path to current working directory
CWD_PATH = os.getcwd()

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,'training','mscoco_label_map.pbtxt')

# Path to video
PATH_TO_VIDEO = os.path.join(CWD_PATH,VIDEO_NAME)

# Number of classes the object detector can identify
NUM_CLASSES = 6


#take frames from video_file

try:
    if not os.path.exists('z1video_frames4'):
        os.makedirs('z1video_frames4')
except OSError:
    print ('Error: Creating directory of data')


# Load the label map.
# Label maps map indices to category names, so that when our convolution
# network predicts `5`, we know that this corresponds to `king`.
# Here we use internal utility functions, but anything that returns a
# dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)

# Define input and output tensors (i.e. data) for the object detection classifier

# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

# Open video file
video = cv2.VideoCapture(PATH_TO_VIDEO)
currentFrame = 0
i=0
j=0
while(video.isOpened()):

    # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
    # i.e. a single-column array, where each item in the column has the pixel RGB value
    ret, frame = video.read()
    if ret:
            
        resize = cv2.resize(frame, (320, 240), interpolation = cv2.INTER_LINEAR)
        frame_expanded = np.expand_dims(frame, axis=0)
    
        # Perform the actual detection by running the model with the image as input
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: frame_expanded})
    
        # Draw the results of the detection (aka 'visulaize the results')
        vis_util.visualize_boxes_and_labels_on_image_array(
            frame,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=4,
            min_score_thresh=0.40)
    
        # All the results have been drawn on the frame, so it's time to display it.
        
        #cv2.imshow('Object detector', frame)
        
        #****************************************************
        # Saves image of the current frame in jpg file
        name = './z1video_frames4/frame' + str(currentFrame) + '.jpg'
    
        print ('Creating...' + name)
        #print("image pixel value",frame)
            
        
        cv2.imwrite(name, frame)
        
        image1=analyse_frame.frame_new1(frame)
        if (image1=='NULL'):
            print('if-main')
            j=j+1
            print(j)
            #continue
        else:
            print('else-main')
            print("the center is",image1)
            image_array.append(image1)
            
            image_list=list(image1)
            x10,y10=image_list
            print('image1-x',x10)
            print("image_list",image_list)
            
            j=j+1
            print("the centre points are",image_array)
            
            
            #xs1 = [x[0] for x in image_array]
            #ys1 = [x[1] for x in image_array]
            
        
        cv2.imshow('Object detector', frame)
            
        # To stop duplicate images
        currentFrame += 1
    
        #****************************************************
        
        # Press 'q' to quit
        if cv2.waitKey(1) == ord('q'):
            break

    else:
        break
    
# Clean up
print(image_array)      
# naming the x axis
plt.xlabel('x - axis')
# naming the y axis
plt.ylabel('y - axis')
# giving a title to my graph
plt.title('Activity detection of a person')


xs = [x[0] for x in image_array]
ys = [x[1] for x in image_array]

xs_max=max(xs)
ys_max=max(ys)

print('xs_max',xs_max)


#print(xs)
#print(ys[2])
j=1
y_dif=[]
x_dif=[]
for i in range(len(xs)):
    if(j<len(xs)):
        #print(j)
        y=ys[j]-ys[i]
        y_dif.append(abs(y))
        
        x=xs[j]-xs[i]
        x_dif.append(abs(x))
        
    j=j+1
print("person action y-axis",y_dif)
print("person action x-axis",x_dif)

if(abs(ys[1]-ys[0])<7):
    print("person in horizontal motion")
    for i in range(len(x_dif)):
        if(x_dif[i]>7):
            print (colored("person running","red"))
        elif(x_dif[i]>2 and x_dif[i]<5 ):
            print(colored("person walking",'blue'))
        elif(x_dif[i]>4 and x_dif[i]<8 ):
            print(colored("person jogging",'green'))
        #else:
         #   print("person in vertical motion")
 
elif(abs(xs[1]-xs[0])<2):
    print("person in vertical  motion")
    for i in range(len(y_dif)):
        if(y_dif[i]>6):
            print("climbing")
        elif(y_dif[i]>2 and y_dif[i]<6 ):
            print("slow climb")
        else:
            print("very slow climb")
 
    
'''    
x1,x2,y1,y2 = plt.axis()
#x1=x_min,x2=x_max,y_min=25,y_max=250 
plt.axis((x1,x2,25,250))
plt.axis([0, 100, 0, 200])

plt.xlabel('xs')
plt.ylabel('ys')

plt.axis([0, 1000, 0, 1000])

'''
#plotting of graph

#plt.xlim(0, xs_max)
plt.xlim(0, 600)
#plt.ylim(0, ys_max)
plt.ylim(0, 600)
plt.plot(xs, ys)
#plt.show()

#interactive graph

    
            #cv2.imshow("imag",images)
            #img1=analyse_frame.frame_new1(cv2.imread(k))
            
        
'''
img1=analyse_frame.frame_new1(cv2.imread('z1video_frames2/frame75.jpg'))
            #print(cv2.imread(k))
cv2.imshow("imag",img1)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''



video.release()
cv2.destroyAllWindows()



