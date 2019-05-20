# -*- coding: utf-8 -*-
"""
Created on Thu Aug 30 18:28:45 2018

@author: TANVEER MUSTAFA
"""
import os
import csv
import math
clear = lambda: os.system('cls')
clear()

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import sys
from test_frames_main1 import analyse_frame
from test_frames_main3_lstm import lstm


from termcolor import colored

from PIL import Image
import matplotlib.pyplot as plt
from os.path import isfile, join



#Future prediction will be done from here
import csv

from pandas import read_csv
from pandas import datetime
from sklearn.metrics import mean_squared_error
from math import sqrt
from matplotlib import pyplot





image_array=[] 

sys.path.append("..")

from utils import label_map_util
from utils import visualization_utils as vis_util

MODEL_NAME = 'inference_graph'
VIDEO_NAME = '2.avi'

CWD_PATH = os.getcwd()

PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

PATH_TO_LABELS = os.path.join(CWD_PATH,'training','mscoco_label_map.pbtxt')

# Path to video
PATH_TO_VIDEO = os.path.join(CWD_PATH,VIDEO_NAME)

# Number of classes the object detector can identify
NUM_CLASSES = 6


try:
    if not os.path.exists('z1video_frames4'):
        os.makedirs('z1video_frames4')
except OSError:
    print ('Error: Creating directory of data')


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

image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')


detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')


num_detections = detection_graph.get_tensor_by_name('num_detections:0')


video = cv2.VideoCapture(PATH_TO_VIDEO)




def future_prediction():
    
    with open('test_frames_main6f_csv_file.csv', 'r', newline ='') as file:
        reader_file = csv.reader(file)
        value = len(list(reader_file))
        
    print("The length of csv file is",value)    
    
    #Data is splitted into train and test set
    df = pd.DataFrame(np.random.randn(value))
    msk = np.random.rand(len(df)) < 0.8
    train = df[msk]
    test = df[~msk]
    print("the test size is",len(test))    
    print("the train size is",len(train))    

    #The root mean squared method is used here
    print(colored("future will be predicted now","yellow"))
    
    
    series = read_csv('test_frames_main6f_csv_file.csv', header=0, parse_dates=[0], index_col=0, squeeze=True)
    # split data into train and test
    
    print(series)
    
   #train, test = X[0:-12], X[-12:]
    '''
    # walk-forward validation
    history = [x for x in train]
    predictions = list()
    for i in range(len(test)):
    	# make prediction
    	predictions.append(history[-1])
    	# observation
    	history.append(test[i])
    # report performance
    rmse = sqrt(mean_squared_error(test, predictions))
    print('RMSE: %.3f' % rmse)
    # line plot of observed vs predicted
    pyplot.plot(test)
    print("test values are",test)
    pyplot.plot(predictions)
    print("predicted values are",predictions)
    pyplot.show()
    '''


accuracy1=[]
def accuracy(Accuracy):
    accuracy1.append(Accuracy)
    print("accuracy&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&",accuracy1)
    
    for i in accuracy1:
        #i=(i+1)-i
        print("accuracy&&&&&&&&&&&&&&&&&&&&&&&&&&&^^^^^^",i)
        
    

def plot_graph():
    
    plt.xlabel('x - axis')
 
    plt.ylabel('y - axis')

    plt.title('Activity detection of a person')
    
    xs = [x[0] for x in image_array]
    print("the value of x_coordinates",xs)    
    ys = [x[1] for x in image_array]
    print("the value of y_coordinates",ys)
    
    
    #here the data is stored in csv file
    myFile = open('test_frames_main6f_csv_file.csv', 'w')  
        
    sensitive_x=327
    sensitive_y=285
    with myFile:    
        #writer = csv.writer(myFile)
        myFields = ['X_coordinates', 'Y_coordinates',
                    'Distance_of _sensitive_location','Accuracy','Direction_angle',
                    'Speed','Direction_X_axis'] 
        
        writer = csv.DictWriter(myFile, fieldnames=myFields, lineterminator='\n')    
        writer.writeheader()  #used for writing the header file
        
       
        
        
        for i,j in zip(xs,ys):    
            #print('i',i,'j',j)
            #The distance is found 
            Distance=math.sqrt((sensitive_x - i)**2 + (sensitive_y - j)**2)
            
            #The speed is found
            Speed=Distance/30
            
            #Angle w.r.t  between two lines
            m2=sensitive_y/j
            m1=sensitive_x/i
            m3=((m2-m1)/(1+m1*m2))
            angle=math.degrees(math.atan(m3))
            
            #Angle w.r.t coordinate axes
            coordinate_axes=abs(sensitive_y-j/sensitive_x-i)
            coordinate=math.degrees(math.atan(coordinate_axes))
            

            #Here the accuracy of hit is found            
            '''
            Accuracy=total_frames_count_hit/total_frames_count
            print("Accuracy",Accuracy)
            '''   
            writer.writerow({'X_coordinates' : i, 'Y_coordinates': j,
                             'Distance_of _sensitive_location':Distance,
                             'Accuracy':Accuracy,'Direction_angle':angle,'Speed':Speed,
                             'Direction_X_axis':coordinate})
            '''        
            for i in accuracy1:    
                writer.writerow({'Accuracy':i})
            '''
    
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
    #print("person action y-axis",y_dif)
    #print("person action x-axis",x_dif)
    
    if(abs(ys[1]-ys[0])<7):
        print("person in horizontal motion")
        for i in range(len(x_dif)):
            if(x_dif[i]>7):
                print (colored("person running","red"))
            elif(x_dif[i]>2 and x_dif[i]<5 ):
                print(colored("person walking",'blue'))
            elif(x_dif[i]>4 and x_dif[i]<8 ):
                print(colored("person jogging",'green'))  
    elif(abs(xs[1]-xs[0])<2):
        print("person in vertical  motion")
        for i in range(len(y_dif)):
            if(y_dif[i]>6):
                print("climbing")
            elif(y_dif[i]>2 and y_dif[i]<6 ):
                print("slow climb")
            else:
                print("very slow climb")  
    
    #plt.xlim(0, xs_max)
    plt.xlim(0, 1000)
    #plt.ylim(0, ys_max)
    plt.ylim(0, 1000)

    
    plt.plot(xs, ys)

currentFrame = 0
i=0
j=0
count=1
total_frames_count=0
total_frames_count_hit=0

while(video.isOpened()):


    ret, frame = video.read()
    if ret:
            
        resize = cv2.resize(frame, (320, 240), interpolation = cv2.INTER_LINEAR)
        frame_expanded = np.expand_dims(frame, axis=0)
    
       
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: frame_expanded})
    
      
        vis_util.visualize_boxes_and_labels_on_image_array(
            frame,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=4,
            min_score_thresh=0.40)
    
      
        name = './z1video_frames4/frame' + str(currentFrame) + '.jpg'
    
       
        
        cv2.imwrite(name, frame)
        image1=analyse_frame.frame_new1(frame)
        
        total_frames_count=total_frames_count+1
        
        if (image1=='NULL'):
          
            j=j+1 
           
            i=i+1
        
        else:
            #print(image1)
            image_array.append(image1)      
            j=j+1   #for next frame
            #print("image array j",j)
            i=i+1   #for counting after every 30 frames
            #print("image array i",i)
            #print("the centre points are",image_array)
            total_frames_count_hit=total_frames_count_hit+1
            Accuracy=total_frames_count_hit/total_frames_count
            plt.show()
        
        if(i==30*count):#here i is used and sice after end of loop +1 is made so it will be greater 
                        #than j like 110/30=3 so count of i will be 3 greater than j
            #plot graph after every 30 frames
            Accuracy=total_frames_count_hit/total_frames_count
            
            accuracy(Accuracy)
            plot_graph()   
            #print("Accuracy",Accuracy)
            #plot prediction after every 30 frames
            #future_prediction()
            count=count+1
            i=i+1
        
    

        cv2.imshow('Object detector', frame)
                
            # To stop duplicate images
        currentFrame += 1
        
        #****************************************************
        
        # Press 'q' to quit
        if cv2.waitKey(1) == ord('q'):
            break

    else:
        break
    
plot_graph()
    


print("length of j",j)
print((i))
print("total_frames_count_hit",total_frames_count_hit)
print("total_frames_count",total_frames_count)


prediction = pd.read_csv('test_frames_main6f_csv_file.csv')
lstm.prediction(prediction)
        
video.release()
cv2.destroyAllWindows()

