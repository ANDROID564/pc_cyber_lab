
import os
clear = lambda: os.system('cls')
clear()

import pylab
import cv2
import csv
import math
import numpy as np
import tensorflow as tf
import sys
from test_frames_main1 import analyse_frame
from termcolor import colored
from PIL import Image
import matplotlib.pyplot as plt
from os.path import isfile, join
from utils import label_map_util
from utils import visualization_utils as vis_util

def frame_capture(file):
    
        
    image_array=[] 
    sys.path.append("..")
    MODEL_NAME = 'inference_graph'
    #VIDEO_NAME = 'ground/walking/14.avi'
    #VIDEO_NAME = 'ground/crawling/1.mp4'
    #VIDEO_NAME = 'ground/climbing/1.mp4'
    #VIDEO_NAME = '2.avi'
    VIDEO_NAME = file
    line1=[(0,315),(950,315)]
    line2=[(0,345),(950,345)]
    
    
    counter=0
    #VIDEO_NAME='2.avi'
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
    
    
    
    def plot_graph():
        plt.xlabel('x - axis')
     
        plt.ylabel('y - axis')
    
        plt.title('Activity detection of a person')
            
        xs = [x[0] for x in image_array]
        ys = [x[1] for x in image_array]
        #for getting vertical point for crawling
        #print(ys[1])
        sensitive_x=365
        sensitive_y=284
        
        crawling_point=150
        j=1
        y_dif=[]
        x_dif=[]
        angle_dif=[]
        dist=[]
        y_dif1=[]
        x_dif1=[]
        dist2=[]
        
        for i in range(len(xs)):
            if(j<len(xs)):
                #print(j) 
                '''apply distance formula between two consecutive frames'''
                y=ys[j]-ys[i]
                y_dif.append(y)
                x=xs[j]-xs[i]
                x_dif.append(abs(x))
            
                dist_y=y**2
                dist_x=x**2
                dist1=abs(dist_y+dist_x)
                dist.append(dist1)
                
                '''apply distance formula between the sensitive locations'''
                y1=ys[j]-sensitive_y
                y_dif1.append(y1)
                x1=xs[j]-sensitive_x
                x_dif1.append(abs(x1))
                
                dist_y1=y1**2
                dist_x1=x1**2
                if(ys[j]>sensitive_y and xs[j]>sensitive_x):    
                    dist11=abs(dist_y1+dist_x1)
                    dist11=math.sqrt(dist11)
                    dist2.append(dist11)
                else:
                    dist11=(dist_y1+dist_x1)
                    dist11=-(math.sqrt(dist11))
                    dist2.append(dist11)                                 
                if(x1!=0):    
                    angle=(y1/x1)
                    angle_inv=100*(math.atan(angle))
                    angle_dif.append(abs(angle_inv))                
                else:
                    angle=0
                    angle_dif.append((angle))                
            j=j+1
        
    
            #print("the distances are",dist)
            #print("the x_dif values ar",x_dif)
            #print("the y_dif values ar",y_dif)
            
                 
        myFile = open('test_frames_main6f_array.csv', 'w')  
        with myFile:    
            #writer = csv.writer(myFile)
            myFields = ['X_coordinates', 'Y_coordinates',
                        'Distance','Direction_angle','Activity','id'] 
            
            writer = csv.DictWriter(myFile, fieldnames=myFields, lineterminator='\n')    
            writer.writeheader()  #used for writing the header file
            
            count_run=0
            count_walk=0
            count_jog=0
            count_stand=0
            j=1        
            for i in range(len(x_dif)):
                #hori=abs(ys[j]-ys[i])
                hori=(ys[j]-ys[i])
                #print("hori values",hori)
                verti=abs(xs[j]-xs[i])
                #print("verti values",verti)
                if(abs(hori)<6 and abs(verti)>1):
                    #print("person in horizontal motion")
                    if(line1[0][1]>=ys[j]):
                        #print("person horizontal more than 50 percent")
                        if(dist[i]>40):
                            #print(dist[i])
                            print (colored("person running","cyan"))
                            count_run=count_run+1
                            Activity="person running"
                            writer.writerow({'X_coordinates' : xs[i], 'Y_coordinates': ys[j],
                             'Distance':dist2[i],
                             'Direction_angle':angle_dif[i],'Activity':Activity,'id':1
                             })
                        elif(dist[i]>1 and dist[i]<=20 ):
                            #print(dist[i])
                            print(colored("person walking",'red'))
                            count_walk=count_walk+1
                            Activity="person walking"
                            writer.writerow({'X_coordinates' : xs[i], 'Y_coordinates': ys[j],
                             'Distance':dist2[i],
                             'Direction_angle':angle_dif[i],'Activity':Activity,'id':2
                             })                                        
                        elif(dist[i]>30 and dist[i]<=40 ):
                            #print(dist[i])
                            print(colored("person jogging",'magenta'))
                            count_jog=count_jog+1
                            Activity="person jogging"
                            writer.writerow({'X_coordinates' : xs[i], 'Y_coordinates': ys[j],
                             'Distance':dist2[i],
                             'Direction_angle':angle_dif[i],'Activity':Activity,'id':3
                             })
                        elif(dist[i]>0 and dist[i]<=1):
                            #print(dist[i])
                
                            print(colored("person standing",'green'))
                            count_stand=count_stand+1
                            Activity="person standing"
                            writer.writerow({'X_coordinates' : xs[i], 'Y_coordinates': ys[j],
                             'Distance':dist2[i],
                             'Direction_angle':angle_dif[i],'Activity':Activity,'id':4
                             })
    
                    
                    elif(line1[0][1]<ys[j]):
                        for i in range(len(x_dif)):
                            if(x_dif[i]>2):
                                print (colored("person crawling","red"))
                            '''
                            elif(x_dif[i]>2 and x_dif[i]<5 ):
                                print(colored("person crawling",'blue'))
                            elif(x_dif[i]>4 and x_dif[i]<8 ):
                                print(colored("person crawling",'green'))
                            '''     
                
                elif(abs(hori)<6 and abs(verti)<1):
                    print(colored("person standing",'green'))
                    
                    Activity="person standing"
                    writer.writerow({'X_coordinates' : xs[i], 'Y_coordinates': ys[j],
                     'Distance':dist[i],
                     'Direction_angle':angle_dif[i],'Activity':Activity,'id':4
                     })
                    
                
                elif(abs(verti)<1 and abs(hori)>2) :
                    print("person in vertical  motion")
                    for i in range(len(y_dif)):
                        if(y_dif[i]>2):
                            #print(y_dif[i])
                            print(colored("person climbing upward",'cyan'))
                            writer.writerow({'X_coordinates' : xs[i], 'Y_coordinates': ys[j],
                             'Distance':dist2[i],
                             'Direction_angle':angle_dif[i],'Activity':Activity,'id':5
                             })
                        elif(y_dif[i]<=2 and y_dif[i]>=-3):
                            #print(y_dif[i])
                            print(colored("person stationary",'magenta'))
                            writer.writerow({'X_coordinates' : xs[i], 'Y_coordinates': ys[j],
                             'Distance':dist2[i],
                             'Direction_angle':angle_dif[i],'Activity':Activity,'id':6
                             })
                        elif(y_dif[i]<=-3):
                            #print(y_dif[i])
                            print(colored("person  going downward",'yellow'))
                            writer.writerow({'X_coordinates' : xs[i], 'Y_coordinates': ys[j],
                             'Distance':dist2[i],
                             'Direction_angle':angle_dif[i],'Activity':Activity,'id':7
                             })
    
                        elif(y_dif[i]>2 and y_dif[i]<6 ):
                            print("slow climb")
                        else:
                            print("very slow climb")  
                        '''
                elif(abs(angle_inv>50) and abs(verti>=2) and abs(hori>=4)):
                    print(colored("person jumping","red"))
                    '''
                    if(abs(hori)>2):    
                        for i in range(len(y_dif)):
                            if(y_dif[i]>6):
                                print(colored("person jumping","red"))
                            elif(y_dif[i]>2 and y_dif[i]<6 ):
                                print(colored("person jumping","red"))
                            else:
                                print("very slow climb")
                '''
                elif(abs(angle_inv>50) and abs(verti>=0) and abs(hori>=4)):
                    print(colored("person swimming","red"))
                    
                    #for i in range(len(y_dif)):
                    #    if(y_dif[i]>6 ):
                    #        print(colored("person swimming","red"))
                            
                        elif(y_dif[i]>2 and y_dif[i]<6 ):
                            print(colored("person swimming","red"))
                        else:
                            print(colored("person swimming","red"))  
                        
                '''   
                    
                j=j+1
        #print("person action y-axis",y_dif)
        #print("person action x-axis",x_dif)
        #print("person angle",angle_dif)
        #myFile = open('test_frames_main6f_array.csv', 'w')  
            
        if(count_run>count_stand):
            if(count_run>count_jog):
                if(count_run>count_walk):
                    print(count_run,"The person might be running")
                    #writer.writerow({'Final_Activity_Prediction' :"The person might be running" })
                else:
                    print(count_walk,"The person might be walking")
                    #writer.writerow({'Final_Activity_Prediction' :"The person might be walking" })
            else:
                print(count_jog,"The person might be jogging")
                #writer.writerow({'Final_Activity_Prediction' :"The person might be jogging" })
        else:
            print(count_stand,"The person might be standing")
            #writer.writerow({'Final_Activity_Prediction' :"The person might be running" })
            #writer.writerow({'Final_Activity_Prediction' :"The person might be running" })
             
        #plt.xlim(0, xs_max)
        plt.xlim(0, 1000)
        #plt.ylim(0, ys_max)
        plt.ylim(0, 1000)
    
        
        plt.plot(xs, ys)
        plt.show()
    
    currentFrame = 0
    i=0
    j=0
    count=1
    while(video.isOpened()):
    
    
        ret, frame = video.read()
        cv2.line(frame,line1[0],line1[1],(0,255,255),2)
        cv2.line(frame,line2[0],line2[1],(0,255,255),2)
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
            if (image1=='NULL'):
              
                j=j+1
               
                i=i+1
            
            else:
               
                image_array.append(image1)
                #print(image_array)
                
                j=j+1
                i=i+1
                plt.show()
            
            if(i==30*count):
                plot_graph()
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
    video.release()
    cv2.destroyAllWindows()




for file in os.listdir("C:/Users/TANVEER MUSTAFA/.spyder-py3/my_filesd/coding/bounding_box/ground/running"):
    if file.endswith(".avi"):
        path=os.path.join("C:/Users/TANVEER MUSTAFA/.spyder-py3/my_filesd/coding/bounding_box/ground/running", file)
        frame_capture(path)
