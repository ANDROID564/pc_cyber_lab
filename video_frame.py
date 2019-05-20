'''
Using OpenCV takes a mp4 video and produces a number of images.

Requirements
----
You require OpenCV 3.2 to be installed.

Run
----
Open the main.py and edit the path to the video. Then run:
$ python main.py

Which will produce a folder called data with the images. There will be 2000+ images for example.mp4.
'''
import cv2
import numpy as np
import os

# Playing video from file:
cap = cv2.VideoCapture('2.avi')

try:
    if not os.path.exists('data1'):
        os.makedirs('data1')
except OSError:
    print ('Error: Creating directory of data')

currentFrame = 0
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret:
            
    
        # Saves image of the current frame in jpg file
        name = './data1/frame' + str(currentFrame) + '.jpg'
        print ('Creating...' + name)
        cv2.imwrite(name, frame)
    
        # To stop duplicate images
        currentFrame += 1
        cv2.imshow('Video', frame)
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
    
    else:
        break
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()