#!/usr/bin/env python

#A real-time implementation which converts the image from ASUS/Kinect to mat file using cv_bridge and ROS openni. The mat file is segmented and regressed (roughness) using the trained caffe model.
from __future__ import print_function

import sys
import cv2
import time
import math
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

# Make sure that caffe is on the python path:
caffe_root = '/home/viveksuryamurthy/Vivek/code/caffe/caffe-segnet-cudnn5-v2/caffe-segnet-cudnn5/'
sys.path.insert(0, caffe_root + 'python') 

import caffe
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class image_converter:
    def __init__(self):
        #Only 4 ms
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/kinect2/qhd/image_color_rect",Image,self.callback, queue_size=10) 
        

    def callback(self,data): 
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)
        self.segmentation(cv_image)

    def colour_label(self, im):

        #Colours
        Sand = [255,255,0] #yellow
        Wood = [204,102,0] #orange
        Stone = [0,0,204] #blue
        Metal = [0,153,153] #greenish blue
        Road = [76,0,153] #violet
        Grass = [102,204,0] #green
        Unlabelled = [0,0,0]

        #Labelling
        r = im.copy()
        g = im.copy()
        b = im.copy()

        label_colours = np.array([Sand, Wood, Stone, Metal, Road, Grass, Unlabelled])
        for l in range(0,6):
            r[im==l] = label_colours[l,0]
            g[im==l] = label_colours[l,1]
            b[im==l] = label_colours[l,2]

            rgb = np.zeros((im.shape[0], im.shape[1], 3))
            rgb[:,:,0] = r/255.0
            rgb[:,:,1] = g/255.0
            rgb[:,:,2] = b/255.0
        return rgb

    def segmentation(self, frame):

        #start = time.time()
        caffe.set_device(1)  # if we have multiple GPUs, pick the first one
        caffe.set_mode_gpu()
    
        ##Input and output dimension
        input_shape = net.blobs['data'].data.shape
        output_shape = net.blobs['argmax'].data.shape
    
        ##Preprocessing]
        frame = frame[348:540,280:664, :]
        frame = cv2.resize(frame, (input_shape[3],input_shape[2]))
        frame_1 = frame[...,::-1]


        #The image is swapped according to caffe (CxHxW) from image format (HxWxC)
        input_image = frame.transpose((2,0,1))
        input_image = np.asarray([input_image])
        

        #Run the model
        start = time.time()
        out = net.forward_all(data=input_image)
        output_image = out['argmax'][0,0]
        out_rough = out['conv1_1_D_R'][0,0]
        #end = time.time()
        #print('%s' % 'Executed SegNet in ', str((end - start)*1000), 'ms')

        ##Softmax layer to determine the elements with low confidence. Argmax does not provide information regarding the confidence.
        softmax = np.max(out['softmax'][0],axis=0)
        low_confidence = np.where(softmax < 0.4)
        output_image[low_confidence] = 6

        #Get the colour
        output_coloured = self.colour_label(output_image)
        
        plt.ion()
        ax1 =plt.subplot(311)
        im1=ax1.imshow(frame_1)
        #plt.gca().set_position([0, 0, 1, 1])
        plt.tight_layout()
        plt.axis('off')

        ax2 = plt.subplot(312)
        im2 = ax2.imshow(output_coloured)
        plt.tight_layout()
        plt.axis('off')

        ax3 = plt.subplot(311)
        im3 = ax3.imshow(out_rough, cmap='hot', interpolation='nearest', vmin=0.0, vmax=1.6)
        #plt.colorbar()
        plt.tight_layout()
        plt.axis('off')
        plt.subplots_adjust(wspace=0, hspace=0.005)
        
        plt.pause(0.00001)
        plt.show()
        plt.ioff()

        #end = time.time()
        #print('%s' % 'Executed SegNet in ', str((end - start)*1000), 'ms')
        ax1.cla()
        ax2.cla()
        ax3.cla()

def main(args):

    #if we have multiple GPUs, pick the first one
    caffe.set_device(1)  
    caffe.set_mode_gpu()   

    #Define and load the model 
    global net
    net = caffe.Net("/home/viveksuryamurthy/Vivek/code/caffe/caffe-segnet-cudnn5-v2/caffe-segnet-cudnn5/models/Segnet/Final_experiment_model/final_model.prototxt", 
                 "/home/viveksuryamurthy/Vivek/code/caffe/caffe-segnet-cudnn5-v2/caffe-segnet-cudnn5/models/Segnet/Final_experiment_model/final_weights.caffemodel", caffe.TEST) 

    #Subscribe to the Kinect
    im = image_converter()
    rospy.init_node('image_converter', anonymous=True)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
