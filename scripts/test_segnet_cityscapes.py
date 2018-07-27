#!/usr/bin/env python
#Shows the segmentation output for six classes. 


import sys
import cv2
# Make sure that caffe is on the python path:
caffe_root = '/home/viveksuryamurthy/Vivek/code/caffe/caffe-segnet-cudnn5-v2/caffe-segnet-cudnn5/'
sys.path.insert(0, caffe_root + 'python') 

import numpy as np
import caffe
import matplotlib.pyplot as plt
from PIL import Image

def colour_label(im):
#Converts the labels to colours for visualization purpose
    #Colours
    Sand = [255,255,0] #yellow
    Wood = [204,102,0] #orange
    Stone = [0,0,204] #blue
    Metal = [0,153,153] #greenish blue
    Road = [76,0,153] #violet
    Grass = [102,204,0] #green
    Unlabelled = [0,0,0]

    #Labelling
    r = (im.copy()).astype(np.float32)
    g = (im.copy()).astype(np.float32)
    b = (im.copy()).astype(np.float32)

    label_colours = np.array([Sand, Wood, Stone, Metal, Road, Grass, Unlabelled])
    for l in range(0,7):
        r[im==l] = label_colours[l,0]
        g[im==l] = label_colours[l,1]
        b[im==l] = label_colours[l,2]

        rgb = np.zeros((im.shape[0], im.shape[1], 3))
        
        #rgb[:,:,0] = r/255.0
       # rgb[:,:,1] = g/255.0
        #rgb[:,:,2] = b/255.0

#for cv imwrite
        rgb[:,:,2] = r#/255
        rgb[:,:,1] = g#/255
        rgb[:,:,0] = b#/255
    return rgb

if __name__ == '__main__':
    caffe.set_device(1)  # if we have multiple GPUs, pick the first one
    caffe.set_mode_gpu()

##load the model
    net = caffe.Net("/home/viveksuryamurthy/Vivek/code/caffe/caffe-segnet-cudnn5-v2/caffe-segnet-cudnn5/models/Segnet/bn_conv_merged_model.prototxt",
       "bn_conv_merged_weights.caffemodel", 
                  caffe.TEST)

##Transformer
#Transforms the image to the required input size
    transformer = caffe.io.Transformer({'data':net.blobs['data'].data.shape})
#The image is swapped according to caffe (CxHxW) from image format (HxWxC)
    transformer.set_transpose('data',(2,0,1))
#RGB to BGR
    transformer.set_channel_swap('data', (2,1,0))
#caffe.io loads the normalized image but the network operates in 0-255.
    transformer.set_raw_scale('data', 255.0)
#This is required only when the batch size needs to be varied
#net.blobs['data'].reshape(1,3,100,100)

##Load image
    I = caffe.io.load_image('/home/viveksuryamurthy/Downloads/sample_3.jpeg')
    #I = I[720:1040,0:320, :]
#Preprocessing
    net.blobs['data'].data[...] = transformer.preprocess('data',I)

##Running the model
    out = net.forward()
    output = out['argmax'][0,0]  
  
## Softmax layer to determine the elements with low confidence. Argmax does not provide information regarding the confidence.
    softmax = np.max(out['softmax'][0],axis=0)
   
#Pixels with lesser confidence than 0.4 are changed to unlabelled.
    low_confidence = np.where(softmax < 0.4)
    output[low_confidence] = 6

#Get the colour
    output_coloured = colour_label(output)

##Post-processing
    cv2.imshow("output_crop.jpg", output_coloured)
    #cv2.imwrite("video_4_seg_full_finetune.jpg", output_coloured)
    cv2.waitKey(10000)


