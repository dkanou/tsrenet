#!/usr/bin/env python
#Compute the test accuracy for test set and outputs the label and prediction .png files for computing metrics

import sys

# import cv2
# Make sure that caffe is on the python path:
caffe_root = '/home/viveksuryamurthy/Vivek/code/caffe/caffe-segnet-cudnn5-v2/caffe-segnet-cudnn5/'
sys.path.insert(0, caffe_root + 'python') 

import numpy as np
import caffe
import matplotlib.pyplot as plt
from PIL import Image

caffe.set_device(0)  # if we have multiple GPUs, pick the first one
caffe.set_mode_gpu()
 
net = caffe.Net("/home/viveksuryamurthy/Vivek/code/caffe/caffe-segnet-cudnn5-v2/caffe-segnet-cudnn5/models/Segnet/net_bn_statistics.prototxt",      # defines the structure of the model
                "webdemo.caffemodel",
                    caffe.TEST)

# We get the number of classes by the size of the vector of the first prediction
numClasses = 6

batch_size=5

# num_batches depends on the number of Images you will test and the batch_size defined on test.prototxt
num_batches = 100 #nImages/batch_size

# Initialize the accuracy reported by the net
acc = 0

for t in range(num_batches):
    # Run the net for the current image batch
    out = net.forward()
    for j in range(batch_size): 
        predict = out['argmax'][j,0].astype(np.uint8)
        label = net.blobs['label'].data[j,0].astype(np.uint8)
        im1 = Image.fromarray(predict)
        lab1 = Image.fromarray(label)
        name_file_pred = "pred"+str(t)+"-"+str(j)+".png"
        name_file_label = "lab"+str(t)+"-"+str(j)+".png"
        im1.save("./Predictions/"+name_file_pred)
        lab1.save("./Labels/"+name_file_label) 
    # Update accuracy with the average accuracy of the current batch
    #acc = acc+net.blobs['accuracy'].data
    #print acc
        
#acc = acc/num_batches
#print acc
