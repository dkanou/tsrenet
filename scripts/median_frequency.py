import numpy as np
import os
from scipy.misc import imread
from skimage.transform import resize
import ast

image_dir_1 = "/home/viveksuryamurthy/Downloads/ADE20K_2016_07_26/ADE20-Training-labels"
image_dir_2 = "/home/viveksuryamurthy/Downloads/ADE20K_2016_07_26/Road-labels-1"
image_dir_3 = "/home/viveksuryamurthy/Vivek/Dataset/opensurface/Opensurface-wood-metal-labels/Training"

image_files_1 = [os.path.join(image_dir_1, file) for file in os.listdir(image_dir_1) if file.endswith('.png')]
image_files_2 = [os.path.join(image_dir_2, file) for file in os.listdir(image_dir_2) if file.endswith('.png')]
image_files_3 = [os.path.join(image_dir_3, file) for file in os.listdir(image_dir_3) if file.endswith('.png')]
image_files_4 = [os.path.join(image_dir_4, file) for file in os.listdir(image_dir_4) if file.endswith('.png')]
image_files =  image_files_1+image_files_2+image_files_3

def median_frequency_balancing(image_files=image_files, num_classes=4):
    '''
    Perform median frequency balancing on the image files, given by the formula:
    f = Median_freq_c / total_freq_c
    where median_freq_c is the median frequency of the class for all pixels of C that appeared in images
    and total_freq_c is the total number of pixels of c in the total pixels of the images where c appeared.
    INPUTS:
    - image_files(list): a list of image_filenames which element can be read immediately
    - num_classes(int): the number of classes of pixels in all images
    OUTPUTS:
    - class_weights(list): a list of class weights where each index represents each class label and the element is the class weight for that label.
    '''
    #Initialize all the labels key with a list value
    label_to_frequency_dict = {}
    for i in xrange(num_classes):
        label_to_frequency_dict[i] = []

    for n in xrange(len(image_files)):
        image = imread(image_files[n])
        image = resize(image, (256,512), order=3)*255
        #print image
        #For each image sum up the frequency of each label in that image and append to the dictionary if frequency is positive.
        for i in xrange(num_classes):
            class_mask = np.equal(image, i)
            class_mask = class_mask.astype(np.float32)
            class_frequency = np.sum(class_mask)

            if class_frequency != 0.0:
                label_to_frequency_dict[i].append(class_frequency)

    class_weights = []
    
    #Get the total pixels to calculate total_frequency later
    total_pixels = 0
    for frequencies in label_to_frequency_dict.values():
        total_pixels += sum(frequencies)

    for i, j in label_to_frequency_dict.items():
        j = sorted(j) #To obtain the median, we got to sort the frequencies
        median_frequency = np.median(j) / sum(j)
        total_frequency = sum(j) / total_pixels
        #print total_pixels
        median_frequency_balanced = median_frequency / total_frequency
        class_weights.append(median_frequency_balanced)

    #Set the last class_weight to 0.0 as it's the background class
    class_weights[-1] = 0.0

    return class_weights

if __name__ == "__main__":
    a = median_frequency_balancing(image_files, num_classes=7)
    print a
#ENet_weighing(image_files, num_classes=12)
