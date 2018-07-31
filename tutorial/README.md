INSTALLATION

1. First, clone the caffe repository from caffe-segnet-cudnn:

  $ git clone https://github.com/TimoSaemann/caffe-segnet-cudnn5.git

2. Caffe installation instructions can be found in the following links:
   https://github.com/BVLC/caffe/wiki/Ubuntu-16.04-or-15.10-Installation-Guide#the-gpu-support-prerequisites
https://github.com/BVLC/caffe/wiki/Caffe-installing-script-for-ubuntu-16.04---support-Cuda-8
Also, WITH_PYTHON_LAYER := 1

3. Note: In case there are multiple accounts in the university system, caffe might link to a different version of library (maybe installed earlier). In that case, it is important to change the following lines in the Makefile.config:

 from LIBRARY\_DIRS := \$(PYTHON\_LIB) /usr/local/lib /usr/lib \\
to LIBRARY\_DIRS := \$(PYTHON\_LIB) \textbf{./build/lib} /usr/local/lib /usr/lib

4. Uncomment  USE_PKG_CONFIG := 1 in case of issues with linking opencv libraries.

PREPARATION

1. Download the dataset. Create the .txt file using createtxt.py. Note: the absolute file path needs to be passed.
2. Download the net_1.prototxt and solver_1.prototxt which defines the model and the hyperparameters respectively. Make sure all the filepaths are correct.
3. Net_bn_statistics.prototxt is used for computing the test set accuracy and predicted label (created for convenience). Net_inference.prototxt is used for inference.
4. Median_frequency.py is used to calculate the weights for classes. At the bottom of net_1.prototxt, you can add the weights under loss param.\
5. Dense_image_data_layer.cpp and caffe.proto have been modified to add data augmentation of contrast & brightness change. Also, mirror function does both horizontal and vertical flip rather than just horizontal flip.

TRAINING & TESTING
1. The command for training(from caffe directory):

./build/tools/caffe train -gpu 1 -weights ./path_to_weights -solver ./path_to_solver/solver_1.prototxt

2. Once the training is done, run the compute_bn_statistics.py which computes the batch normalization parameter for the training set and integrates it into the model.
3. Merge_bn_layers.py integrates the bn layers into the model, thereby saving some time while execution.
4. The output model is used for computing accuracy (compute_test_accuracy.py) and testing on any image (test_segmentation.py). 
5. For class accuracy and IOU, use the matlab script compute_test_results.m.
6. For time of execution, use

./build/tools/caffe time -model ./path_to_net.prototxt/net_inference.prototxt -gpu 0 -iterations 100











