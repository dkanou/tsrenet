import os, sys
import numpy as np

#create a lmdb file for images and labels extracted from pcd using python wrapper pypcd (install pypcd before running the code). The pcd files have roughness and curvature values.

caffe_root = '/home/viveksuryamurthy/Vivek/code/caffe/caffe-segnet-cudnn5-v2/caffe-segnet-cudnn5/'
sys.path.insert(0, caffe_root + 'python') 

import caffe
import lmdb
import random
from argparse import ArgumentParser
from PIL import Image
import pypcd

def make_parser():
    parser = ArgumentParser()
    parser.add_argument(
        'image_dir',
        help="Directory containing image"
    )
    parser.add_argument(
        'label_dir',
        help="Directory containing label"
    )
    return parser

def image_lmdb(image_dir, image_name):

    in_db = lmdb.open('im_roughtrain_1375_v2-lmdb', map_size=int(1e12))
    with in_db.begin(write=True) as in_txn:
        for in_idx,filenumber in enumerate(shuffle_index):
        #for filename in glob.glob(os.path.join(args.image_dir, '*.png')):
        # load image:
        # - as np.uint8 {0, ..., 255}
        # - in BGR (switch from RGB)
        # - in Channel x Height x Width order (switch from H x W x C)
            im = np.array(Image.open(os.path.join(image_dir, image_name[filenumber]))) # or load whatever ndarray you need
            im = im[:,:,::-1]
            im = im.transpose((2,0,1))
            #if im.shape[0]>3:
            #    print image_name[filenumber]
            im_dat = caffe.io.array_to_datum(im)

            in_txn.put('{:0>10d}'.format(in_idx), im_dat.SerializeToString())
    in_db.close()

def label_lmdb(label_dir, label_name):
    lab_db = lmdb.open('roughness_train_1375_v2-lmdb', map_size=int(1e12))

    with lab_db.begin(write=True) as lab_txn:
        for lab_idx,labelnumber in enumerate(shuffle_index):

            #Extract pcd using python wrapper
            pc = pypcd.PointCloud.from_path(os.path.join(label_dir, label_name[labelnumber]))

            rough = pc.pc_data['y'] 

            #pc.pcd gives a structured array. To convert it to regular array,
            lab = np.array(rough.view(np.float32).reshape(pc.height,pc.width,1)) # or load whatever ndarray you need
            #lab = lab.transpose()
            #if (np.isnan(lab)).any():
            #    print label_name[labelnumber]        
            lab = lab.transpose((2,1,0))
            lab_dat = caffe.io.array_to_datum(lab)
            lab_txn.put('{:0>10d}'.format(lab_idx), lab_dat.SerializeToString())
    lab_db.close()

if __name__ == '__main__':
    global shuffle_index
    parser = make_parser()
    args = parser.parse_args()

    #os.listdir lists the file based on how it is in the filesystem. It is always better to sort it for shuffling purposes.
    image_name = sorted([name for name in os.listdir(args.image_dir) ])
    label_name = sorted([nam for nam in os.listdir(args.label_dir) ])
    #print image_name[99], label_name[99]
    #Check if the number of images and labels are the same.
    no_of_image = len(image_name)
    no_of_label = len(label_name)
    if no_of_image == no_of_label:

        #Manual shuffling 
        shuffle_index = range(no_of_label)
        random.shuffle(shuffle_index)
        print shuffle_index

        #Create lmdb              
        image_lmdb(args.image_dir, image_name)
        label_lmdb(args.label_dir, label_name)

        
