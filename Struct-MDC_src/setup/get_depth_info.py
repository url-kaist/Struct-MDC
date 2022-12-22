'''
Author: Alex Wong <alexw@cs.ucla.edu>

If you use this code, please cite the following paper:

A. Wong, and S. Soatto. Unsupervised Depth Completion with Calibrated Backprojection Layers.
https://arxiv.org/pdf/2108.10531.pdf

@inproceedings{wong2021unsupervised,
  title={Unsupervised Depth Completion with Calibrated Backprojection Layers},
  author={Wong, Alex and Soatto, Stefano},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={12747--12756},
  year={2021}
}
'''
import os, sys, glob, argparse
import multiprocessing as mp
from tkinter import E
import numpy as np
import cv2
sys.path.insert(0, 'src')
import data_utils
from PIL import Image

from liegroups import SE3, SO3


dataset_name = 'KITTI_v1'
skip = 1
unused_img = 0


MAX_DEPTH = 79.0 # for KITTI

train_sparse_depth_paths = 'training/kitti/kitti_train_sparse_depth_150.txt'
val_sparse_depth_paths   = 'testing/kitti/kitti_test_sparse_depth_150.txt'


train_sparse_depth_paths = data_utils.read_paths(train_sparse_depth_paths)
val_sparse_depth_paths   = data_utils.read_paths(val_sparse_depth_paths)

min_depth = +float('inf')
max_depth = -float('inf')



def load_depth(path, data_format='HW'):
    '''
    Loads a depth map from a 16-bit PNG file

    Arg(s):
        path : str
            path to 16-bit PNG file
        data_format : str
            HW, CHW, HWC
    Returns:
        numpy[float32] : depth map
    '''


    # Loads depth map from 16-bit PNG file
    z = np.array(Image.open(path), dtype=np.float32)


    # Assert 16-bit (not 8-bit) depth map
    z = z / 256.0
    z[z <= 0] = 0.0

    # clip
    z = np.clip(z, 0.0, MAX_DEPTH)
    

    return z





## 1) search in training dataset
for e in train_sparse_depth_paths:

    s_depth = load_depth(e)


    cur_min, cur_max = np.min(s_depth), np.max(s_depth) 

    min_depth = min(cur_min, min_depth)
    max_depth = max(cur_max, max_depth)


    tmp = s_depth[s_depth > 200]




    # DEBUG print
    print('current minimum: %.4f' %(min_depth))
    print('current Maximum: %.4f' %(max_depth))




## 2) search in validation dataset
for e in val_sparse_depth_paths:

    s_depth = load_depth(e)


    cur_min, cur_max = np.min(s_depth), np.max(s_depth) 

    min_depth = min(cur_min, min_depth)
    max_depth = max(cur_max, max_depth)







    # DEBUG print
    print('current minimum: %.4f' %(min_depth))
    print('current Maximum: %.4f' %(max_depth))

