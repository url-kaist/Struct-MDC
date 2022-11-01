#!/usr/bin/env python3
#-*- encoding: utf-8 -*-



# Matplotlib Tri
#import triangle as tr
#from matplotlib.tri import Triangulation, TriAnalyzer, LinearTriInterpolator
#import matplotlib.pyplot as plt

import cv2
import scipy.interpolate
from scipy.spatial import Delaunay

# etc
from multiprocessing import Process, cpu_count, Manager, Pool
import numpy as np
from numpy.core.umath_tests import inner1d
import time
from math import *
import sys
import csv
import pandas as pd

import argparse
import os
import glob




#####################################################################
# Interpolate the mesh depth,
# using image coordinates of each mesh vertex.
#####################################################################





eps = 1e-10
total_cnt = 0
total_time = 0

# v: shape [2,]
# p: shape [M,N]
def in_triangle(v1, v2, v3, p):

    M, N = p.shape

    inside = False

    # Compute vectors
    vec0 = np.repeat(np.expand_dims(v3-v1,0), M, axis=0)
    vec1 = np.repeat(np.expand_dims(v2-v1,0), M, axis=0)
    vec2 = p  - np.repeat(np.expand_dims(v1,0), M, axis=0)


    # Compute dot products
    dot00 = inner1d(vec0,vec0)
    dot01 = inner1d(vec0,vec1)
    dot02 = inner1d(vec0,vec2)
    dot11 = inner1d(vec1,vec1)
    dot12 = inner1d(vec1,vec2)



    # Compute barycentric coordinates
    invDenom = 1 / (dot00 * dot11 - dot01 * dot01)
    u = (dot11 * dot02 - dot01 * dot12) * invDenom
    v = (dot00 * dot12 - dot01 * dot02) * invDenom



    # Check if point is in triangle
    mask_1 = u >= 0
    mask_2 = v >= 0
    tmp = u+v
    mask_3 = tmp <= 1 # '=' for including three tri points also.

    return mask_1 * mask_2 * mask_3


def backproject_to_camera(depth, uv_pts, intrinsic):
    '''
    Backprojects pixel coordinates to 3D camera coordinates
    '''
    
    # make homog. [u,v] -> [u,v,1]
    num_pt, _ = uv_pts.shape
    one = np.ones(num_pt)
    one = np.expand_dims(one, -1)
    
    uv_homog = np.concatenate([uv_pts, one], axis=1)
    uv_homog = np.expand_dims(uv_homog, -1)
    
    # back-project to 3D cam
    K_inv = np.linalg.inv(intrinsic)
    K_inv = np.tile(K_inv, (num_pt,1,1))
    xyz = np.matmul(K_inv, uv_homog)
    xyz = xyz[:,:,0]
    
    # multiply depth
    depth_ = np.expand_dims(depth, -1)
    xyz = xyz * depth_
    
    return xyz
    
  
def depth_interpolation_new(row, col, points, vals, sparse_depth):

    new_pts = np.array([points[:,1], points[:,0]]).T

    grids = tuple(np.mgrid[0:row, 0:col])

    depths = scipy.interpolate.griddata(new_pts, vals, grids, rescale=False)
    depth_map = depths.astype(np.uint16)


    # boundary depth (line, point feature)
    np.copyto(depth_map, sparse_depth, where=sparse_depth!=0)



    return depth_map










####################################################################################
USE_MULTI_THREAD = False

def generate(ROW, COL, mesh_xy, sparse_depth, intrinsic, filename):

    start_time = time.time()
    assert (len(mesh_xy) % 3 == 0)

    depth_map = np.zeros((ROW,COL), dtype=float)

    ##########################################
    if USE_MULTI_THREAD:   # slow ....

        pass
#        threads = cpu_count()
#        manager = Manager()
#        return_dict = manager.dict()

#        num_tri = len(xy)//3
#        p = num_tri // threads

#        procs = []
#        for i in range(threads):
#            if i == threads-1:
#                xy_par = xy[3*p*i:,:]
#                z_par  = z[3*p*i:]
#            else:
#                xy_par = xy[3*p*i:3*p*(i+1),:]
#                z_par  = z[3*p*i:3*p*(i+1)]

#            proc = Process(target=depth_interpolation, args=(xy_par,z_par,seq,  i,return_dict))
#            procs.append(proc)
#            proc.start()

#        for proc in procs:
#            proc.join()

#        pts = np.array(return_dict.values(), dtype=object)
#        pts = np.vstack(pts[:]).astype(np.float)
    ##########################################


    ##########################################
    else:


        # depth at each vertex of mesh
        mesh_xy = mesh_xy.astype(np.int64)
        mesh_depth = sparse_depth[mesh_xy[:,1], mesh_xy[:,0]]


        # interp_depth, interp_normV = depth_interpolation_new(ROW, COL, mesh_xy, mesh_depth, sparse_depth, intrinsic)
        interp_depth = depth_interpolation_new(ROW, COL, mesh_xy, mesh_depth, sparse_depth)

        # save as png (depth image)
        pos = filename.find('mesh_uv')
        end_pos = pos + len('mesh_uv')
        f_name = filename[end_pos:filename.find('.csv')]
        out_name = filename[:pos] + 'mesh_depth' + f_name + '.png'
        cv2.imwrite(out_name, interp_depth)
        
        # save as numpy
        # interp_normV = np.array(interp_normV, dtype=np.float32) # compressed
        # np.save(filename[:pos] + 'mesh_normal' + f_name, interp_normV)
        

    ##########################################





def generate_from_mesh_coord(seq_name):


    # open all csv files
    directory = os.path.join(seq_name, 'mesh_uv/')


    files = glob.glob(directory+'*.csv')
    total = len(files)

    total_cnt = 0
    total_time = 0


    for i, file_name in enumerate(files):

        # 1) get mesh uv coordinates
        mesh_uv = np.genfromtxt(file_name,delimiter=',') #[:,:]

        # 2) get sparse depth data
        pos = file_name.find('mesh_uv')
        end_pos = pos + len('mesh_uv')
        f_name = file_name[end_pos:file_name.find('.csv')]

        depth_name = file_name[:pos] + 'sparse_depth' + f_name + '.png'

        # print(depth_name)
        sparse_depth = cv2.imread(depth_name, -1)

        r, c = sparse_depth.shape
        
        # 3) get intrinsic matrix
        intrinsic = np.loadtxt(file_name[:pos]+'K.txt', delimiter = ' ')



        # print(mesh_uv.shape, mesh_uv.ndim)
        if mesh_uv.ndim < 2:
            print('err: ' + str(file_name))
            print(mesh_uv.shape, mesh_uv.ndim)
            assert 0
        # print(mesh_uv[:,0].shape)

        mesh_u = mesh_uv[:,0]
        mesh_u = np.expand_dims(np.clip(mesh_u, 0, c-1), -1)
        mesh_v = mesh_uv[:,1]
        mesh_v = np.expand_dims(np.clip(mesh_v, 0, r-1), -1)
        mesh_uv = np.concatenate([mesh_u, mesh_v], axis=1)

        mesh_uv = np.around(mesh_uv).astype('int')


        # 4) depth interpolation
        # print(str(file_name) + ' start.')
        tic = time.time()
        generate(r, c, mesh_uv, sparse_depth, intrinsic, file_name)
        elapsed = time.time()-tic

        ## print INFO
        total_cnt += 1
        total_time += elapsed
        print('[%d/%d] Process time: %f ms (avg: %f ms)' %(i+1,total,  elapsed*1000, (total_time/total_cnt)*1000))
        # print(str(file_name) + ' completed.')




if __name__ == '__main__':

    # read parameters
    parser = argparse.ArgumentParser()


    parser.add_argument("--root_path",     type=str, default="/home/zinuok/Dataset")
    parser.add_argument("--dataset_name",  type=str, default="my_data")

    parser.add_argument('--test',
        type=bool, required=False, help='Do testing for DEBUG purpose?')
    
    args = parser.parse_args()

    # get data path: containing all sequences
    path      = os.path.join(args.root_path, args.dataset_name)
    data_path = os.path.join(path, 'data')

    print("data_path: %s" %(data_path))

    if args.test != None:
        sequences = []
        seq = 'test'
        path = os.path.join(data_path, 'void_150')
        path = os.path.join(path, 'data')
        seq = os.path.join(path, seq)
        
        print(seq)
        
        # multiprocess per sequence
        pool = Pool(processes=1)
        sequences.append(seq)
        pool.map(generate_from_mesh_coord, sequences)
        pool.close()
        pool.join()
        
        assert 0
        




    # get all sequence names
    sequences = []
    f = open(data_path.replace('data', 'sequences.txt'), 'r')
    while True:
        line = f.readline()

        if not line: break
        seq = line.strip('\n') #[:-1]
        seq = os.path.join(data_path, seq)

        target = os.path.join(seq, 'image')
        cur    = os.path.join(seq, 'mesh_depth')
        target = sum(len(files) for _, _, files in os.walk(target))
        cur = sum(len(files) for _, _, files in os.walk(cur))

        if cur < target:
            sequences.append(seq)

    # sequences = ['/home/zinuok/Dataset/PLAD_v2/data/test1', 
    #              '/home/zinuok/Dataset/PLAD_v2/data/test2', 
    #              '/home/zinuok/Dataset/PLAD_v2/data/test3', 
    #              '/home/zinuok/Dataset/PLAD_v2/data/test4']

    print('-'*10)
    for e in sequences:
        print(e)
    print('-'*10)
    print(len(sequences))
    # assert 0
    
    # sequences=['/home/zinuok/Dataset/void_parsed_line/void_150/data/office0']
    
    # for seq in sequences:
    #     generate_from_mesh_coord(seq)

    # multiprocess per sequence
    pool = Pool(processes=8)
    pool.map(generate_from_mesh_coord, sequences)
    pool.close()
    pool.join()


