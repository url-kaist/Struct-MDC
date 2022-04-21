######################################
#         Visualization tool         #
######################################
# AUTHOR LICENSE INFO.
'''
Author: Jinwoo Jeon <zinuok@kaist.ac.kr>

If you use this code, please cite the following paper:

J. Jeon, H. Lim, D. Seo, H. Myung. "Struct-MDC: Mesh-Refined Unsupervised Depth Completion Leveraging Structural Regularities from Visual SLAM"
'''

# DESCRIPTIONS
'''
- RGB + SparseDepth visualization.
- SparseDepth is magnified just for visual purpose.
'''



from matplotlib.pylab import plt
import matplotlib
import numpy as np
import numpy.ma as ma
from PIL import Image
import imageio


## CONFIGURATIONS
#############################
seq_num = '0000000593'
dataset_type = 'nyu'
mode = 'line'

mag = 2 # magnifying constant
#############################








if mode == 'line':
    alg = 'my'
if mode == 'point':
    alg = 'kbnet'

# VOID
if dataset_type == 'void':
    rgb     = imageio.imread('void/kbnet/outputs/image/'+seq_num+'.png')
    s_depth = matplotlib.pyplot.imread('nyu/sparse_depth/'+seq_num+'.png')

# NYUv2
if dataset_type == 'nyu':
    rgb     = imageio.imread('nyu/image/'+seq_num+'.png')
    s_depth = matplotlib.pyplot.imread('nyu/'+alg+'/outputs/sparse_depth/'+seq_num+'.png')

# PLAD
if dataset_type == 'plad':
    rgb     = imageio.imread('plad/my/outputs/image/'+seq_num+'.png')
    s_depth = matplotlib.pyplot.imread('plad/my/outputs/sparse_depth/'+seq_num+'.png')
######################### DATA PATH #########################

H, W = s_depth.shape


# magnify
for m in range(1,mag+1):
    
    zero_vert = np.zeros((H,m))
    zero_horz = np.zeros((m,W))


    # vertical magnifying
    tmp1 = np.concatenate([s_depth[m:,:],zero_horz], axis=0)
    tmp2 = np.concatenate([zero_horz,s_depth[:-m,:]], axis=0)
    np.copyto(s_depth, tmp1, where=s_depth==0)
    np.copyto(s_depth, tmp2, where=s_depth==0)


    # horizontal magnifying
    tmp3 = np.concatenate([s_depth[:,m:],zero_vert], axis=1)
    tmp4 = np.concatenate([zero_vert,s_depth[:,:-m]], axis=1)
    np.copyto(s_depth, tmp3, where=s_depth==0)
    np.copyto(s_depth, tmp4, where=s_depth==0)



plt.imsave('depth_map.png', s_depth, cmap=plt.cm.jet)


# merge RGB + depth map
depth = imageio.imread('depth_map.png')[:,:,:3]
depth_r = depth[:,:,0]
depth_g = depth[:,:,1]
depth_b = depth[:,:,2]


tmp = np.ones((H,W)) * 127
b_ = depth_b - tmp


# make transfer mask
rg_ = np.logical_or(depth_r,depth_g)
rgb_ = np.logical_or(rg_, b_)

rgb_r = rgb[:,:,0]
rgb_g = rgb[:,:,1]
rgb_b = rgb[:,:,2]

np.copyto(rgb_r, depth_r, where=rgb_==True)
np.copyto(rgb_g, depth_g, where=rgb_==True)
np.copyto(rgb_b, depth_b, where=rgb_==True)

rgbd_map = np.stack([rgb_r, rgb_g, rgb_b], axis=2)

# imageio.imsave('rgb'+str(seq_num)+'.png', rgb)
imageio.imsave(str(seq_num)+'_sparse_depth'+'.png', depth)
imageio.imsave(str(seq_num)+'_rgbd'+'.png', rgbd_map)
