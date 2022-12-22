######################################
#     modified by Jinwoo Jeon        #
######################################
'''
Author: Jinwoo Jeon <zinuok@kaist.ac.kr>

If you use this code, please cite the following paper:
J. Jeon, H. Lim, D. Seo, H. Myung. "Struct-MDC: Mesh-Refined Unsupervised Depth Completion Leveraging Structural Regularities from Visual SLAM"

@article{jeon2022struct,
  title={Struct-MDC: Mesh-Refined Unsupervised Depth Completion Leveraging Structural Regularities From Visual SLAM},
  author={Jeon, Jinwoo and Lim, Hyunjun and Seo, Dong-Uk and Myung, Hyun},
  journal={IEEE Robotics and Automation Letters},
  volume={7},
  number={3},
  pages={6391--6398},
  year={2022},
  publisher={IEEE}
}
'''

######################################
#     original License info.         #
######################################
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
import os
import torch
import numpy as np
from matplotlib import pyplot as plt


def log(s, filepath=None, to_console=True):
    '''
    Logs a string to either file or console

    Arg(s):
        s : str
            string to log
        filepath
            output filepath for logging
        to_console : bool
            log to console
    '''

    if to_console:
        print(s)

    if filepath is not None:
        if not os.path.isdir(os.path.dirname(filepath)):
            os.makedirs(os.path.dirname(filepath))
            with open(filepath, 'w+') as o:
               o.write(s + '\n')
        else:
            with open(filepath, 'a+') as o:
                o.write(s + '\n')

def colorize(T, colormap='magma'):
    '''
    Colorizes a 1-channel tensor with matplotlib colormaps

    Arg(s):
        T : torch.Tensor[float32]
            1-channel tensor
        colormap : str
            matplotlib colormap
    '''

    cm = plt.cm.get_cmap(colormap)
    shape = T.shape

    # Convert to numpy array and transpose
    if shape[0] > 1:
        T = np.squeeze(np.transpose(T.cpu().numpy(), (0, 2, 3, 1)))
    else:
        T = np.squeeze(np.transpose(T.cpu().numpy(), (0, 2, 3, 1)), axis=-1)

    # Colorize using colormap and transpose back
    color = np.concatenate([
        np.expand_dims(cm(T[n, ...])[..., 0:3], 0) for n in range(T.shape[0])],
        axis=0)
    color = np.transpose(color, (0, 3, 1, 2))

    # Convert back to tensor
    return torch.from_numpy(color.astype(np.float32))
