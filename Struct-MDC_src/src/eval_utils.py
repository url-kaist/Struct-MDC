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
import numpy as np

def GetDepthPrintableRatios(valid_gt, valid_preds):

    ratios = np.maximum(valid_preds / valid_gt, valid_gt / valid_preds)
    total_num = ratios.size

    r05   = 100 * ratios[ratios < 1.05].size / (total_num + 1e-7)
    r10   = 100 * ratios[ratios < 1.10].size / (total_num + 1e-7)
    r25   = 100 * ratios[ratios < 1.25].size / (total_num + 1e-7)
    r25_2 = 100 * ratios[ratios < 1.25**2].size / (total_num + 1e-7)
    r25_3 = 100 * ratios[ratios < 1.25**3].size / (total_num + 1e-7)

    return {'D_1.05': round(r05, 1), 'D_1.10': round(r10, 1), 'D_1.25': round(r25, 1),
            'D_1.56': round(r25_2, 1), 'D_1.95': round(r25_3, 1)}



def root_mean_sq_err(src, tgt):
    '''
    Root mean squared error

    Arg(s):
        src : numpy[float32]
            source array
        tgt : numpy[float32]
            target array
    Returns:
        float : root mean squared error
    '''

    return np.sqrt(np.mean((tgt - src) ** 2))

def mean_abs_err(src, tgt):
    '''
    Mean absolute error

    Arg(s):
        src : numpy[float32]
            source array
        tgt : numpy[float32]
            target array
    Returns:
        float : mean absolute error
    '''

    return np.mean(np.abs(tgt - src))

def inv_root_mean_sq_err(src, tgt):
    '''
    Inverse root mean squared error

    Arg(s):
        src : numpy[float32]
            source array
        tgt : numpy[float32]
            target array
    Returns:
        float : inverse root mean squared error
    '''

    return np.sqrt(np.mean(((1.0 / tgt) - (1.0 / src)) ** 2))

def inv_mean_abs_err(src, tgt):
    '''
    Inverse mean absolute error

    Arg(s):
        src : numpy[float32]
            source array
        tgt : numpy[float32]
            target array
    Returns:
        float : inverse mean absolute error
    '''

    return np.mean(np.abs((1.0 / tgt) - (1.0 / src)))

def mean_abs_rel_err(src, tgt):
    '''
    Mean absolute relative error (normalize absolute error)

    Arg(s):
        src : numpy[float32]
            source array
        tgt : numpy[float32]
            target array

    Returns:
        float : mean absolute relative error between source and target
    '''

    return np.mean(np.abs(src - tgt) / tgt)
