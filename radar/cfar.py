"""
CFAR Processing for Radical dataset
"""

import numpy as np
from scipy.ndimage import convolve1d


def ca_cfar(x, prob_fa=0.05, num_train=8, num_gaurd=2):
    """ 
        Cell Averaging False Alarm function to obtain Dynamic Threshold 
        based on specified Probability of False Alarm. Uses 2 sided
        average of training cells to estimate the noise level.
        Inputs:
            x - input signal vector
            prob_fa - Probability of False Alarm (0-1)
            num_train - Number of training cells to estimate the noise
            num_gaurd - Number of Gaurd Cells
        Outputs:
    """
    # compute dynamic threshold scale factor
    a = num_train*(prob_fa**(-1/num_train) - 1)
    
    # get kernel to efficiently compute CFAR along signal vector
    cfar_kernel = np.ones((1 + 2*num_gaurd + 2*num_train), dtype=x.dtype) / (2*num_train)
    cfar_kernel[num_train : num_train + (2*num_gaurd) + 1] = 0.

    # compute dynamic cfar threshold
    noise_level = convolve1d(x, cfar_kernel, mode='nearest')
    threshold = a*noise_level

    return noise_level, threshold
