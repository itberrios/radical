"""
Doppler Processing for Radical dataset
"""

import numpy as np


def process_doppler(range_cube):
    """ Process Doppler from Range Cube 
        Inputs:
        Outputs:
    """
    range_doppler = np.fft.fftshift(np.fft.fft(range_cube, axis=2), axes=2)

    return range_doppler