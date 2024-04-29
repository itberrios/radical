"""
Range Processing for Radical dataset
"""

import numpy as np


def process_range(adc_data, use_window=False):
    """ Processes range information from raw Radar data cube by taking the FFT over the
        Range Bins, this is also called the Fast Time FFT
        Inputs:
            adc_data - raw radar data cube obtained in each Coherent Processing Interval (CPI)
                (num_chirps_per_frame, num_rx_antennas, num_adc_samples)
            use_window - determines whether to window data prior to FFT
        Outputs:
            range_cube - processed Radar Cube (ADC data) with range information
                (rang bins, num virtual antennas, num chirps per frame)
    """
    # get window for FFT
    window = 1
    if use_window:
        window = np.hamming(adc_data.shape[2])

    # perform FFT over accumulated range bins (ADC samples)
    range_cube = np.fft.fft(window*adc_data, axis=2).transpose(2, 1, 0)

    return range_cube

