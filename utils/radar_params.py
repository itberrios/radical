"""
Lightweight script to process Radar parameters from config file for Radical Dataset
"""

LIGHT_SPEED = 299792458 # m/s

def compute_range_resolution(adc_sample_rate, num_adc_samples, chirp_slope):
    """
        Compute Range Resolution in meters
        Inputs:
            adc_sample_rate (Msps)
            num_adc_samples (unitless)
            chirp_slope (Mz/usec)
        Outputs:
            range_resolution (meters)
    """

    # compute ADC sample period T_c in msec
    adc_sample_period = 1 / adc_sample_rate * num_adc_samples # msec

    # next compute the Bandwidth in GHz
    bandwidth = adc_sample_period * chirp_slope # GHz

    # Coompute range resolution in meters
    range_resolution = LIGHT_SPEED / (2 * (bandwidth * 1e9)) # meters

    return range_resolution, bandwidth


def compute_doppler_resolution(num_chirps, bandwidth, chirp_interval, num_tx):
    """
        Compute Doppler Resolution in meters/second
        Inputs:
            num_chirps
            bandwidth - bandwidth of each chirp (GHz)
            chirp_interval - total interval of a chirp including idle time (usec)
            num_tx
        Outputs:
            doppler_resolution (ms)
    """
    # compute center frequency in GHz
    center_freq = (77 + bandwidth/2) # GHz

    # compute center wavelength 
    lmbda = LIGHT_SPEED/(center_freq * 1e9) # meters

    # compute doppler resolution in meters/second
    doppler_resolution = lmbda / (2 * num_chirps * num_tx * chirp_interval)

    return doppler_resolution