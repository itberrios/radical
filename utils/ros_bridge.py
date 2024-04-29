# ref: 
#     https://publish.illinois.edu/radicaldata/
#     https://databank.illinois.edu/datasets/IDB-3289560

import numpy as np

def rosmsg_to_radarcube_v1(rosmsg, config, reformat=True):
    data = np.array(rosmsg.data)
    data = data.reshape(-1, 8)  #4 real followed by 4 imaginary => 8 numbers
    data = data[:, :4] + 1j * data[:, 4:]
    data = data.reshape(-1)
    data = data.reshape(
        config['numChirps'], config['profiles'][0]['adcSamples'], config['numLanes'])
    data = np.moveaxis(data, 1, 2)

    if reformat:
        if len(config['chirps']) ==2:
            data = np.concatenate((data[0::2, ...], data[1::2, ...]), axis=1)
        elif len(config['chirps']) ==3:
            data = np.concatenate((data[:, 0::3, ...], data[:, 1::3, ...], data[:, 2::3, ...]), axis=1)
        elif len(config['chirps']) == 4:  # assume chirp-a, chirp-b, chirp-a, chirp-b
            data = np.concatenate((data[0::2, ...], data[1::2, ...]), axis=1)

    return data
