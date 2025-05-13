import numpy as np
from scipy import signal

def compute_power(data, fs):
    f, Pxx_den = signal.periodogram(data, fs) # data: (n_channels, n_timepoints) will take periodogram on last axis by default

    f_thres = [4, 8, 13, 30, 50, 70, 90, 110, 128]
    poses = []
    for fi in range(len(f_thres) - 1):
        cond1_pos = np.where(f_thres[fi] < f)[0]
        cond2_pos = np.where(f_thres[fi + 1] >= f)[0]
        poses.append(np.intersect1d(cond1_pos, cond2_pos))

    ori_shape = Pxx_den.shape[:-1]
    Pxx_den = Pxx_den.reshape(-1, len(f))
    band_sum = [np.sum(Pxx_den[:, band_pos], axis=-1) + 1 for band_pos in poses]
    band_sum = [np.log10(_band_sum)[:, np.newaxis] for _band_sum in band_sum]
    band_sum = np.concatenate(band_sum, axis=-1)
    ori_shape += (8,)
    band_sum = band_sum.reshape(ori_shape)

    return band_sum