import numpy as np
import scipy.signal as signal


def resample(data, orig_sr, target_sr):
    ratio = orig_sr / target_sr
    nums = int(len(data) / ratio)
    if len(data.shape) > 1:
        data = np.mean(data, axis=1, dtype=data.dtype)
    sampled = signal.resample(data, nums)
    return sampled