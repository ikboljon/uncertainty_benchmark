from scipy.ndimage import gaussian_filter
import numpy as np


def add_gaussian_noise(data, sigma=0.1):
    noise = np.random.normal(sigma=sigma, size=(data.shape))
    data = data + noise
    return data

def increase_contrast(data, contrast_factor=1.5):
    data = data/data.max() * contrast_factor
    return data
    
def decrease_contrast(data, contrast_factor=0.4):
    data = data/data.max() * contrast_factor
    return data

def add_gaussian_blur(data, low=0.25, high=0.75):
    data_blur = np.zeros_like(data)
    for i in range(0, data_blur.shape[0]):
        sampl = np.random.uniform(low=0.25, high=0.75, size=(1,))
        result = gaussian_filter(data[i], sigma=sampl[0])
        data_blur[i] = result
    return data_blur