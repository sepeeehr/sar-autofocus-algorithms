import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.io import loadmat

def TDBP(received_signal, radar_location, imaging_points, fc, fs):
    '''Time-Domain Backprojection'''

    N = received_signal.shape[0] # fast time samples
    M = received_signal.shape[1] # slow time samples
    K = imaging_points.shape[1] # number of imaging points

    reconstructed_image = np.zeros(K, dtype=complex)

    time_vector = np.arange(N)/fs

    #for k in range(K):
    for k in tqdm(range(K)):
        for m in range(M):
            distance = np.linalg.norm(radar_location[:, m]-imaging_points[:, k])
            if k==0 and m==0:
                print(f'distance is: {distance}')
            delay = np.array(2*distance/3e8)
            pixel_value = np.interp(delay, time_vector, received_signal[:, m])
            if k==0 and m==0:
                print(f'pixel_value is: {pixel_value}')
            reconstructed_image[k] = reconstructed_image[k] + pixel_value*np.exp(1j*2*np.pi*fc*delay)
            if k==0 and m==0:
                print(f'reconstructed_image[k]: {reconstructed_image[k]}')
    return reconstructed_image

if __name__ == '__main__':
    M = 50
    N = 1000
    K = 10
    fc = 79e9
    fs = 5e9
    data = loadmat('data.mat')
    received_signal = data['received_signal']
    radar_location = data['radar_location']
    imaging_points = data['imaging_points']
    reconstructed_image = TDBP(received_signal, radar_location, imaging_points, fc, fs)
    reconstructed_image.resize(64, 64)
    plt.imshow(np.abs(reconstructed_image))
    plt.show()
