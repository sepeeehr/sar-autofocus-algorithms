import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.io import loadmat
from _tdbp import TDBP as TDBP_fast

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
            delay = np.array(2*distance/3e8)
            pixel_value = np.interp(delay, time_vector, received_signal[:, m])
            reconstructed_image[k] = reconstructed_image[k] + pixel_value*np.exp(1j*2*np.pi*fc*delay)
    return reconstructed_image

def autofocus_entropy(*, mimo_images, iteration_number, norm='Tikhonov', gamma=1e-6, eta = 1e3):
    # to does: 1- imaging domain as input, 2- M and N should not be hardcoded 3- wavelength should be defined as input of the function
    M = 41
    N = 64
    wavelength = 3e8/79e9
    delta_R = np.zeros((3, M)) # Error matrix
    gradient_R = np.zeros((3,M)) # Error matrix gradient
    gradient_F = np.zeros((3,M)) # Regularizer matrix gradient
    #autofocused_image = np.sum(mimo_images, axis=-1)    
    radar_location_x = np.linspace(-5, -4, 41)
    radar_location_y = np.zeros(41)
    radar_location_z = np.zeros(41)
    pbar = tqdm(range(iteration_number)) 
     
    X = np.resize(imaging_points[0, :], (N, N))
    Y = np.resize(imaging_points[1, :], (N, N))
    Z = np.resize(imaging_points[2, :], (N, N))
    
    # Precompute k tensor
    kx = np.zeros((N, N, M))
    ky = np.zeros((N, N, M))
    kz = np.zeros((N, N, M))

    for m in range(M):
        dx = X - radar_location_x[m]
        dy = Y - radar_location_y[m]
        dz = Z - radar_location_z[m]

        R  = np.sqrt(dx**2 + dy**2 +dz**2)

        kx[:, :, m] = 4*np.pi/wavelength * dx/R
        ky[:, :, m] = 4*np.pi/wavelength * dy/R
        kz[:, :, m] = 4*np.pi/wavelength * dz/R

    #breakpoint()


    for iter in pbar:
        autofocused_image = np.zeros((64, 64), dtype=complex)

        # Step 1: Reconstruct compensated mimo-SAR image
        for m in range(M):
                
            compensation_matrix = np.exp(-1j*kx[:, :, m]*delta_R[0, m] - 1j*ky[:, :, m]*delta_R[1, m] - 1j*kz[:, :, m]*delta_R[2, m])
             
            autofocused_image = autofocused_image + mimo_images[:, :, m]*compensation_matrix      

        if iter % 10 == 0:
            pbar.set_description(f"Entropy is: {calculate_entropy(autofocused_image)}")
        
        # Step 2: Precompute common terms
        C = np.sum(np.abs(autofocused_image)**2)                
        common_term1 = np.log(np.abs(autofocused_image)**2/C) + 1
        common_term2 = np.sum(np.abs(autofocused_image)**2 * common_term1)
        
        # Step 3: Update delta_R matrix
        for m in range(M):

            compensation_matrix = np.exp(-1j*kx[:, :, m]*delta_R[0, m] - 1j*ky[:, :, m]*delta_R[1, m] - 1j*kz[:, :, m]*delta_R[2, m])
            common_term3 = np.imag(np.conj(autofocused_image)*mimo_images[:, :, m]*compensation_matrix)

            gradient_R[0, m] = -1/C*np.sum(common_term1*2*kx[:, :, m]*common_term3) + 1/C**2 * common_term2 * np.sum(2*kx[:, :, m]*common_term3)
            gradient_R[1, m] = -1/C*np.sum(common_term1*2*ky[:, :, m]*common_term3) + 1/C**2 * common_term2 * np.sum(2*ky[:, :, m]*common_term3)
            gradient_R[2, m] = -1/C*np.sum(common_term1*2*kz[:, :, m]*common_term3) + 1/C**2 * common_term2 * np.sum(2*kz[:, :, m]*common_term3)
            
        if norm == 'Tikhonov':
            gradient_F = 2*delta_R
        delta_R = delta_R - gamma*(gradient_R + eta*gradient_F) 
              
    return autofocused_image

def calculate_entropy(image):
    """Calculate image entropy"""
    C = np.sum(np.abs(image)**2)
    entropy = -np.sum(np.abs(image)**2/C*np.log(np.abs(image)**2/C))
    return entropy
 
if __name__ == '__main__':
    import time
    np.random.default_rng()
    
    fc = 79e9
    fs = 5e9
    data = loadmat('data.mat')
    
    received_signal = data['received_signal']
    radar_location_nominal = data['radar_location_nominal']
    radar_location_actual = data['radar_location_actual']
    x_error = np.random.random((1, 41))
    print('x_error is: {x_error}')
    imaging_points = data['imaging_points']

    print(f"radar_location_nominal.shape is: {radar_location_nominal.shape}")
    print(f"radar_location_nominal[:, 0:8:] is: {radar_location_nominal[0, 0::8]}")

    reconstructed_image_nominal = TDBP_fast(received_signal, radar_location_nominal, imaging_points, fc, fs)
    reconstructed_image_nominal.resize(64, 64)

    reconstructed_image_actual = TDBP_fast(received_signal, radar_location_actual, imaging_points, fc, fs)
    reconstructed_image_actual.resize(64, 64)

    #plt.imshow(np.abs(reconstructed_image_fast))
    #plt.show()

    
    #breakpoint()
    
    mimo_images = np.zeros((64, 64, 41), dtype=complex) 
    for i in range(41):
        reconstructed_image = TDBP_fast(received_signal[:, (8*i):(8*(i+1))], radar_location_nominal[:, (8*i):(8*(i+1))], imaging_points, fc, fs)
        mimo_images[:, :, i] = np.resize(reconstructed_image, (64, 64))
    
    autofocused_image = autofocus_entropy(mimo_images = mimo_images, iteration_number = 1000, norm='Tikhonov', gamma=1e-7, eta = 1e4)

    
    fig, axes = plt.subplots(1, 3)

    axes[0].imshow(np.abs(reconstructed_image_nominal))
    axes[0].set_title(f"Nominal Trajectory - Entropy: {calculate_entropy(reconstructed_image_nominal):.2f}:")

    axes[1].imshow(np.abs(reconstructed_image_actual))
    axes[1].set_title(f"Actual Trajectory - Entropy: {calculate_entropy(reconstructed_image_actual):.2f}")

    #plt.tight_layout()
    #plt.show()
    
    
    axes[2].imshow(np.abs(autofocused_image))
    axes[2].set_title(f"Autofocused - Entropy: {calculate_entropy(autofocused_image):.2f}")
    plt.tight_layout()
    plt.show()
