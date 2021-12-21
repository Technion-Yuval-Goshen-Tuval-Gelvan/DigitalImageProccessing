import cv2
import matplotlib.pyplot as plt
import numpy as np
import glob
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift
import random
import scipy.io

def load_blurred_images():
    blurred_images = []
    filenames = glob.glob('blurred/*.jpg')
    filenames.sort()
    for filename in filenames:
        im = cv2.imread(filename, cv2.IMREAD_GRAYSCALE) / 255.0
        blurred_images.append(im)
        random.shuffle(blurred_images)
    return blurred_images


def images_to_dft(images):
    blurred_images_fft = []
    for image in images:
        blurred_images_fft.append(fftshift(fft2(image, shape=FFT_SIZE)))
    return blurred_images_fft


def compute_fft_weights(images_fft, p=11):
    # smoothing filter
    images_fft_smoothed = cv2.GaussianBlur(np.abs(images_fft), (11, 11), 1)

    fft_array_p = np.power(images_fft_smoothed, p)

    weights_sum = np.sum(fft_array_p, axis=0)
    weights = np.divide(fft_array_p, weights_sum)

    return weights


def get_reconstructed_fft(images_fft, weights):
    weighted_fft = np.multiply(images_fft, weights)
    return np.sum(weighted_fft, axis=0)


def psnr(im1, im2):
    mse = np.mean(np.power(np.subtract(im1, im2), 2))
    PIXEL_MAX = 1.0
    return 20 * np.log10(PIXEL_MAX) - 10 * np.log10(mse)


original_image = cv2.imread('DIPSourceHW1.jpg', cv2.IMREAD_GRAYSCALE) / 255.0
image_size = original_image.shape
FFT_SIZE = image_size

blurred_images = load_blurred_images()
blurred_images_fft = images_to_dft(blurred_images)
blurred_images_fft = np.stack(blurred_images_fft)

# reconstruct from different number of images:
p = 9
psnr_list = []
for num_images in range(1, 101):
    curr_images_fft = blurred_images_fft[0:num_images]

    fft_weights = compute_fft_weights(curr_images_fft, p=p)
    reconstructed_fft = get_reconstructed_fft(curr_images_fft, fft_weights)
    reconstructed_image = ifft2(ifftshift(reconstructed_fft), shape=FFT_SIZE).real

    cv2.imwrite(f"deblurred/{num_images}_images_deblur.jpg", reconstructed_image * 255)

    psnr_list.append(psnr(original_image, reconstructed_image))

scipy.io.savemat('psnr_values.mat', mdict={'psnr': psnr_list})

plt.plot(list(range(1, 101, 5)), psnr_list[::5])
plt.title('PSNR vs. Number of Images')
plt.xlabel('Number of Images')
plt.ylabel('PSNR [dB]')
plt.savefig('psnr_graph.png')
plt.show()
