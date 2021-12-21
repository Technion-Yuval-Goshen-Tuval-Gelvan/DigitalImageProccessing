import scipy.io
import scipy.signal
import cv2
import matplotlib.pyplot as plt
import numpy as np
import glob
from scipy import fftpack


def load_blurred_images():
    blurred_images = []
    for filename in glob.glob('blurred/*.png'):
        im = cv2.imread(filename, cv2.IMREAD_GRAYSCALE) / 255.0
        blurred_images.append(im)
    return blurred_images


def images_to_dft(images):
    blurred_images_fft = []
    for image in images:
        blurred_images_fft.append(fftpack.fft2(fftpack.ifftshift(image)))
    return blurred_images_fft


def compute_fft_weights(images_fft, p=11):
    fft_array_p = np.power(np.abs(images_fft), p)

    weights_sum = np.sum(fft_array_p, axis=0)
    fft_weights = fft_array_p / weights_sum
    return fft_weights


def get_reconstructed_fft(images_fft, weights):
    weighted_fft = np.multiply(images_fft, weights)
    weights_sum = np.sum(weights, axis=0)
    return np.sum(weighted_fft, axis=0) / weights_sum


def psnr(im1, im2):
    mse = np.mean((im1 - im2) ** 2)
    PIXEL_MAX = 1.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))


original_image = cv2.imread('DIPSourceHW1.jpg', cv2.IMREAD_GRAYSCALE) / 255.0

blurred_images = load_blurred_images()
blurred_images_fft = images_to_dft(blurred_images)
blurred_images_fft = np.stack(blurred_images_fft)

# reconstruct from different number of images:
p = 13
for num_images in range(1, 100):
    fft_weights = compute_fft_weights(blurred_images_fft[:num_images], p=p)

    reconstructed_fft = get_reconstructed_fft(blurred_images_fft[:num_images], fft_weights)

    reconsturcted_image = np.real(fftpack.fftshift(fftpack.ifft2(reconstructed_fft)))
    reconsturcted_image = cv2.normalize(reconsturcted_image, None, 0, 1, cv2.NORM_MINMAX)
    print(f"num images = {num_images}   PSNR = {psnr(original_image, reconsturcted_image)}")

    plt.imshow(reconsturcted_image, cmap='gray')
    plt.title('num_images = ' + str(num_images))
    plt.show()
