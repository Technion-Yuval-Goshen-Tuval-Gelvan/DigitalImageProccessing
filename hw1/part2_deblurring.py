import cv2
import matplotlib.pyplot as plt
import numpy as np
import glob
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift


def load_blurred_images():
    blurred_images = []
    filenames = glob.glob('blurred/*.jpg')
    filenames.sort()
    for filename in filenames:
        im = cv2.imread(filename, cv2.IMREAD_GRAYSCALE) / 255.0
        blurred_images.append(im)
    return blurred_images


def images_to_dft(images):
    blurred_images_fft = []
    for image in images:
        blurred_images_fft.append(fftshift(fft2(image)))
    return blurred_images_fft


def compute_fft_weights(images_fft, p=11):
    # smoothing filter
    images_fft_smoothed = cv2.GaussianBlur(np.abs(images_fft), (5, 5), 0)

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


# def get_deblurred_fft(curr_images_fft, p=11):
#     deblurred_fft = np.zeros(curr_images_fft[0].shape, dtype=complex)
#     total_weight = np.zeros_like(deblurred_fft)
#     for image in curr_images_fft:
#         weight = np.abs(image)
#         weight = cv2.GaussianBlur(weight, (21, 21), 3)
#         weight = weight ** p
#
#         deblurred_fft += weight * image
#         total_weight += weight
#
#     return deblurred_fft / total_weight


original_image = cv2.imread('DIPSourceHW1.jpg', cv2.IMREAD_GRAYSCALE) / 255.0

blurred_images = load_blurred_images()
blurred_images_fft = images_to_dft(blurred_images)
blurred_images_fft = np.stack(blurred_images_fft)

# reconstruct from different number of images:
p = 13
for num_images in range(1, 100):
    curr_images_fft = blurred_images_fft[0:num_images]

    fft_weights = compute_fft_weights(curr_images_fft, p=p)
    reconstructed_fft = get_reconstructed_fft(curr_images_fft, fft_weights)
    reconstructed_image = ifft2(ifftshift(reconstructed_fft)).real

    print(f"num images = {num_images}   PSNR = {psnr(original_image, reconstructed_image)}")

    plt.imshow(reconstructed_image, cmap='gray')
    plt.title('num_images = ' + str(num_images))
    plt.show()

    plt.imshow(np.log(np.abs(reconstructed_fft)))
    plt.colorbar()
    plt.title('num_images = ' + str(num_images))
    plt.show()
