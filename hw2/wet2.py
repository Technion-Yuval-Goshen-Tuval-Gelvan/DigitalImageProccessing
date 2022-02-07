import cv2
import scipy.signal
from matplotlib import pyplot as plt
import numpy as np
from wet2_utils import *

# --- parameters
LOW_RES_RATIO = 4
HIGH_RES_RATIO = 2
ALPHA = LOW_RES_RATIO // HIGH_RES_RATIO

KERNEL_SIZE = 7
GAUSSIAN_STD = 1
SINC_SCALE = 3
# ---


def create_patches(img, num_patches=50, patch_size=9):
    patches = []
    for _ in range(num_patches):
        x = np.random.randint(0, img.shape[0] - patch_size)
        y = np.random.randint(0, img.shape[1] - patch_size)
        patch = img[x:x + patch_size, y:y + patch_size]
        patches.append(patch)
    return patches


# def calc_Rj_for_patch(patch, kernel_size):
#


def estimate_kernel(img, num_large_patches=20, num_small_patches=100, large_patch_size=16,
                    num_iterations=10, kernel_size=KERNEL_SIZE):
    large_patches = create_patches(img, num_large_patches, large_patch_size)
    small_patches = create_patches(img, num_small_patches, large_patch_size // ALPHA)


    # plot some patches:
    plot_sample_pathces(large_patches[:9])

    # initialize delta kernel:
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[kernel_size // 2, kernel_size // 2] = 1

    for _ in range(num_iterations):
        down_sampled_large_patches = []
        for patch in large_patches:
            down_sampled_large_patches.append(down_sample(patch, kernel, ALPHA))


    return kernel



continuous_image = cv2.imread('DIPSourceHW2.png', cv2.IMREAD_GRAYSCALE)

gaussian_kernel = gaussian_kernel(KERNEL_SIZE, std=GAUSSIAN_STD)
l_im_gaussian = down_sample(continuous_image, gaussian_kernel, LOW_RES_RATIO)
h_im_gaussian = down_sample(continuous_image, gaussian_kernel, HIGH_RES_RATIO)
cv2.imwrite('l_im_gaussian.png', l_im_gaussian)
cv2.imwrite('h_im_gaussian.png', h_im_gaussian)


sinc_kernel = sinc_kernel(KERNEL_SIZE, scale=SINC_SCALE)
l_im_sinc = down_sample(continuous_image, sinc_kernel, LOW_RES_RATIO)
h_im_sinc = down_sample(continuous_image, sinc_kernel, HIGH_RES_RATIO)
cv2.imwrite('l_im_sinc.png', l_im_sinc)
cv2.imwrite('h_im_sinc.png', h_im_sinc)


kernel = estimate_kernel(l_im_gaussian)
plt.imshow(kernel, cmap='gray')
plt.show()

