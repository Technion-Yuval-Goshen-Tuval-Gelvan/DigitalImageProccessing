import cv2
import scipy.signal
from matplotlib import pyplot as plt
import numpy as np
from wet2_utils import *
from conv_as_matrix import *

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


def regularization_term(kernel):
    # k_pad_x = np.pad(kernel, [[0, 0], [0, 1]])
    # k_x = k_pad_x[:, 1:] - k_pad_x[:, :-1]
    #
    # k_pad_y = np.pad(kernel, [[0, 1], [0, 0]])
    # k_y = k_pad_y[1:, :] - k_pad_y[:-1, :]
    #
    # k_derivatives = np.sqrt(k_x ** 2 + k_y ** 2)
    # k_derivatives = k_derivatives / np.sum(k_derivatives)
    k_laplacian = cv2.Laplacian(kernel, cv2.CV_64F)
    plt.imshow(k_laplacian, cmap='gray')
    plt.show()
    k_derivatives = k_laplacian.reshape((-1, 1))
    return k_derivatives @ k_derivatives.T



def estimate_kernel(img, num_large_patches=30, num_small_patches=30, large_patch_size=16,
                    num_iterations=50, kernel_size=KERNEL_SIZE, reg_weight=1, weights_sigma=0.1):
    large_patches = create_patches(img, num_large_patches, large_patch_size)
    small_patches = create_patches(img, num_small_patches, large_patch_size // ALPHA)

    R_j_list = []
    for patch in large_patches:
        R_j = get_downsample_convolution_matrix(patch, ALPHA, kernel_size)
        R_j_list.append(R_j)  # save them for later (last step)

    # plot some patches:
    plot_sample_pathces(large_patches[:9])

    # initialize delta kernel:
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[kernel_size // 2, kernel_size // 2] = 1

    for _ in range(num_iterations):
        down_sampled_large_patches = [] # r_j^alpha in the paper
        for j, patch in enumerate(large_patches):
            R_j = R_j_list[j]
            fliped_stacked_kernel = np.flipud(np.fliplr(kernel))
            fliped_stacked_kernel = np.reshape(fliped_stacked_kernel, (-1, 1))
            down_sampled_large_patches.append(np.squeeze(R_j @ fliped_stacked_kernel))
        down_sampled_large_patches = np.array(down_sampled_large_patches)

        W = []
        for qi in small_patches: # calculate each line in w_ij
            qi = qi.reshape((1, -1))
            w_i = qi - down_sampled_large_patches  # for each j, using broadcasting
            w_i = np.linalg.norm(w_i, axis=1, ord=2)
            w_i = np.exp(-0.5 * w_i/(weights_sigma**2))
            w_i = w_i / np.sum(w_i)
            W.append(w_i)
        W = np.array(W)
        W /= np.sum(W, axis=1)[:, np.newaxis]

        # calculate new kernel k = A^-1 @ b:
        A = np.zeros((kernel_size**2, kernel_size**2))
        b = np.zeros((kernel_size**2, 1))
        for j in range(num_large_patches):
            R_j = R_j_list[j]
            A += R_j.T @ R_j * np.sum(W[:, j])
            for i in range(num_small_patches):
                qi = np.flipud(np.fliplr(small_patches[i]))
                qi = qi.reshape((-1, 1))
                b += W[i, j] * R_j.T @ qi

        A = A / weights_sigma**2
        # add regularization C^T @ C
        A += reg_weight * regularization_term(kernel)

        kernel = np.linalg.inv(A) @ b
        kernel = kernel.reshape((kernel_size, kernel_size))
        kernel = kernel/np.sum(kernel)


    return kernel



continuous_image = cv2.imread('DIPSourceHW2.png', cv2.IMREAD_GRAYSCALE) / 255

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


kernel = estimate_kernel(l_im_gaussian, num_iterations=15)
plt.imshow(kernel, cmap='gray')
plt.show()

