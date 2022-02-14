import cv2
import scipy.signal
import sklearn
from matplotlib import pyplot as plt
import numpy as np
from scipy import fftpack, signal
from scipy.linalg import circulant
from wet2_utils import *
from conv_as_matrix import *
import sklearn.neighbors

# --- parameters
LOW_RES_RATIO = 3
HIGH_RES_RATIO = 1
ALPHA = LOW_RES_RATIO // HIGH_RES_RATIO

KERNEL_SIZE = 12
GAUSSIAN_STD = 1
SINC_SCALE = 3


#  Weiner filter
def reconstract_image(img, psf, epsilon=0.1):
    # Normalize the psf
    if np.sum(psf):
        psf /= np.sum(psf)

    F_img = fftpack.fft2(img)
    F_psf = fftpack.fft2(psf, shape=img.shape)
    F_psf = np.conj(F_psf) / (np.abs(F_psf) ** 2 + epsilon)
    deblur_img = F_img * F_psf
    deblur_img = np.abs(fftpack.ifft2(deblur_img))
    return deblur_img


def create_patches(img, patch_size=15, step_size=1):
    patches = []
    for i in range(0, int(img.shape[0] - patch_size), step_size):
        for j in range(0, int(img.shape[1] - patch_size), step_size):
            patch = img[i:i + patch_size, j:j + patch_size]
            patches.append(patch)
    return patches


def regularization_term(kernel_size):
    a = -1
    b = 4

    laplacian_matrix = np.zeros((kernel_size ** 2, kernel_size ** 2))

    diag_mat_1 = np.zeros((kernel_size, kernel_size))
    for i in range(kernel_size):
        for j in range(kernel_size):
            if i == j:
                diag_mat_1[i, j] = b
            if abs(i - j) == 1:
                diag_mat_1[i, j] = a

    diag_mat_2 = np.zeros((kernel_size, kernel_size))
    for i in range(kernel_size):
        diag_mat_2[i, i] = a

    start_index = 0
    end_index = kernel_size

    for i in range(kernel_size):
        laplacian_matrix[start_index:end_index, start_index:end_index] = diag_mat_1
        start_index += kernel_size
        end_index += kernel_size

    start_index_x = 0
    end_index_x = kernel_size
    start_index_y = kernel_size
    end_index_y = 2 * kernel_size

    for i in range(kernel_size - 1):
        laplacian_matrix[start_index_x:end_index_x, start_index_y:end_index_y] = diag_mat_2
        laplacian_matrix[start_index_y:end_index_y, start_index_x:end_index_x] = diag_mat_2

        start_index_x += kernel_size
        end_index_x += kernel_size
        start_index_y += kernel_size
        end_index_y += kernel_size

    return laplacian_matrix @ laplacian_matrix.T


def estimate_kernel(img, large_patch_size=15, num_iterations=5, kernel_size=KERNEL_SIZE, weights_sigma=0.06,
                    k_neighbors=5, show_plots=False):

    small_patch_size = large_patch_size // ALPHA
    large_patches = create_patches(img, large_patch_size, ALPHA)
    small_patches = create_patches(img, small_patch_size, 1)

    R_j_list = []
    for patch in large_patches:
        R_j = get_downsample_convolution_matrix(patch, ALPHA, kernel_size)
        R_j_list.append(R_j)  # save them for later (last step)

    q_vec = []
    for patch in small_patches:
        q_i = patch.reshape(patch.size)
        q_vec.append(q_i)
    q_vec = np.array(q_vec)

    # plot some patches:
    if show_plots:
        plot_sample_pathces(large_patches[:9])

    # initialize delta kernel:
    kernel = fftpack.fftshift(scipy.signal.unit_impulse((kernel_size, kernel_size)))
    kernel = kernel.reshape(kernel.size)
    kernel_reshaped = kernel.reshape((kernel_size, kernel_size))
    reg_term = regularization_term(kernel_size)

    for i in range(num_iterations):
        print("iteration: ", i+1)
        down_sampled_large_patches = []  # r_j^alpha in the paper
        for j, patch in enumerate(large_patches):
            R_j = R_j_list[j]
            down_sampled_large_patches.append(R_j @ kernel)
        down_sampled_large_patches = np.array(down_sampled_large_patches)

        # plot downsampled large patches:
        if show_plots:
            _down_sampled = []
            for i in range(len(down_sampled_large_patches)):
                p = down_sampled_large_patches[i].reshape((small_patch_size, small_patch_size))
                _down_sampled.append(p)
            plot_sample_pathces(_down_sampled[:9])

        tree = sklearn.neighbors.BallTree(down_sampled_large_patches, leaf_size=2)
        W = np.zeros((len(q_vec), len(down_sampled_large_patches)))
        for i, q_i in enumerate(q_vec):
            expand_q_i = np.expand_dims(q_i, 0)
            _, indices = tree.query(expand_q_i, k=k_neighbors)
            for j in indices:
                W[i, j] = np.exp(-0.5 * (np.linalg.norm(q_i - down_sampled_large_patches[j]) ** 2) /
                                     (weights_sigma ** 2))

        W_sum = np.sum(W, axis=1)

        for row in range(W.shape[0]):
            row_sum = W_sum[row]
            if row_sum:
                W[row] = W[row] / row_sum  ## normalize each column

        # calculate new kernel k = A^-1 @ b:
        A = np.zeros((kernel_size ** 2, kernel_size ** 2))
        b = np.zeros_like(kernel)

        for i in range(W.shape[0]):
            for j in range(W.shape[1]):
                if not W[i, j]:
                    continue
                A += W[i, j] * R_j_list[j].T @ R_j_list[j] + reg_term
                b += W[i, j] * R_j_list[j].T @ q_vec[i]

        A = A / (weights_sigma ** 2)
        epsilon = 1e-12
        epsilon_mat = np.eye(kernel.shape[0]) * epsilon
        kernel = np.linalg.inv(A + epsilon_mat) @ b
        kernel_reshaped = kernel.reshape((kernel_size, kernel_size))

        if show_plots:
            plt.imshow(kernel_reshaped)
            plt.show()

    return kernel_reshaped


def psnr(im1, im2):
    mse = np.mean(np.power(np.subtract(im1, im2), 2))
    PIXEL_MAX = 1.0
    return 20 * np.log10(PIXEL_MAX) - 10 * np.log10(mse)


def get_images(path, scale_factor=1):

    continuous_image = cv2.imread('DIPSourceHW2.png', cv2.IMREAD_GRAYSCALE) / scale_factor
    continuous_image = continuous_image[:-3, :-3]

    gaussian_kernel = get_gaussian_kernel(KERNEL_SIZE, std=GAUSSIAN_STD)
    l_im_gaussian = down_sample(continuous_image, gaussian_kernel, LOW_RES_RATIO)
    h_im_gaussian = down_sample(continuous_image, gaussian_kernel, HIGH_RES_RATIO)

    sinc_kernel = get_sinc_kernel(KERNEL_SIZE, scale=SINC_SCALE)
    l_im_sinc = down_sample(continuous_image, sinc_kernel, LOW_RES_RATIO)
    h_im_sinc = down_sample(continuous_image, sinc_kernel, HIGH_RES_RATIO)

    return l_im_gaussian, h_im_gaussian, l_im_sinc, h_im_sinc


def main():
    l_im_gaussian, h_im_gaussian, l_im_sinc, h_im_sinc = get_images('DIPSourceHW2.png', scale_factor=1)
    cv2.imwrite('l_im_gaussian.png', l_im_gaussian)
    cv2.imwrite('h_im_gaussian.png', h_im_gaussian)
    cv2.imwrite('l_im_sinc.png', l_im_sinc)
    cv2.imwrite('h_im_sinc.png', h_im_sinc)

    # to get the kernel we need to normalize the original image:
    l_im_gaussian, h_im_gaussian, l_im_sinc, h_im_sinc = get_images('DIPSourceHW2.png', scale_factor=255)

    best_gaussian_kernel = estimate_kernel(l_im_gaussian, num_iterations=5, weights_sigma=0.1, large_patch_size=15,
                                      k_neighbors=5, show_plots=False)
    l_im_gaussian_recon = reconstract_image(h_im_gaussian, best_gaussian_kernel)
    recon_gaussian_psnr = psnr(l_im_gaussian_recon, h_im_gaussian)
    print("PSNR Gaussian Reconstraction:", recon_gaussian_psnr)

    best_sinc_kernel = estimate_kernel(l_im_sinc, num_iterations=5, weights_sigma=0.1, large_patch_size=15,
                                  k_neighbors=5, show_plots=False)
    l_im_sinc_recon = reconstract_image(h_im_sinc, best_sinc_kernel)
    recon_sinc_psnr = psnr(l_im_sinc_recon, h_im_sinc)
    print("PSNR Sinc Reconstraction:", recon_sinc_psnr)

    plt.imsave(f'l_im_gaussian_recon_psnr_{recon_gaussian_psnr}.png', l_im_gaussian_recon, cmap='gray')
    plt.show()
    plt.imsave(f'l_im_sinc_recon_recon_psnr_{recon_sinc_psnr}.png', l_im_sinc_recon, cmap='gray')
    plt.show()


if __name__ == '__main__':
    main()