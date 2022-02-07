import numpy as np
import cv2
from matplotlib import pyplot as plt
import scipy.signal


def gaussian_kernel(size, std=1):
    edge = size // 2
    ax = np.linspace(-edge, edge, num=size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2 * std **2))
    return kernel / kernel.sum()


def sinc_kernel(size, scale=1):
    edge = size // 2
    ax = np.linspace(-edge, edge, num=size)
    kernel = np.outer(ax, ax)
    kernel = np.sinc(kernel / scale )

    return kernel / kernel.sum()


def down_sample(image, kernel, ratio):
    blurred_image = scipy.signal.convolve2d(image, kernel, mode='same')
    return blurred_image[::ratio, ::ratio]


def plot_and_save_four_images(l_im_gaussian, l_im_sinc, h_im_gaussian, h_im_sinc, path):
    fig, axs = plt.subplots(2, 2)
    axs[0, 0].imshow(l_im_gaussian, cmap='gray')
    axs[0, 0].set_title('low res with gaussian')
    axs[0, 1].imshow(h_im_gaussian, cmap='gray')
    axs[0, 1].set_title('high res with gaussian')
    axs[1, 0].imshow(l_im_sinc, cmap='gray')
    axs[1, 0].set_title('low res with sinc')
    axs[1, 1].imshow(h_im_sinc, cmap='gray')
    axs[1, 1].set_title('high res with sinc')

    for ax in axs.flat:
        ax.axis('off')
    plt.show()
    fig.savefig('four_blurred_images.png')


def plot_sample_pathces(patch_list):
    fig, axs = plt.subplots(3, 3)
    for i, ax in enumerate(axs.flat):
        ax.imshow(patch_list[i], cmap='gray')
        ax.axis('off')
    plt.show()


