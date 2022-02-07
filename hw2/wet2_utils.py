import numpy as np
import cv2
from matplotlib import pyplot as plt


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
    blurred_image = cv2.filter2D(image, -1, kernel)
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


F = np.array([[1, 2, 3,],
                  [4, 5, 6],
                  [7, 8, 9]])

I = np.array([[10, 20],
                   [30, 40]])

# number columns and rows of the input
I_row_num, I_col_num = I.shape

# number of columns and rows of the filter
F_row_num, F_col_num = F.shape

#  calculate the output dimensions
output_row_num = I_row_num + F_row_num - 1
output_col_num = I_col_num + F_col_num - 1
print('output dimension:', output_row_num, output_col_num)

F_zero_padded = np.pad(F, ((output_row_num - F_row_num, 0),
                           (0, output_col_num - F_col_num)),
                        'constant', constant_values=0)
print('F_zero_padded: ', F_zero_padded)

from scipy.linalg import toeplitz

# use each row of the zero-padded F to creat a toeplitz matrix.
#  Number of columns in this matrices are same as numbe of columns of input signal
toeplitz_list = []
for i in range(F_zero_padded.shape[0]-1, -1, -1): # iterate from last row to the first row
    c = F_zero_padded[i, :] # i th row of the F
    r = np.r_[c[0], np.zeros(I_col_num-1)] # first row for the toeplitz fuction should be defined otherwise
                                                        # the result is wrong
    toeplitz_m = toeplitz(c,r) # this function is in scipy.linalg library
    toeplitz_list.append(toeplitz_m)
    print('F '+ str(i)+'\n', toeplitz_m)

# doubly blocked toeplitz indices:
#  this matrix defines which toeplitz matrix from toeplitz_list goes to which part of the doubly blocked
c = range(1, F_zero_padded.shape[0]+1)
r = np.r_[c[0], np.zeros(I_row_num-1, dtype=int)]
doubly_indices = toeplitz(c, r)
print('doubly indices \n', doubly_indices)

## creat doubly blocked matrix with zero values
toeplitz_shape = toeplitz_list[0].shape # shape of one toeplitz matrix
h = toeplitz_shape[0]*doubly_indices.shape[0]
w = toeplitz_shape[1]*doubly_indices.shape[1]
doubly_blocked_shape = [h, w]
doubly_blocked = np.zeros(doubly_blocked_shape)

# tile toeplitz matrices for each row in the doubly blocked matrix
b_h, b_w = toeplitz_shape # hight and withs of each block
for i in range(doubly_indices.shape[0]):
    for j in range(doubly_indices.shape[1]):
        start_i = i * b_h
        start_j = j * b_w
        end_i = start_i + b_h
        end_j = start_j + b_w
        doubly_blocked[start_i: end_i, start_j:end_j] = toeplitz_list[doubly_indices[i,j]-1]

print('doubly_blocked: ', doubly_blocked)

vectorized_I = np.flip(I, axis=0)
vectorized_I = vectorized_I.reshape(-1, 1)
print('vectorized_I: ', vectorized_I)

doubly_blocked @ vectorized_I

