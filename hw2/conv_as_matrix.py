import numpy as np
from scipy.linalg import toeplitz, circulant
import scipy.signal


def create_circulant_matrix(row, num_cols, num_rows):
    """ create circulant matrix suitble for convolution with a row vector """
    mat = []
    for i in range(num_rows):
        mat.append(row)
        row = np.roll(row, -1)
    return np.array(mat)[:, :num_cols]


def get_image_convolution_matrix(im, k_size):
    """ creates matrix operator that acts as convolution when applied on a stacked kernel.
        im*kernal = conv_mat @ reshape(flip(kernel), (-1, 1)) """
    # create circulant matrix from each row in F:
    im_size = im.shape[0]

    im = np.pad(im, ((k_size//2, k_size//2), (k_size//2, k_size//2)))

    # create circulant matrix from each row in im:
    circulant_matrices = []
    for row in im:
        row_c = create_circulant_matrix(row, k_size, im_size)
        # have to roll the rows one down:
        # row_c = np.roll(row_c, 1, axis=0)
        circulant_matrices.append(row_c)

    # create indices for the doubly blocked matrix, each index corresponds to a circulant matrix:
    indices_row = np.arange(im_size + k_size - 1)
    doubly_indices = create_circulant_matrix(indices_row, k_size, im_size)
    # have to roll the rows one down here as well :
    # doubly_indices = np.roll(doubly_indices, 1, axis=0)

    # create the doubly blocked matrix:
    doubly_blocked_matrix = np.zeros((im_size ** 2, k_size ** 2))
    for i in range(doubly_indices.shape[0]):
        for j in range(doubly_indices.shape[1]):
            block = circulant_matrices[doubly_indices[i, j]]
            doubly_blocked_matrix[i * im_size: (i + 1) * im_size, j * k_size: (j + 1) * k_size] = block

    # also make sure to flip the kernel:
    conv_mat = doubly_blocked_matrix @ np.fliplr(np.eye(k_size ** 2))

    return conv_mat


def get_downsample_matrix(im_size, ratio):
    """
    creates matrix that downsamples the image by a given ratio when the image rows are stacked
    to a single vector
    """
    mat = np.zeros((im_size ** 2 // ratio ** 2, im_size ** 2))
    new_im_size = im_size // ratio
    for i in range(new_im_size):
        for j in range(new_im_size):
            mat[i*new_im_size + j, i*new_im_size*ratio**2 + j*ratio] = 1
    return mat


def get_downsample_convolution_matrix(im, ratio, k_size):
    """
    creates matrix that when multiplied by a stacked kernel acts as convolution + downsample operator
    """
    conv_mat = im.reshape(im.size)
    conv_circulant = create_circulant_matrix(conv_mat, im.size, im.size)

    # optional: not take in count the edges
    conv_toplitz = get_image_convolution_matrix(im, k_size)

    downsampled_matrix = get_downsample_matrix(im.shape[0], ratio)
    return downsampled_matrix @ conv_circulant
