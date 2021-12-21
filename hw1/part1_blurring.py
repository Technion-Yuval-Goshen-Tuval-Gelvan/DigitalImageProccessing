import scipy.io
import scipy.signal
import cv2
import matplotlib.pyplot as plt
import numpy as np

def load_trajectories(filename):
    """
    Loads the trajectories from the given file.
    """
    trajectories_mat = scipy.io.loadmat(filename)
    points_x = trajectories_mat['X']
    points_y = trajectories_mat['Y']

    # create list of trajectories where each trajectory is both X and Y coordinates
    trajectories = []
    for k in range(len(points_x)):
        traj = np.array([points_x[k], points_y[k]])
        trajectories.append(traj)

    return trajectories


def plot_trajectories(trajectories):
    """
    plot all trajectories in a single figure grid and save to file
    """
    plt.figure(figsize=(10, 10))
    for i in range(len(trajectories)):
        plt.subplot(10, 10, i+1)
        plt.plot(trajectories[i][0], trajectories[i][1])
        plt.xticks([])
        plt.yticks([])
        plt.xlim([-10, 10])
        plt.ylim([-10, 10])

    plt.savefig('trajectories.png')
    plt.show()


def create_psf(trajectories, psf_size=20):
    """
    Create PSF matrix for each trajectory, save it as an image and as matrix,
    and return list of PSF's
    """
    psf_list = []
    for k in range(len(trajectories)):
        hist, _, _, _ = plt.hist2d(trajectories[k][0],
                                   trajectories[k][1],
                                   range=[[-10, 10], [-10, 10]],
                                   bins=psf_size,
                                   cmap='gray',
                                   density=True)
        plt.savefig(f"PSFs/PSF_{k}.png")
        scipy.io.savemat(f"PSFs/PSF_{k}.mat", {'items': hist})
        psf_list.append(hist)

    return psf_list


def motion_blur(img, psf_list):
    """
    Apply blur using all the PSF's and save to file
    """
    for k in range(len(psf_list)):
        blurred_img = scipy.signal.convolve2d(img, psf_list[k], mode='same', boundary='wrap')
        num = str(k).zfill(3)
        cv2.imwrite(f"blurred/blurred_{num}.jpg", blurred_img)


trajectories = load_trajectories('100_motion_paths.mat')
plot_trajectories(trajectories)

psf_list = create_psf(trajectories, psf_size=20)

im = cv2.imread("DIPSourceHW1.jpg", cv2.IMREAD_GRAYSCALE)

motion_blur(im, psf_list)

