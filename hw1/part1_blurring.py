import numpy as np
import scipy.io
import matplotlib.pyplot as plt


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
    plot all trajectories in a single figure grid
    """
    plt.figure(figsize=(10, 10))
    for i in range(len(trajectories)):
        plt.subplot(10, 10, i+1)
        plt.plot(trajectories[i][0], trajectories[i][1])
        plt.xticks([])
        plt.yticks([])

    plt.show()


trajectories = load_trajectories('100_motion_paths.mat')
plot_trajectories(trajectories)

# if __name__ == '__main__':
#     main()
