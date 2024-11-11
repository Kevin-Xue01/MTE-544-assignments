"""
Student Name & Last Name: Kevin Xue
Origianl Author : Pi Thanacha Choopojcharoen
You must change the name of your file to MTE_544_AS2_Q2_(your full name).py
Do not use jupyter notebook.

*You may want to install the following libraries if you haven't done so.*

pip install numpy matplotlib pandas scipy

"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
import random
import pandas as pd
def plot_ellipse(Q, b, ax):
    S = scipy.linalg.sqrtm(np.linalg.inv(Q))
    
    eigvals, eigvecs = np.linalg.eigh(S)
    aa, bb = eigvals

    theta = np.linspace(0, 2 * np.pi, 100)
    unit_circle = np.array([np.cos(theta), np.sin(theta)])
    ellipse_param = np.array([aa * unit_circle[0], bb * unit_circle[1]])
    ellipse_points = eigvecs @ ellipse_param + b

    ax.plot(ellipse_points[0, :], ellipse_points[1, :], 'b-', label='Fitted Ellipse')
    ax.plot(b[0], b[1], 'ro', label='Center')
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend(loc='best')
    ax.grid(True)

def visualize_data(p, ax, inliers, threshold):
    ax.scatter(p[:, 0], p[:, 1], color='red', alpha=0.5, label='Raw Measurements (Ellipse)')
    ax.scatter(inliers[:, 0], inliers[:, 1], color='purple', alpha=0.7, label='Inliers')

    for point in inliers:
        circle = plt.Circle(point, threshold, color='orange', fill=False, linestyle='--', alpha=0.7)
        ax.add_patch(circle)


def is_positive_definite(matrix):
    try:
        np.linalg.cholesky(matrix)
        return True
    except np.linalg.LinAlgError:
        return False

def fit_ellipse_subset(points):
    # Get the x and y coordinates from each of the passed in 5 points 
    x = points[:, 0]
    y = points[:, 1]

    # building the variable matrix: M, and the output vector: out
    M = np.vstack([x**2, 2*x*y, y**2, -2*x, -2*y]).T
    out = np.array([-1, -1, -1, -1, -1])
    
    # solving the system of equations
    res = np.linalg.solve(M, out)
    A, B, C, D, E = res

    # calculating alpha based on the formula provided
    alpha = 1 / (np.array([D, E]) @ np.linalg.inv(np.array([[A, B], [B, C]])) @ np.array([[D], [E]]) - 1)
    
    # calculating Q based on the formula provided
    Q = alpha * np.array([[A, B], [B, C]])

    # calculating b based on the formula provided
    b = alpha * np.linalg.inv(Q) @ np.array([[D], [E]])
    return Q, b

def ransac_ellipse(data, num_iterations=1000, threshold=0.2):
    # Given the data sets, perform RANSAC to find the best Q and b as well as the inliers
    # Hint: You should use fit_ellipse_subset 
    # Hint: in some case, the Q matrix might not be positive defintie, use is_positive_definite to check.
    # define a new matrix to store the best set of inliers (longest one with the most data points)
    best_set_of_inliers = []
    best_Q = []
    best_b = []
    
    for i in range(num_iterations):
        # reset the inliers
        inliers = []
        # choose a random set of 5 points from the data for fitting the ellipse
        subset = data[np.random.choice(data.shape[0], 5, replace=False), :]
        # print(subset.shape)
        Q, b = fit_ellipse_subset(subset)

        # calculate the distance of each data point from the fitted ellipse, and update the inliers array
        for j in range(len(data)):
            # calculate the difference between the data point and b
            pj_minus_b = data[j] - b.T
            
            # calculate the distance from the data point to the ellipse given the condition
            dist = np.abs(np.sqrt((pj_minus_b @ Q @ pj_minus_b.T) - 1))

            # if the point is within the threshold, add it to the set of inliers
            if dist <= threshold:
                inliers.append(data[j])

        # check if the Q matrix is positive definite
        pos_def = is_positive_definite(Q)
        if not pos_def: continue

        # detect if the ellipse is too eccentric*
        [e_vals,_] = np.linalg.eigh(scipy.linalg.sqrtm(np.linalg.inv(Q)))
        if e_vals[0]>20 or e_vals[1]>20:
            continue

        # if the length of the inliers array is longer than that of best_set_of_inliers, store it as the new best set of inliers
        if len(inliers) > len(best_set_of_inliers):
            best_set_of_inliers = inliers
            best_Q = Q
            best_b = b

    best_set_of_inliers = np.array(best_set_of_inliers)


    return best_Q, best_b, best_set_of_inliers

if __name__ == "__main__":
    # Load the data from CSV file and select N random points
    N = 500
    all_data = pd.read_csv('data_x_y.csv').to_numpy()
    dataset = all_data[np.random.choice(all_data.shape[0], N, replace=False), :]
    
    Q, b_est, inliers = ransac_ellipse(dataset, num_iterations=3000, threshold=0.2)

    # Plot the raw measurements and fitted ellipse
    fig, ax1 = plt.subplots(1, 1, figsize=(16, 8))
    visualize_data(dataset, ax1, inliers, threshold=0.5)
    plot_ellipse(Q, b_est, ax1)
    ax1.set_title("RANSAC Ellipse Fitting with Threshold Visualization")

    plt.show()

    # dataset = np.array([
    #     [2.92, -6.01],
    #     [3.40, -7.20],
    #     [4.99, -7.84],
    #     [5.48, -7.04],
    #     [4.20, -5.91]
    # ])
    # Q, b = fit_ellipse_subset(dataset)
    # fig, ax1 = plt.subplots(1, 1, figsize=(16, 8))
    # plot_ellipse(Q, b, ax1)
    # ax1.set_title("RANSAC Ellipse Fitting with Threshold Visualization")

    # plt.show()