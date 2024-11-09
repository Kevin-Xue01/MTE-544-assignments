"""
Student Name & Last Name: 
Origianl Author : Pi Thanacha Choopojcharoen
You must change the name of your file to MTE_544_AS2_Q2_(your full name).py
Do not use jupyter notebook.

*You may want to install the following libraries if you haven't done so.*

pip install numpy matplotlib pandas scipy

"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
import pandas as pd

def plot_ellipse(Q, b, ax):
    lambda_var = 9.80665
    S = scipy.linalg.sqrtm(np.linalg.inv(Q))/lambda_var
    
    eigvals, eigvecs = np.linalg.eigh(lambda_var*S)
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
    # Extract x and y coordinates
    x = points[:, 0]
    y = points[:, 1]
    
    # Formulate the design matrix for the least squares problem
    D = np.column_stack([x**2, x*y, y**2, x, y, np.ones_like(x)])
    
    # Solve the normal equation using SVD for least squares
    _, _, V = np.linalg.svd(D)
    coeffs = V[-1, :]
    
    # Separate the quadratic form matrix Q and vector b
    Q = np.array([[coeffs[0], coeffs[1] / 2],
                  [coeffs[1] / 2, coeffs[2]]])
    b = np.array([coeffs[3], coeffs[4]])
    c = coeffs[5]
    
    return Q, b, c

def ransac_ellipse(data, num_iterations=1000, threshold=0.2):
    best_inliers = []
    best_Q = None
    best_b = None

    for _ in range(num_iterations):
        # Randomly sample a subset of points (e.g., minimum number needed is 5 for an ellipse)
        subset = data[np.random.choice(data.shape[0], 5, replace=False)]
        print(subset)
        # Fit an ellipse to this subset
        Q, b, c = fit_ellipse_subset(subset)
        
        # Check if Q is positive definite (ellipse constraint)
        if not is_positive_definite(Q):
            continue  # Skip this iteration if the matrix is not positive definite
        
        # Count inliers
        inliers = []
        for point in data:
            x, y = point
            # Ellipse equation: [x y]Q[x y]^T + b[x y] + c ≈ 0
            distance = abs(np.array([x, y]) @ Q @ np.array([x, y]) + b @ np.array([x, y]) + c)
            if distance < threshold:
                inliers.append(point)
        
        # Update if this model has the most inliers
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_Q = Q
            best_b = b

    return best_Q, best_b, np.array(best_inliers)

if __name__ == "__main__":
    # Load the data from CSV file and select N random points
    N = 500
    all_data = pd.read_csv('data_x_y.csv').to_numpy()
    dataset = all_data[np.random.choice(all_data.shape[0], N, replace=False), :]

    # dataset is p
    dataset = np.array([
        [2.92, -6.01],
        [3.40, -7.20],
        [4.99, -7.84],
        [5.48, -7.04],
        [4.20, -5.91]
    ])

    Q, b_est, inliers = ransac_ellipse(dataset)
    # Plot the raw measurements and fitted ellipse
    fig, ax1 = plt.subplots(1, 1, figsize=(16, 8))
    visualize_data(dataset, ax1, inliers, threshold=0.1)
    plot_ellipse(Q, b_est, ax1)
    ax1.set_title("RANSAC Ellipse Fitting with Threshold Visualization")

    plt.show()