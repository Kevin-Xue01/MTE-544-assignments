import numpy as np
import matplotlib.pyplot as plt

dt = 0.1   # Time step (seconds)
total_simulation_time = 30     # Total time of simulation (seconds)
time = np.arange(0, total_simulation_time + dt, dt) # timestamp array
T = 0.2 # track length
r = 0.1 # wheel radius

def run_simulation_constant_input(ul, ur):
    # Initialize state [x, y, theta]
    state = np.array([0.0, 0.0, 0.0])

    _trajectory = [] # empty list to store state at each timestamp

    # Simulation loop
    for _ in time:
        # Store the current state
        _trajectory.append(state.copy())

        # Construct state matrix
        A = np.array([[0.5 * r * np.cos(state[2]), 0.5 * r * np.cos(state[2])], [0.5 * r * np.sin(state[2]), 0.5 * r * np.sin(state[2])], [r / T, -1 * r / T]])
        
        # Use euler's method to update state based on the derivative of the state at the current timestamp
        state += np.matmul(A, np.array([[ur], [ul]])).reshape(-1) * dt
        
    return np.insert(np.array(_trajectory), 0, time, axis=1) # return list of states, along with the timestamp at each state

def run_simulation(ul, ur):
    # Initialize state [x, y, theta]
    state = np.array([0.0, 0.0, 0.0])

    _trajectory = [] # empty list to store state at each timestamp

    # Simulation loop
    for i, _ in enumerate(time):
        # Store the current state
        _trajectory.append(state.copy())

        # Construct state matrix
        A = np.array([[0.5 * r * np.cos(state[2]), 0.5 * r * np.cos(state[2])], [0.5 * r * np.sin(state[2]), 0.5 * r * np.sin(state[2])], [r / T, -1 * r / T]])
        
        # Use euler's method to update state based on the derivative of the state at the current timestamp
        state += np.matmul(A, np.array([[ur[i]], [ul[i]]])).reshape(-1) * dt
        
    return np.insert(np.array(_trajectory), 0, time, axis=1) # return list of states, along with the timestamp at each state

trajectory = run_simulation_constant_input(..., ...)

ul_profile = ...
ur_profile = ...

trajectory = run_simulation(ul_profile, ur_profile)

# fig, axs = plt.subplots(4, figsize=(5, 8))  # Increase figure size

# # Adjust space between plots and add margins
# fig.subplots_adjust(wspace=2, hspace=2)

# # Plot each figure on its own subplot
# axs[0].plot(trajectory[:, 1], trajectory[:, 2], color='b', marker='x', markersize=2)
# axs[0].set_xlabel("x (meters)")
# axs[0].set_ylabel("y (meters)")
# axs[0].set_title("2D Trajectory")

# axs[1].plot(trajectory[:, 0], trajectory[:, 1], color='r')
# axs[1].set_ylabel("x (meters)")
# axs[1].set_xlabel("t (seconds)")
# axs[1].set_title("x(t)")

# axs[2].plot(trajectory[:, 0], trajectory[:, 2], color='g')
# axs[2].set_ylabel("y (meters)")
# axs[2].set_xlabel("t (seconds)")
# axs[2].set_title("y(t)")

# axs[3].plot(trajectory[:, 0], trajectory[:, 3], color='m')
# axs[3].set_ylabel("θ (rad)")
# axs[3].set_xlabel("t (seconds)")
# axs[3].set_title("θ(t)")

# plt.tight_layout()
# plt.show()