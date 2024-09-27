import numpy as np
import matplotlib.pyplot as plt

circular_path_radius = 200.0 # [cm]
omega = 1.0 # rad/s
r = 10.0 # wheel radius [cm]
l = 25.0 # G.C. to wheel center distance [cm]
dt = 0.1 # Time step (seconds)
total_simulation_time = 30 # Total time of simulation (seconds)
time = np.arange(0, total_simulation_time + dt, dt) # timestamp array

def run_simulation_constant_input(u1, u2, u3):
    # Initialize state [x, y, theta]
    state = np.array([0.0, 0.0, 0.0])

    _trajectory = [] # empty list to store state at each timestamp

    # Simulation loop
    for t in time:
        # Store the current state
        _trajectory.append(state.copy())

        # Construct state matrix
        A = np.linalg.inv((1/r) * np.array([
            [-np.sin(state[2]), np.cos(state[2]), l], 
            [np.cos(state[2] + 7*np.pi/6), np.sin(state[2] + 7*np.pi/6), l*(-1/2)*np.sin(7*np.pi/6) - l*(np.sqrt(3)/2)*np.cos(7*np.pi/6)], 
            [np.cos(state[2] + 11*np.pi/6), np.sin(state[2] + 11*np.pi/6), l*(-1/2)*np.sin(11*np.pi/6) + l*(np.sqrt(3)/2)*np.cos(11*np.pi/6)]
        ]))
        
        # Use euler's method to update state based on the derivative of the state at the current timestamp
        state += np.matmul(A, np.array([[u1], [u2], [u3]])).reshape(-1) * dt
        
    return np.insert(np.array(_trajectory), 0, time, axis=1) # return list of states, along with the timestamp at each state

def run_simulation(u1, u2, u3):
    # Initialize state [x, y, theta]
    state = np.array([0.0, 0.0, 0.0])

    _trajectory = [] # empty list to store state at each timestamp

    # Simulation loop
    for i, _ in enumerate(time):
        # Store the current state
        _trajectory.append(state.copy())

        # Construct state matrix
        A = np.linalg.inv((1/r) * np.array([
            [-np.sin(state[2]), np.cos(state[2]), l], 
            [np.cos(state[2] + 7*np.pi/6), np.sin(state[2] + 7*np.pi/6), l*(-1/2)*np.sin(7*np.pi/6) - l*(np.sqrt(3)/2)*np.cos(7*np.pi/6)], 
            [np.cos(state[2] + 11*np.pi/6), np.sin(state[2] + 11*np.pi/6), l*(-1/2)*np.sin(11*np.pi/6) + l*(np.sqrt(3)/2)*np.cos(11*np.pi/6)]
        ]))
        
        # Use euler's method to update state based on the derivative of the state at the current timestamp
        state += np.matmul(A, np.array([[u1[i]], [u2[i]], [u3[i]]])).reshape(-1) * dt
        
    return np.insert(np.array(_trajectory), 0, time, axis=1) # return list of states, along with the timestamp at each state

trajectory = run_simulation_constant_input(2, -2, 0) # 60 degree straight line path 

u1_profile = (omega/r)*(circular_path_radius*np.cos(omega*time) + l)
u2_profile = (omega/r)*(0.5*circular_path_radius*np.sqrt(3)*np.sin(omega*time) - 0.5*circular_path_radius*np.cos(omega*time) + l)
u3_profile = (omega/r)*(-0.5*circular_path_radius*np.sqrt(3)*np.sin(omega*time) - 0.5*circular_path_radius*np.cos(omega*time) + l)

trajectory = run_simulation(u1_profile, u2_profile, u3_profile)

fig, axs = plt.subplots(4, figsize=(5, 8))  # Increase figure size

# Adjust space between plots and add margins
fig.subplots_adjust(wspace=2, hspace=2)

# Plot each figure on its own subplot
axs[0].plot(trajectory[:, 1], trajectory[:, 2], color='b', marker='x', markersize=2)
axs[0].set_xlabel("x (meters)")
axs[0].set_ylabel("y (meters)")
axs[0].set_title("2D Trajectory")

axs[1].plot(trajectory[:, 0], trajectory[:, 1], color='r')
axs[1].set_ylabel("x (meters)")
axs[1].set_xlabel("t (seconds)")
axs[1].set_title("x(t)")

axs[2].plot(trajectory[:, 0], trajectory[:, 2], color='g')
axs[2].set_ylabel("y (meters)")
axs[2].set_xlabel("t (seconds)")
axs[2].set_title("y(t)")

axs[3].plot(trajectory[:, 0], trajectory[:, 3], color='m')
axs[3].set_ylabel("θ (rad)")
axs[3].set_xlabel("t (seconds)")
axs[3].set_title("θ(t)")

plt.tight_layout()
plt.show()