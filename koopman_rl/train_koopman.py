#---------------  Koopman Training for go2 ---------------#
#--------------- len 2025/4/12 ---------------#

# Import necessary libraries
import pandas as pd
import ast
import numpy as np
import matplotlib.pyplot as plt
import pykoopman as pk
from pykoopman.regression import DMDRegressor
from pydmd import DMD
from scipy.io import savemat


# Constants to control the plotting and training behavior
PLOT_TEST = False
PLOT_VALIDATION = False
PLOT_VALIDATION_SINGLESTEP = False
ITER_TRAIN_MATRIX = True

# Configure numpy printing behavior
np.set_printoptions(threshold=np.inf)

# Load data from a CSV file
file_path = 'D:\\koopman_robot\\Koopman_optimization_rl_policy\\resourse\\data\\g1_koopman_data_hard.csv'
csv_data = pd.read_csv(file_path)
all_columns_data = []

# Convert string representations of lists or tuples into actual Python objects(delete "" if needed)
for col in csv_data.columns:
    if csv_data[col].dtype == 'object':
        try:
            # Attempt to convert strings to Python objects (e.g., lists or tuples)
            csv_data[col] = csv_data[col].apply(
                lambda x: ast.literal_eval(x) 
                if isinstance(x, str) and (x.startswith('[') or x.startswith('('))
                else x
            )
        except Exception as e:
            print(f"Error parsing column {col}: {e}")
        
        # Handle cases where the column contains lists, tuples, or np.ndarrays
        if isinstance(csv_data[col].iloc[0], (list, np.ndarray, tuple)): # the first column is not data
            expanded_columns = np.array([np.array(val) 
                                         if isinstance(val, (list, np.ndarray, tuple)) 
                                         else np.array([val]) 
                                         for val in csv_data[col]])
            all_columns_data.append(expanded_columns)
        else:
            all_columns_data.append(csv_data[col].values.reshape(-1, 1))
    else:
        all_columns_data.append(csv_data[col].values.reshape(-1, 1)) #convert it to numpy?

# Merge all the column data into a single matrix and exclude the last 9 columns
merged_data = np.hstack(all_columns_data)
merged_data = merged_data[:, :-9] # why ?

# Define specific indices related to the data (joint positions, velocities, and torques)
q_indices = np.arange(14, 14 + 10 * 12, 10)
dq_indices = np.arange(15, 15 + 10 * 12, 10)
tauE_indices = np.arange(17, 17 + 10 * 12, 10)

# Create a new matrix with only the required features
needed_data = np.zeros((merged_data.shape[0], 52))
needed_data[:, :3] = merged_data[:, -3:]  # Last three columns: assumed end-state info
needed_data[:, 3:16] = merged_data[:, :13]  # First 13 columns: assumed initial state info
needed_data[:, 16:28] = merged_data[:, q_indices]  # Joint positions
needed_data[:, 28:40] = merged_data[:, dq_indices]  # Joint velocities
needed_data[:, -12:] = merged_data[:, tauE_indices]  # Torques

# Print the shape of the needed data matrix
print("Needed Data Shape = ", needed_data.shape)

# Create trajectories by slicing the data into chunks of 100
traj = []
number_of_100steps = needed_data.shape[0] // 100
for i in range(number_of_100steps):
    data_clip = np.zeros((100, 52))
    data_clip = needed_data[i:i+100, :]
    traj.append(data_clip)
traj.append(needed_data[-100:, :])  # Add the last part of data that doesn't fill a full 100-step trajectory

# Print the length of the trajectory list
print("Length of traj = ", len(traj))
print("### data loaded ###")

# Specify the number of validation samples and model dimensions
validation_num = 1
n = 40  # Number of states in the system
m = 12  # Number of control inputs
nr = 3  # Presumably related to the number of outputs, though unused in the code

# Directory where the Koopman model will be saved
model_save_dir = 'D:\\koopman_robot\\Koopman_optimization_rl_policy\\resourse\\koopman_model'

# Train the Koopman model iteratively on the trajectory data
if ITER_TRAIN_MATRIX:
    A_matrices = []
    B_matrices = []
    for i in range(len(traj)):
        traj_val = traj[i]
        traj_train = [traj[j] for j in range(len(traj)) if j != i]  # Use all trajectories except the current one for training
        look_forward = 1  # The lookahead step for the model (1 means predict the next state)
        dt = 0.01  # Time step for the simulation

        # Define and initialize the neural network-based Koopman model (NNDMDc)
        dlk_regressor = pk.regression.NNDMDc(
            mode='Dissipative_control', 
            n=n, m=m,
            dt=dt, look_forward=look_forward,
            config_encoder=dict(input_size=n, hidden_sizes=[256] * 5, output_size=128, activations='relu'),
            config_decoder=dict(input_size=128, hidden_sizes=[256] * 5, output_size=n, activations='relu'),
            batch_size=128, lbfgs=False, normalize=False, normalize_mode='equal',
            normalize_std_factor=1.0, include_state=True,
            trainer_kwargs=dict(max_epochs=5000)
        )
        model = pk.Koopman(regressor=dlk_regressor)

        # Fit the model to the training data
        model.fit(traj_train, dt=dt)
        print("### training finished ###")

        # Retrieve the Koopman matrices (A, B, K)
        A_discrete, A, B, K = model.get_Koopman_Operator()
        A_matrices.append(A)  # Store the A matrix from the model
        B_matrices.append(B)  # Store the B matrix from the model

        # Save the matrices as .npy files
        # (Uncomment the following lines if you wish to save them)
        # np.save("A_discrete.npy", A_discrete)
        # np.save("A.npy", A)
        # np.save("B.npy", B)
        # np.save("K.npy", K)
        # print("### Matrices saved ###")

        # Single-step validation: Evaluate the model by predicting one step ahead
        if PLOT_VALIDATION_SINGLESTEP:
            index_start_step = 0
            shortest_traj_length = 100000  # Find the shortest trajectory length for validation
            print(f"######## Validating single step for trajectory {i} ########")
            
            for j in range(validation_num):
                traj_i = traj_val[j]
                if traj_i.shape[0] < shortest_traj_length:
                    shortest_traj_length = traj_i.shape[0]
            print(f"### Shortest traj length = {shortest_traj_length} ###")

            # Single-step validation
            for k in range(10):  # For each validation step
                pred_X = np.zeros((n, validation_num))  # Predicted state variables
                real_X = np.zeros((n, validation_num))  # Real state variables

                for j in range(validation_num):  # Loop through the validation trajectories
                    traj_i = traj_val[j]
                    traj_i = np.atleast_2d(traj_i)  # Ensure that traj_i is at least 2D

                    if traj_i.shape[0] > index_start_step + 1:  # Ensure enough steps for prediction
                        real_X[:, j] = traj_i[index_start_step + 1, :n].flatten()  # True next state
                        pxT = traj_i[index_start_step, :n].reshape((1, n))  # Current state
                        uT = traj_i[index_start_step, n:].reshape((1, m))  # Control input

                        # Predict the next state using the Koopman model
                        pred_X[:, j] = model.predict(x=pxT, u=uT).reshape(n)
                    else:
                        print(f"Warning: traj_i has insufficient steps for trajectory {j}, skipping this trajectory.")
                        real_X[:, j] = np.zeros(n)
                        pred_X[:, j] = np.zeros(n)

                # Plot the prediction vs actual for selected indices
                fig, axs = plt.subplots(3, 1, figsize=(6, 12))
                index_x = [0, 6, -1]
                axs[0].plot(real_X[index_x[0], :], label='real X', color='red')
                axs[0].plot(pred_X[index_x[0], :], label='pred X', color='blue')
                axs[0].set_title(f'X index {index_x[0]}')
                axs[0].legend()
                axs[1].plot(real_X[index_x[1], :], label='real X', color='red')
                axs[1].plot(pred_X[index_x[1], :], label='pred X', color='blue')
                axs[1].set_title(f'X index {index_x[1]}')
                axs[1].legend()
                axs[2].plot(real_X[index_x[2], :], label='real X', color='red')
                axs[2].plot(pred_X[index_x[2], :], label='pred X', color='blue')
                axs[2].set_title(f'X index {index_x[2]}')
                axs[2].legend()
                plt.title(f"VALIDATING SINGLE STEP for trajectory {j}")
                plt.tight_layout()
                plt.show()

                # Increment the starting step for the next validation
                index_start_step += 1
                if index_start_step >= shortest_traj_length - 1:
                    print("##### traj_val used up! #####")
                    index_start_step -= 1

        # Perform testing if enabled
        if PLOT_TEST:
            ### testing ### 
            predict_horizon = 10
            index_traj = 0
            print("######## testing ########")

            for i in range(10):  # For each test iteration

                # Ensure that the selected trajectory is long enough for prediction
                while traj_train[index_traj].shape[0] < predict_horizon:
                    index_traj += 1
                traj_used = traj_train[index_traj]  # Get the trajectory data

                # Initialize arrays for predicted and real states
                pred_X = np.zeros((n, predict_horizon))
                real_X = np.zeros((n, predict_horizon))
                real_X[:, :] = traj_used[:predict_horizon, :n].T  # Real states
                pred_X[:, 0] = real_X[:, 0]  # Set the first prediction to match the real value

                # Predict the states for the entire horizon
                for j in range(1, predict_horizon):
                    pxT = traj_used[j - 1, :n].reshape((1, n))  # Current state
                    uT = traj_used[j - 1, n:].reshape((1, m))  # Control input
                    pred_X[:, j] = model.predict(x=pxT, u=uT).reshape(n)  # Predict next state

                # Plot the predicted vs real values for selected indices
                fig, axs = plt.subplots(3, 1, figsize=(6, 12))
                index_x = [0, 6, -1]
                axs[0].plot(real_X[index_x[0], :], label='real X', color='red')
                axs[0].plot(pred_X[index_x[0], :], label='pred X', color='blue')
                axs[0].set_title(f'X index {index_x[0]}')
                axs[0].legend()
                axs[1].plot(real_X[index_x[1], :], label='real X', color='red')
                axs[1].plot(pred_X[index_x[1], :], label='pred X', color='blue')
                axs[1].set_title(f'X index {index_x[1]}')
                axs[1].legend()
                axs[2].plot(real_X[index_x[2], :], label='real X', color='red')
                axs[2].plot(pred_X[index_x[2], :], label='pred X', color='blue')
                axs[2].set_title(f'X index {index_x[2]}')
                axs[2].legend()
                plt.title("TESTING")
                plt.tight_layout()
                plt.show()

                # Update trajectory index for the next test
                index_traj += 1
                if index_traj >= len(traj_train):
                    print("##### traj_train used up! #####")
                    index_traj -= 1

    # After iterating through all trajectories, concatenate the A and B matrices and save them as .mat files
    A_matrices_np = np.concatenate(A_matrices, axis=0)
    B_matrices_np = np.concatenate(B_matrices, axis=0)
    file2 = "A.mat"
    file3 = "B.mat"
    savemat(file2, {'A': A_matrices_np})  # Save matrix A
    savemat(file3, {'B': B_matrices_np})  # Save matrix B
    print(f"### Matrices saved to {file2}, {file3}")
