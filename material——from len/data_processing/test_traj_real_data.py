import h5py
import pandas as pd
import ast
import re
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import pykoopman as pk
from pykoopman.regression import DMDRegressor
from pydmd import DMD
from scipy.io import savemat
from legged_gym.kmpc.koopman import NNDMDc_Module

### funciton
PLOT_TEST = False
PLOT_VALIDATION = False
PLOT_VALIDATION_SINGLESTEP = False

ITER_TRAIN_MATRIX = True

np.set_printoptions(threshold=np.inf)

### load data
file_path = 'output9.csv'
csv_data = pd.read_csv(file_path)
all_columns_data = []
for col in csv_data.columns:
    if csv_data[col].dtype == 'object':
        try:
            csv_data[col] = csv_data[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) and (x.startswith('[') or x.startswith('(')) else x)
        except Exception as e:
            print(f"Error parsing column {col}: {e}")
        if isinstance(csv_data[col].iloc[0], (list, np.ndarray, tuple)):
            expanded_columns = np.array([np.array(val) if isinstance(val, (list, np.ndarray, tuple)) else np.array([val]) for val in csv_data[col]])
            all_columns_data.append(expanded_columns)
        else:
            all_columns_data.append(csv_data[col].values.reshape(-1, 1))
    else:
        all_columns_data.append(csv_data[col].values.reshape(-1, 1))
merged_data = np.hstack(all_columns_data)
merged_data = merged_data[:, :-9]
# print("Merged Data Shape:", merged_data.shape)
# print(merged_data[0])

q_indices = np.arange(14, 14 + 10 * 12, 10) # give Arithmetic Sequence from 14
dq_indices = np.arange(15, 15 + 10 * 12, 10) # from 15
tauE_indices = np.arange(17, 17 + 10 * 12, 10) # from 17

needed_data = np.zeros((merged_data.shape[0], 52))
needed_data[:, :3] = merged_data[:, -3:] # put first 3 columns to last 3 columns
needed_data[:, 3:16] = merged_data[:, :13] 
needed_data[:, 16:28] = merged_data[:, q_indices] 
needed_data[:, 28:40] = merged_data[:, dq_indices]
needed_data[:, -12:] = merged_data[:, tauE_indices]

print("Needed Data Shape = ", needed_data.shape)
traj = []
number_of_100steps = needed_data.shape[0] // 100
for i in range(number_of_100steps):
    data_clip = np.zeros((100, 52))
    data_clip = needed_data[i:i+100, :]
    traj.append(data_clip)
traj.append(needed_data[-100:, :])
print("Length of traj = ", len(traj))
print("### data loaded ###")


### split test data and validation data
validation_num = 1
n = 40
m = 12
nr = 3

### train network

### train or load koopman model
model_save_dir = '/home/ltx/Koopman_Project/legged_gym-master/legged_gym/scripts/model'

### iteratively train koopman matrix
if ITER_TRAIN_MATRIX:
    A_matrices = []
    B_matrices = []
    for i in range(len(traj)):

        ### Initialize Training
        traj_val = traj[i]
        traj_train = [traj[j] for j in range(len(traj)) if j != i]
        look_forward = 1
        dt = 0.01
        dlk_regressor = pk.regression.NNDMDc(mode='Dissipative_control',
                                                n=n, m=m,
                                                dt=dt, look_forward=look_forward,
                                                config_encoder=dict(input_size=n,
                                                                    hidden_sizes=[256] * 5,
                                                                    output_size=128,
                                                                    activations='relu'),
                                                config_decoder=dict(input_size=128,
                                                                    hidden_sizes=[256] * 5,
                                                                    output_size=n,
                                                                    activations='relu'),
                                                batch_size=128, lbfgs=False,
                                                normalize=False, normalize_mode='equal',
                                                normalize_std_factor=1.0,
                                                include_state=True,
                                                trainer_kwargs=dict(max_epochs=5000))
        model = pk.Koopman(regressor=dlk_regressor)

        ### Train Model
        model.fit(traj_train, dt=dt)
        print("### training finished ###")

        ### Save Koopman Matrix
        A_discrete, A, B, K = model.get_Koopman_Operator()
        A_matrices.append(A)
        B_matrices.append(B)

        # store the matrix
        # file1 = "A_discrete.npy"
        # file2 = "A.npy"
        # file3 = "B.npy"
        # file4 = "K.npy"
        # np.save(file1, A_discrete)
        # np.save(file2, A)
        # np.save(file3, B)
        # np.save(file4, K)
        # print("### Matrices saved ###")

        ### 单步验证：只对当前训练好的模型进行验证和绘图
        if PLOT_VALIDATION_SINGLESTEP:
            index_start_step = 0
            shortest_traj_length = 100000
            print(f"######## Validating single step for trajectory {i} ########")
            
            # 计算最短轨迹长度
            for j in range(validation_num):
                traj_i = traj_val[j]
                if traj_i.shape[0] < shortest_traj_length:
                    shortest_traj_length = traj_i.shape[0]
            print(f"### Shortest traj length = {shortest_traj_length} ###")

            # 单步验证
            for k in range(10):
                pred_X = np.zeros((n, validation_num))
                real_X = np.zeros((n, validation_num))

                index_start_step = 1  # 起始步骤

                for j in range(validation_num):
                    traj_i = traj_val[j]
                    traj_i = np.atleast_2d(traj_i)  # 确保 traj_i 至少是二维数组

                    # 确保 traj_i 的长度足够以进行索引
                    if traj_i.shape[0] > index_start_step + 1:
                        # 仅当数据足够长时进行后续操作
                        real_X[:, j] = traj_i[index_start_step+1, :n].flatten()  # 这里修改为flatten()
                        pxT = traj_i[index_start_step, :n].reshape((1, n))
                        uT = traj_i[index_start_step, n:].reshape((1, m))

                        # 预测数据
                        pred_X[:, j] = model.predict(x=pxT, u=uT).reshape(n)
                    else:
                        print(f"Warning: traj_i has insufficient steps for trajectory {j}, skipping this trajectory.")
                        # 如果轨迹数据不足，使用零数组或跳过
                        real_X[:, j] = np.zeros(n)
                        pred_X[:, j] = np.zeros(n)

                # 绘制预测与实际值的对比
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

                index_start_step += 1
                if index_start_step >= shortest_traj_length - 1:
                    print("##### traj_val used up! #####")
                    index_start_step -= 1


        if PLOT_TEST:
            ### testing ###
            predict_horizon = 10
            index_traj = 0
            print("######## testing ########")

            for i in range(10):

                # 确保选择的轨迹数据足够长
                while traj_train[index_traj].shape[0] < predict_horizon:
                    index_traj += 1
                traj_used = traj_train[index_traj]  # (len, 27)

                # 初始化预测与实际值数组
                pred_X = np.zeros((n, predict_horizon))  # (21, len)
                real_X = np.zeros((n, predict_horizon))  # (21, len)
                real_X[:, :] = traj_used[:predict_horizon, :n].T
                pred_X[:, 0] = real_X[:, 0]

                # 进行预测
                for j in range(1, predict_horizon):
                    pxT = traj_used[j-1, :n].reshape((1, n))
                    uT = traj_used[j-1, n:].reshape((1, m))
                    pred_X[:, j] = model.predict(x=pxT, u=uT).reshape(n)

                # 绘制预测与实际值的对比
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

                # 更新轨迹索引
                index_traj += 1
                if index_traj >= len(traj_train):
                    print("##### traj_train used up! #####")
                    index_traj -= 1





    
    A_matrices_np = np.concatenate(A_matrices, axis=0)
    B_matrices_np = np.concatenate(B_matrices, axis=0)
    file2 = "A.mat"
    file3 = "B.mat"
    savemat(file2, {'A': A_matrices_np})
    savemat(file3, {'B': B_matrices_np})
    print(f"### Matrices saved to {file2}, {file3}")
