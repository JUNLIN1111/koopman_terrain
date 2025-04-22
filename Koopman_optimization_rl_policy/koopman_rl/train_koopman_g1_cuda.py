#--------------- Junlin Wu 2025/4/22 ---------------
#--------------- Koopman Training for G1 (CUDA Enabled) ---------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pykoopman as pk
from scipy.io import savemat
from g1_utils import get_data_path, GROUND_TYPES
import torch

# 检查 CUDA 可用性并设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 配置参数
ground_type = "sand"
PLOT_TEST = False
PLOT_VALIDATION = False
PLOT_VALIDATION_SINGLESTEP = True
ITER_TRAIN_MATRIX = True
np.set_printoptions(threshold=np.inf)

# 加载数据
file_path = get_data_path(ground_type)
csv_data = pd.read_csv(file_path)
merged_data = csv_data.values

# 构造特征矩阵
needed_data = np.zeros_like(merged_data)
needed_data[:, :13] = merged_data[:, :13]
needed_data[:, 13:16] = merged_data[:, 13:16]
needed_data[:, 16:28] = merged_data[:, 16:28]
needed_data[:, 28:40] = merged_data[:, 28:40]
needed_data[:, 40:] = merged_data[:, 40:]

print(f"Needed Data Shape = {needed_data.shape}")

# 分割轨迹
traj = []
number_of_100steps = needed_data.shape[0] // 100
for i in range(number_of_100steps):
    data_clip = needed_data[i*100:(i+1)*100, :]
    traj.append(data_clip)
if needed_data.shape[0] % 100 != 0:
    traj.append(needed_data[-100:, :])

print(f"Length of traj = {len(traj)}")
print(f"### {ground_type} data loaded ###")

# 模型参数
validation_num = 1
n = 40
m = 12
dt = 0.002

# 输出目录
model_save_dir = r"D:\koopman_robot\Koopman_optimization_rl_policy\resourse\koopman_model"

# 迭代训练
if ITER_TRAIN_MATRIX:
    A_matrices = []
    B_matrices = []
    for i in range(len(traj)):
        traj_val = traj[i]
        traj_train = [traj[j] for j in range(len(traj)) if j != i]
        look_forward = 1

        # 初始化 NNDMDc
        dlk_regressor = pk.regression.NNDMDc(
            mode='Dissipative_control',
            n=n, m=m,
            dt=dt, look_forward=look_forward,
            config_encoder=dict(input_size=n, hidden_sizes=[256] * 5, output_size=128, activations='relu'),
            config_decoder=dict(input_size=128, hidden_sizes=[256] * 5, output_size=n, activations='relu'),
            batch_size=64, lbfgs=False, normalize=False, normalize_mode='equal',
            normalize_std_factor=1.0, include_state=True,
            trainer_kwargs=dict(max_epochs=50, accelerator='gpu' if torch.cuda.is_available() else 'cpu', devices=1)
        )
        model = pk.Koopman(regressor=dlk_regressor)

        # 将训练数据转换为 Tensor 并移动到 GPU
        traj_train_tensors = [torch.tensor(t, dtype=torch.float32).to(device) for t in traj_train]

        # 训练模型
        model.fit(traj_train_tensors, dt=dt)
        print(f"### Training finished for trajectory {i}/{len(traj)} ###")

        # 保存编码器和解码器（确保移动到 CPU 保存）
        model.regressor.net.encoder.to("cpu")
        model.regressor.net.decoder.to("cpu")
        torch.save(model.regressor.net.encoder.state_dict(), f"{model_save_dir}\\encoder_sand_traj_{i}.pth")
        torch.save(model.regressor.net.decoder.state_dict(), f"{model_save_dir}\\decoder_sand_traj_{i}.pth")
        print(f"### Saved encoder and decoder for trajectory {i} ###")

        # 单步验证
        if PLOT_VALIDATION_SINGLESTEP:
            index_start_step = 0
            shortest_traj_length = traj_val.shape[0]
            print(f"######## Validating single step for trajectory {i} ########")
            print(f"traj_val shape: {traj_val.shape}")

            for k in range(5):
                pred_X = np.zeros((n, validation_num))
                real_X = np.zeros((n, validation_num))

                for t in range(validation_num):
                    if traj_val.shape[0] > index_start_step + t + 1:
                        print(f"traj_val[{index_start_step + t}, :n]: {traj_val[index_start_step + t, :n]}")
                        print(f"traj_val[{index_start_step + t + 1}, :n]: {traj_val[index_start_step + t + 1, :n]}")
                        real_X[:, t] = traj_val[index_start_step + t + 1, :n].flatten()
                        pxT = torch.tensor(traj_val[index_start_step + t, :n].reshape((1, n)), dtype=torch.float32).to(device)
                        uT = torch.tensor(traj_val[index_start_step + t, n:n+m].reshape((1, m)), dtype=torch.float32).to(device)
                        pred = model.predict(x=pxT, u=uT).cpu().reshape(n)  # 预测结果移回 CPU
                        print(f"Predicted X at step {t}: {pred}")
                        pred_X[:, t] = pred
                    else:
                        print(f"Warning: traj_val has insufficient steps at index {index_start_step + t}")
                        break

                fig, axs = plt.subplots(3, 1, figsize=(6, 12))
                index_x = [0, 6, n-1]
                for idx, ax in enumerate(axs):
                    ax.plot(range(validation_num), real_X[index_x[idx], :], label='real X', color='red')
                    ax.plot(range(validation_num), pred_X[index_x[idx], :], label='pred X', color='blue')
                    ax.scatter(range(validation_num), real_X[index_x[idx], :], color='red', s=20)
                    ax.scatter(range(validation_num), pred_X[index_x[idx], :], color='blue', s=20)
                    ax.set_title(f'X index {index_x[idx]}')
                    ax.set_xlabel('Step')
                    ax.set_ylabel('Value')
                    ax.legend()
                    ax.grid(True)
                plt.suptitle(f"Validating Single Step for Trajectory {i}")
                plt.tight_layout()
                plt.savefig(f"{model_save_dir}\\validation_traj_{i}_step_{k}.png")
                plt.close()

                index_start_step += validation_num
                if index_start_step >= shortest_traj_length - validation_num:
                    print("##### traj_val used up! #####")
                    break

        # 获取 Koopman 算子
        A_discrete, A, B, K = model.get_Koopman_Operator()
        A_matrices.append(A)
        B_matrices.append(B)

    # 保存矩阵
    A_matrices_np = np.stack(A_matrices, axis=0)
    B_matrices_np = np.stack(B_matrices, axis=0)
    file2 = f"{model_save_dir}\\A_{ground_type}.mat"
    file3 = f"{model_save_dir}\\B_{ground_type}.mat"
    savemat(file2, {'A': A_matrices_np})
    savemat(file3, {'B': B_matrices_np})
    print(f"### Matrices saved to {file2}, {file3}")