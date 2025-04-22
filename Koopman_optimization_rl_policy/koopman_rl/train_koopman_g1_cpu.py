#--------------- Junlin Wu 2025/4/20 ---------------
#--------------- Koopman Training for G1 (CPU Fixed) ---------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pykoopman as pk
from scipy.io import savemat
from g1_utils import get_data_path, GROUND_TYPES
import torch

# 强制 CPU
torch.cuda.is_available = lambda: False
torch.Tensor.cuda = lambda self, *args, **kwargs: self
torch.nn.Module.cuda = lambda self, *args, **kwargs: self

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
            trainer_kwargs=dict(max_epochs=50)
        )
        model = pk.Koopman(regressor=dlk_regressor)

        # 训练模型
        model.fit(traj_train, dt=dt)
        print(f"### Training finished for trajectory {i}/{len(traj)} ###")

        # 获取 Koopman 算子
        A_discrete, A, B, K = model.get_Koopman_Operator()
        A_matrices.append(A)
        B_matrices.append(B)

        # 单步验证
        if PLOT_VALIDATION_SINGLESTEP:
            index_start_step = 0
            shortest_traj_length = traj_val.shape[0]
            print(f"######## Validating single step for trajectory {i} ########")

            for k in range(5):
                pred_X = np.zeros((n, validation_num))
                real_X = np.zeros((n, validation_num))

                if traj_val.shape[0] > index_start_step + 1:
                    real_X[:, 0] = traj_val[index_start_step + 1, :n].flatten()
                    pxT = torch.tensor(traj_val[index_start_step, :n].reshape((1, n)), dtype=torch.float32)
                    uT = torch.tensor(traj_val[index_start_step, n:n+m].reshape((1, m)), dtype=torch.float32)
                    pred_X[:, 0] = model.predict(x=pxT, u=uT).reshape(n)
                else:
                    print(f"Warning: traj_val has insufficient steps at index {index_start_step}")
                    real_X[:, 0] = np.zeros(n)
                    pred_X[:, 0] = np.zeros(n)

                fig, axs = plt.subplots(3, 1, figsize=(6, 12))
                index_x = [0, 6, n-1]
                for idx, ax in enumerate(axs):
                    ax.plot(real_X[index_x[idx], :], label='real X', color='red')
                    ax.plot(pred_X[index_x[idx], :], label='pred X', color='blue')
                    ax.set_title(f'X index {index_x[idx]}')
                    ax.legend()
                plt.suptitle(f"Validating Single Step for Trajectory {i}")
                plt.tight_layout()
                plt.savefig(f"{model_save_dir}\\validation_traj_{i}_step_{k}.png")
                plt.close()

                index_start_step += 1
                if index_start_step >= shortest_traj_length - 1:
                    print("##### traj_val used up! #####")
                    break

    # 保存矩阵
    A_matrices_np = np.concatenate(A_matrices, axis=0)
    B_matrices_np = np.concatenate(B_matrices, axis=0)
    file2 = f"{model_save_dir}\\A_{ground_type}.mat"
    file3 = f"{model_save_dir}\\B_{ground_type}.mat"
    savemat(file2, {'A': A_matrices_np})
    savemat(file3, {'B': B_matrices_np})
    print(f"### Matrices saved to {file2}, {file3}")