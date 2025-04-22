# dir import
from rsl_rl.env import VecEnv
from rsl_rl.runners import OnPolicyRunner
from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR

# lib import
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
# import cvxpy as cp
import pykoopman as pk
from pykoopman.common  import advance_linear_system
from pykoopman.regression import DMDRegressor
from pydmd import DMD
from pykoopman.regression._nndmdc import FFNN


class Koopman():
    def __init__(self, n, m, nr, K=1000, rank=3):
        np.set_printoptions(threshold=np.inf)
        self.n = n      # size of x
        self.m = m      # size of u
        self.nr = nr    # size of highlevel command r and r_hat
        self.K = K      # size of history snapshots
        self.rank = rank    # size of x in use

        # collect data to start Koopman method
        self.warm_up_counter = 0
        self.warm_up = False
        
        # Form Data Set X
        self.X = np.zeros((self.K+1, self.n))
        self.U = np.zeros((self.K, self.m))       
        self.last_x = np.zeros(self.n)
        self.next_u = np.zeros(self.m)

        # for prediction test
        self.div_step = self.K  # step before divergence
        self.div = False        # whether met divergence

        # train NNDMD only in first time 
        self.trained = False
        self.data_appended = False
        self.model = None
        self.look_forward = 1
    
        # data collection for one traj
        self.traj_start = True
        self.traj_cnt = 0
        self.traj_xu = []
        self.traj_xux = []
        self.traj = []
        

    def store_x_and_u(self, x, u):
        '''适用于[x_t]的存储'''
        if self.warm_up_counter < self.K:
            # load in data
            self.X[self.warm_up_counter, :] = x[:]
            self.U[self.warm_up_counter, :] = u[:]
        elif self.warm_up_counter == self.K:
            self.X[self.warm_up_counter, :] = x[:]
            self.next_u[:] = u[:]
        else:
            # shift matrix
            self.X[:-1, :] = self.X[1:, :]
            self.X[-1, :] = x[:].detach().numpy()
            self.U[:-1, :] = self.U[1:, :]
            self.U[-1, :] = self.next_u[:]
            self.next_u[:] = u[:]
            self.warm_up = True
        self.warm_up_counter += 1
        print(f"### process cnt: {self.warm_up_counter} ###")

        return self.warm_up, self.X, self.U


    def data_collection_for_one_traj_xu(self, x, u, termination):
        '''适用于一段段的traj的存储'''

        u_flatten = torch.flatten(u)
        xu = torch.cat((x, u_flatten)).numpy()
        self.traj_xu.append(xu)

        if self.warm_up_counter < self.K:
            if termination:
                xu_np = np.array(self.traj_xu)
                self.traj.append(xu_np)
                self.traj_xu = []
        else:
            xu_np = np.array(self.traj_xu)
            self.traj.append(xu_np)
            self.traj_xu = []
            self.warm_up = True

        self.warm_up_counter += 1
        print(f"### process cnt: {self.warm_up_counter} ###")
        return self.warm_up, self.traj


    def data_collection_for_one_traj_xux(self, x, u, x1, termination):
        '''存储[x, u, x_t+1]的pair'''

        u_flatten = torch.flatten(u)
        xux = torch.cat((x, u_flatten, x1)).numpy()

        if self.warm_up_counter < self.K:
            if termination:
                xux_np = np.array(self.traj_xux)
                self.traj.append(xux_np)
                self.traj_xux = []
            else:
                self.traj_xux.append(xux)
        else:
            xux_np = np.array(self.traj_xux)
            self.traj.append(xux_np)
            self.traj_xux = []
            self.warm_up = True

        self.warm_up_counter += 1
        print(f"### process cnt: {self.warm_up_counter} ###")
        return self.warm_up, self.traj


    def _get_traj(self):
        '''搭配data_collection_for_one_traj'''
        return self.traj


    def store_data(self, x, r):
        '''正常存储，适用于[x_t, r_t+1]的存储        '''
        p = torch.cat((x, r))
        if self.warm_up_counter < self.K:
            # load in data
            self.X[self.warm_up_counter, :] = p[:]
        else:
            # shift matrix
            self.X[:-1, :] = self.X[1:, :]
            self.X[-1, :] = p[:]
            self.warm_up = True
        self.warm_up_counter += 1
        print("####################################")
        print("process cnt: ", self.warm_up_counter)
        return self.warm_up, self.X


    def store_data_hat(self, x, r):
        '''错位存储，适用于[x_t, r_hat_t]的存储'''
        if self.warm_up_counter < self.K:
            if self.warm_up_counter == 0:
                # 第一次只存x
                self.X[self.warm_up_counter, :self.n] = x[:]
            else:
                # 第二到倒数第二次，存x和r
                self.X[self.warm_up_counter, :self.n] = x[:]
                self.X[self.warm_up_counter-1, self.n:] = r[:]
        elif self.warm_up_counter == self.K:
            # 最后一次，只存r，备用x
            self.last_x = x
            self.X[self.warm_up_counter-1, self.n:] = r[:]
            self.warm_up = True
        else:
            # 平移矩阵后存入x和r，备用x
            self.X[:-1, :] = self.X[1:, :]
            self.X[-1, :self.n] = self.last_x
            self.X[-1, self.n:] = r[:]
            self.last_x = x
        self.warm_up_counter += 1
        print("####################################")
        print("process cnt: ", self.warm_up_counter)
        return self.warm_up, self.X
  

    def step_koopman_dmd(self):  # using DMD alg
        if self.warm_up:
            # init
            dmd = DMD(svd_rank=self.rank)
            model = pk.Koopman(regressor=DMDRegressor(dmd))
            # solve problem
            model.fit(self.X[:, :self.rank])
            self.At = model.koopman_matrix

            # prediction test
            # self.pX = np.zeros((self.K, self.rank))
            # for i in range(self.K-1):
            #     pn = self.X[i, :self.rank]
            #     self.pX[i+1, :] = np.dot(self.At, pn)
            # fig, axes = plt.subplots(1, 3, figsize=(12, 8))
            # for i in range(3):
            #     ax = axes[i%3]
            #     ax.plot(self.X[1:, i],label=f'true x:{i}')
            #     ax.plot(self.pX[1:, i], label=f'DMD predicted x:{i}', linestyle='--')
            #     ax.set_title(f'DMD: index {i} & rank {self.rank}')
            #     ax.set_xlabel('Index')
            #     ax.set_ylabel('Velocity')
            #     ax.legend()
            # print("A = ", self.At)
            # print("rank of A = ", np.linalg.matrix_rank(self.At))
            # print("cond of A = ", np.linalg.cond(self.At))
            # plt.tight_layout()
            # plt.show()
        else:
            print('Haven\'t warmed up......')
        return self.At


    def step_koopman_dmdc(self): # using DMDc alg
        if self.warm_up:
            # init
            DMDc = pk.regression.DMDc(svd_rank=self.rank)
            model = pk.Koopman(regressor=DMDc)
            # set input U
            # solve problem
            model.fit(self.X[:, :self.rank], self.U)
            self.At = model.state_transition_matrix
            self.Bt = model.control_matrix
        else:
            print('Haven\'t warmed up......')
        return self.At, self.Bt


    def step_koopman_edmd(self, observable='IDENTITY', rbf_type='gauss'): # using EDMD alg with various observables
        self.observable = observable
        self.rbf_type = rbf_type
        if self.warm_up:
            if self.observable == 'IDENTITY':
                # Identity observables
                self.ob = pk.observables.Identity()
            elif self.observable == 'TRIGONO':
                # Sin, Cos observables
                tri_observables = [lambda x: np.sin(x), lambda x: np.cos(x)]
                tri_observable_names = [
                    lambda s: f"sin({s})",
                    lambda s: f"cos({s})",
                ]
                self.ob = pk.observables.CustomObservables(tri_observables, observable_names=tri_observable_names)
            elif self.observable == 'RBF':
                # RBF observables
                centers = np.random.uniform(-1,1,(self.rank, 6))    # decides the fit ability of the model (upper dim)
                self.ob = pk.observables.RadialBasisFunction(
                    rbf_type=self.rbf_type,
                    n_centers=centers.shape[1],
                    centers=centers,
                    kernel_width=1,
                    polyharmonic_coeff=1.0,
                    include_state=True,
                )
            else:
                raise NotImplementedError

            # solve problem
            EDMD = pk.regression.EDMD()
            model = pk.Koopman(observables=self.ob, regressor=EDMD)
            model.fit(x=self.X[:, :self.rank])  # U ?
            
            ### prediction test
            self.pX = np.zeros((self.K, self.rank))   # 1-step prediction of X1
            self.ppX = np.zeros((self.K+1, self.rank))    # propagation prediction of X1
            self.ppX[0, :] = self.X[0, :self.rank]      # init propagation
            for i in range(self.K):
                xT = self.X[i, :self.rank].reshape((1, self.rank))
                self.pX[i, :] = model.predict(x=xT)

                # not working!!!
                if not np.isnan(self.ppX[i, :]).any() and not self.div:
                    # print("### not NAN yet at ", i)
                    pxT = self.ppX[i, :].reshape((1, self.rank))
                    self.ppX[i+1, :] = model.predict(x=pxT)
                    if np.isnan(self.ppX[i+1, :]).any():
                        self.div = True
                        self.div_step = i+1
                        print("### div_step = ", self.div_step)
                        print("### the place of Nan: ", np.isnan(self.ppX[i+1, :]))

            fig, axes = plt.subplots(1, 3, figsize=(12, 8))
            for i in range(3):
                ax = axes[i%3]
                ax.plot(self.X[1:, i], label=f'true x:{i}')
                ax.plot(self.pX[:, i], label=f'predicted x:{i}', linestyle='--')
                ax.plot(self.ppX[1:self.div_step, i], label=f'propagated x:{i}', linestyle=':')
                ax.set_title(f'EDMD with IDENTITY: index {i} & rank {self.rank}')
                ax.set_xlabel('Index')
                ax.set_ylabel('Velocity')
                ax.legend()
            plt.show()
            self.p_error_square = np.square(self.pX[:, 0] - self.X[1:, 0])
            self.p_error = np.sum(self.p_error_square, axis=0)
            print("### prediction error square = ", self.p_error/self.K)
            self.pp_error_square = np.square(self.ppX[1:, 0] - self.X[1:, 0])
            self.pp_error = np.sum(self.pp_error_square, axis=0)
            print("### propagation error square = ", self.pp_error/self.K)
        else:
            print('Haven\'t warmed up......')
        return model


    def step_koopman_edmdc(self, observable='IDENTITY', rbf_type='gauss'): # using EDMD alg with various observables
        self.observable = observable
        self.rbf_type = rbf_type
        if self.warm_up:
            if self.observable == 'IDENTITY':
                # Identity observables
                self.ob = pk.observables.Identity()
            elif self.observable == 'TRIGONO':
                # Sin, Cos observables
                tri_observables = [lambda x: np.sin(x), lambda x: np.cos(x)]
                tri_observable_names = [
                    lambda s: f"sin({s})",
                    lambda s: f"cos({s})",
                ]
                self.ob = pk.observables.CustomObservables(tri_observables, observable_names=tri_observable_names)
            elif self.observable == 'RBF':
                # RBF observables
                centers = np.random.uniform(-1,1,(self.rank, 10))    # decides the fit ability of the model (upper dim)
                self.ob = pk.observables.RadialBasisFunction(
                    rbf_type=self.rbf_type,
                    n_centers=centers.shape[1],
                    centers=centers,
                    kernel_width=1,
                    polyharmonic_coeff=1.0,
                    include_state=True,
                )
            else:
                raise NotImplementedError

            # solve problem
            EDMDc = pk.regression.EDMDc()
            model = pk.Koopman(observables=self.ob, regressor=EDMDc)
            model.fit(x=self.X[:, :self.rank], u=self.U)
            
            ### prediction test
            # self.pX = np.zeros((self.K, self.rank))   # 1-step prediction of X1
            # self.ppX = np.zeros((self.K+1, self.rank))    # propagation prediction of X1
            # self.ppX[0, :] = self.X[0, :self.rank]      # init propagation
            # for i in range(self.K):
            #     xT = self.X[i, :self.rank].reshape((1, self.rank))
            #     uT = self.U[i, :].reshape((1, self.m))
            #     self.pX[i, :] = model.predict(x=xT, u=uT)

            #     # not working!!!
            #     if not np.isnan(self.ppX[i, :]).any() and not self.div:
            #         # print("### not NAN yet at ", i)
            #         pxT = self.ppX[i, :].reshape((1, self.rank))
            #         uT = self.U[i, :].reshape((1, self.m))
            #         self.ppX[i+1, :] = model.predict(x=pxT, u=uT)
            #         if np.isnan(self.ppX[i+1, :]).any():
            #             self.div = True
            #             self.div_step = i+1
            #             print("### div_step = ", self.div_step)
            #             print("### the place of Nan: ", np.isnan(self.ppX[i+1, :]))

            # fig, axes = plt.subplots(1, 3, figsize=(12, 8))
            # for i in range(3):
            #     ax = axes[i%3]
            #     ax.plot(self.X[1:, i], label=f'true x:{i}')
            #     ax.plot(self.pX[:, i], label=f'predicted x:{i}', linestyle='--')
            #     ax.plot(self.ppX[1:51, i], label=f'propagated x:{i}', linestyle=':')
            #     ax.set_title(f'EDMDc with IDENTITY: index {i} & rank {self.rank}')
            #     ax.set_xlabel('Index')
            #     ax.set_ylabel('Velocity')
            #     ax.legend()
            # plt.show()
            # self.p_error_square = np.square(self.pX[:, 0] - self.X[1:, 0])
            # self.p_error = np.sum(self.p_error_square, axis=0)
            # print("### prediction error square = ", self.p_error/(self.K+1))
            # self.pp_error_square = np.square(self.ppX[1:, 0] - self.X[1:, 0])
            # self.pp_error = np.sum(self.pp_error_square, axis=0)
            # print("### propagation error square = ", self.pp_error/(self.K+1))
        else:
            print('Haven\'t warmed up......')
        return model


    def step_koopman_nndmd(self, dt):
        if self.warm_up:
            self.look_forward = 10
            if not self.data_appended:
                traj_list = []
                traj_list.append(self.X)
                self.data_appended = True

            if not self.trained:
                dlk_regressor = pk.regression.NNDMD(mode='Dissipative',
                                                    dt=dt, look_forward=self.look_forward,
                                                    config_encoder=dict(input_size=self.rank,
                                                                        hidden_sizes=[256] * 10,
                                                                        output_size=128,
                                                                        activations='relu'),
                                                    config_decoder=dict(input_size=128,
                                                                        hidden_sizes=[256] * 10,
                                                                        output_size=self.rank,
                                                                        activations='relu'),
                                                    batch_size=128, lbfgs=False,
                                                    normalize=True, normalize_mode='equal',
                                                    normalize_std_factor=1.0,
                                                    include_state=True,
                                                    trainer_kwargs=dict(max_epochs=10))

                self.model = pk.Koopman(regressor=dlk_regressor)
                self.model.fit(traj_list, dt=dt) 
                # print("###########################################")
                # print('### shape of trained koopman model = ', dlk_regressor.A.shape)
                # print("###########################################")
                print("###########################################")
                print(' what is C ?', dlk_regressor.C)
                print('### shape of trained C model = ', dlk_regressor.C.shape)
                print("###########################################")
                # print("###########################################")
                # print('### shape of trained W model = ', dlk_regressor.W.shape)
                # print("###########################################")
                self.trained = True
            else:
                print('# Lifting function already trained, did nothing this time #')
        else:
            print('Haven\'t warmed up......')

        return self.model
    

    def step_koopman_nndmdc(self, dt, traj_list=None, use_trained_K=False, model_save_dir=None):
        if use_trained_K:
            if not self.trained:
                # prepare koopman model
                self.nndmdc_model = NNDMDc_Module(n=self.n, m=self.m, nr=self.nr, dt=0.01, 
                                        config_encoder=dict(input_size=self.n,
                                                            hidden_sizes=[256] * 5,
                                                            output_size=128,
                                                            activations='relu'),
                                        config_decoder=dict(input_size=128,
                                                            hidden_sizes=[256] * 5,
                                                            output_size=self.n,
                                                            activations='relu'))
                self.nndmdc_model.load_model(model_save_dir)
                self.trained = True
            return self.nndmdc_model
        else:
            if not self.trained:
                dlk_regressor = pk.regression.NNDMDc(mode='Dissipative_control',
                                                    n=self.n, m=self.m,
                                                    dt=dt, look_forward=self.look_forward,
                                                    config_encoder=dict(input_size=self.n,
                                                                        hidden_sizes=[256] * 10,
                                                                        output_size=128,
                                                                        activations='relu'),
                                                    config_decoder=dict(input_size=128,
                                                                        hidden_sizes=[256] * 10,
                                                                        output_size=self.n,
                                                                        activations='relu'),
                                                    batch_size=128, lbfgs=False,
                                                    normalize=False, normalize_mode='equal',
                                                    normalize_std_factor=1.0,
                                                    include_state=True,
                                                    trainer_kwargs=dict(max_epochs=50))
                self.model = pk.Koopman(regressor=dlk_regressor)
                if traj_list is None:
                    raise ValueError("The variable 'traj_list' cannot be None.")
                self.model.fit(traj_list, dt=dt)
                self.trained = True
            return self.model



class NNDMDc_Module():
    def __init__(self, n, m, nr, dt=1.0,
                 config_encoder=dict(), config_decoder=dict()):
        
        np.set_printoptions(threshold=np.inf)
        self.n = n      # size of x
        self.m = m      # size of u
        self.nr = nr    # size of highlevel command r and r_hat
        self.dt = dt
        self.dim = config_encoder["output_size"] + self.m
        self.encoder_input_size = config_encoder["input_size"]
        self.encoder_output_size = config_encoder["output_size"]
        self.decoder_input_size = config_decoder["input_size"]
        self.decoder_output_size = config_decoder["output_size"]

        self._encoder = FFNN(
            input_size=config_encoder["input_size"],
            hidden_sizes=config_encoder["hidden_sizes"],
            output_size=config_encoder["output_size"],
            activations=config_encoder["activations"],
        )
        self._decoder = FFNN(
            input_size=config_decoder["input_size"],
            hidden_sizes=config_decoder["hidden_sizes"],
            output_size=config_decoder["output_size"],
            activations=config_decoder["activations"],
        )
        print("### successfully initialized ! ###")


    def load_model(self, model_save_dir):
        """
        从指定目录加载模型参数和矩阵。
        
        参数：
            model_save_dir (str): 模型保存的文件夹路径，其中包含
                                - encoder.pth
                                - decoder.pth
                                - K_state.npy
                                - K_action.npy
        """
        self._encoder.load_state_dict(torch.load(os.path.join(model_save_dir, "encoder.pth")))
        self._decoder.load_state_dict(torch.load(os.path.join(model_save_dir, "decoder.pth")))
        K_state = np.load(os.path.join(model_save_dir, "K_state.npy"))
        K_action = np.load(os.path.join(model_save_dir, "K_action.npy"))
        K_full = np.hstack((K_state, K_action))  # 拼接完整矩阵 [K_state, K_action]
        self._K = torch.from_numpy(K_full).float()

        self._encoder.eval()
        self._decoder.eval()
        for param in self._encoder.parameters():
            param.requires_grad = False
        for param in self._decoder.parameters():
            param.requires_grad = False

        print("### Model and matrices successfully loaded from: {} ###".format(model_save_dir))

    def predict(self, x, u):
        """
        前向传播函数，确保 x 和 u 是 1D 的 torch.float32 类型张量。
        """
        if not isinstance(x, torch.Tensor) or x.dtype != torch.float32:
            x = torch.tensor(x, dtype=torch.float32)
        if not isinstance(u, torch.Tensor) or u.dtype != torch.float32:
            u = torch.tensor(u, dtype=torch.float32)
        
        if x.dim() == 1 and u.dim() == 1:
            encoded_state = self._encoder(x)
            encoded = torch.cat((encoded_state, u), dim=0)
            forwarded = torch.matmul(self._K, encoded)
            decoded = self._decoder(forwarded)
        else:

            encoded_state = self._encoder(x)
            encoded = torch.cat((encoded_state, u), dim=1)
            forwarded = torch.zeros((x.shape[0], self.decoder_input_size))
            for i in range(x.shape[0]):
                forwarded[i, :] = torch.matmul(self._K, encoded[i, :])
            decoded = self._decoder(forwarded)

        return decoded


