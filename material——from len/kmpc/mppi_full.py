# dir import
from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
from legged_gym.envs import *
# for policy
from rsl_rl.modules import ActorCritic
# lib import
import torch
import numpy as np
import matplotlib.pyplot as plt
from pytorch_mppi import mppi
import pinocchio as pin
import pykoopman as pk
from pykoopman.common  import advance_linear_system
from pykoopman.regression import DMDRegressor
from pydmd import DMD
from legged_gym.kmpc.koopman import NNDMDc_Module

# Mode choice ( True: u as the full control; False = du as the delta control )
FULL_MODE = True

# hard constraints
dU_MAX = 1.
dU_MIN = -dU_MAX
U_MAX = 10.
U_MIN = -U_MAX
f_max = 100
f_min = 0

class MPPIController():
    def __init__(self, n, m, nr, cfg, T=50, R=10):
        ''' init parameters for mppi '''
        self.n = n      # size of state
        self.m = m      # size of action 
        self.nr = nr    # size of command
        self.T = T      # prediction horizon
        self.R = R      # num of rollouts
        self.cfg = cfg

        self.d = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.dtype = torch.float32
        # iden = (dU_MAX/2) * torch.ones(self.m)   # for sampling delta_u for du
        iden = 5 * torch.ones(self.m)   # for sampling delta_u for u
        self.noise_sigma = torch.diag(torch.tensor(iden, dtype=self.dtype, device=self.d))  # noise set
        self.lambda_ = 2.   # λ in mppi

        self.sf = self.cfg.terrain.static_friction
        self.df = self.cfg.terrain.dynamic_friction

        self.u_nominal = torch.zeros((self.R, self.m),device=self.d)
        self.next_x = torch.zeros((self.R, self.n), device=self.d)

        ### load policy & urdf
        self.load_policy()
        self.init_urdf_file()

        ### load koopman model: NNDMDc:
        self.kmodel = NNDMDc_Module(n=self.n, m=self.m, nr=self.nr, dt=0.01, 
                                    config_encoder=dict(input_size=self.n,
                                                        hidden_sizes=[256] * 5,
                                                        output_size=128,
                                                        activations='relu'),
                                    config_decoder=dict(input_size=128,
                                                        hidden_sizes=[256] * 5,
                                                        output_size=self.n,
                                                        activations='relu'))

        ### init dynamics matrix
        self.Ri = torch.zeros((4, 3, 3))        # 每个维度对应一条腿
        self.Ji = torch.zeros((4, 3, 3))        

    ########################
    ##### compute cost #####
    ########################

    def running_cost(self, x, du): 
        cost = torch.zeros(self.R, device=self.d)
        lin_vel = x[:, :2].to('cuda')
        lin_vel_command = self.r[:2]
        projected_gravity = x[:, 6:9]
        
        # print("u shape = ", du.shape)
        
        u_cost = torch.sum(torch.square(du), dim=-1)
        ori_cost = torch.sum(torch.square(projected_gravity[:]), dim=-1)   
        ref_cost = torch.sum(torch.square(lin_vel - lin_vel_command), dim=-1)

        # friction_constraints = self._friction_constraints(x, du)

        cost = ref_cost + u_cost
        # cost = friction_constraints
        print("#### cost ####")
        # print(f"## u_cost = {u_cost[0]} ## ref_cost = {ref_cost[0]} ## constraints = f{friction_constraints[0]} ##")
        return cost
    
    
    def _friction_constraints(self, x, du):
        ### friction constraints
        torques = self._PD_gain(x, du)
        cost_f = torch.zeros((self.R, 4), device=self.d)

        forces = self._transform_forces(x, du, torques)
        fx = torch.zeros((self.R, 4), device=self.d)
        fy = torch.zeros((self.R, 4), device=self.d)
        fz = torch.zeros((self.R, 4), device=self.d)

        fx = forces[:, :, 0]
        fy = forces[:, :, 1]
        fz = forces[:, :, 2]

        cost_z1 = torch.relu(fz - f_max)  # fz > f_max 时的违约成本
        cost_z2 = torch.relu(f_min - fz)  # fz < f_min 时的违约成本
        cost_x = torch.relu(torch.abs(fx) - self.df * torch.abs(fz))
        cost_y = torch.relu(torch.abs(fy) - self.df * torch.abs(fz))
        cost_f[:, 0] += cost_z1.sum(dim=1)  # fz > f_max 的总违约成本
        cost_f[:, 1] += cost_z2.sum(dim=1)  # fz < f_min 的总违约成本
        cost_f[:, 2] += cost_x.sum(dim=1)   # x方向摩擦力约束
        cost_f[:, 3] += cost_y.sum(dim=1)   # y方向摩擦力约束

        soft_constraints = cost_f.sum(dim=-1)
        return soft_constraints

    def _transform_forces(self, x, du, torques):
        '''针对多个rollout'''
        # torques尺寸是[1000, 12]
        jacobians = torch.zeros((self.R, 3, 12), device=self.d)
        for i in range(self.R):     # 针对于每一个rollout分别计算
            xi = torch.zeros(self.n, device=self.d)
            xi[:] = x[i, :]
            jacobians[i, :, :] = self._compute_jacobian(xi)
        rotation_matrix = self._compute_rotation_matrix(x)
        f = torch.zeros((self.R, 4, 3), device=self.d)
        for i in range(self.R):
            # 提取每个机器人的 torques，形状是 [12]，分成 4 个 3D 力矩
            leg_torques = torques[i, :].view(4, 3)  # 重塑为 [4, 3]
            
            for leg_idx in range(4):
                # 对于每条腿的力矩计算
                # 旋转矩阵是 [3, 3], Jacobian 是 [3, 3], 力矩是 [3]
                # leg_force = leg_rotation_matrix * leg_jacobian * leg_torques
                leg_rotation_matrix = rotation_matrix[i, :, :]  # [3, 3]
                leg_jacobian = jacobians[i, :, leg_idx*3:(leg_idx+1)*3]  # [3, 3]
                
                # 计算该条腿的力矩
                f[i, leg_idx, :] = torch.matmul(leg_rotation_matrix, torch.matmul(leg_jacobian, leg_torques[leg_idx, :]))
        return f
                


    def init_urdf_file(self):
        urdf_path = f"{LEGGED_GYM_ROOT_DIR}/resources/robots/go2/urdf/go2.urdf"
        model_path = f"{LEGGED_GYM_ROOT_DIR}/resources/robots/go2/dae"
        self.urdf_model = pin.buildModelFromUrdf(urdf_path)
        self.urdf_data = self.urdf_model.createData()
        self.geom_model = pin.buildGeomFromUrdf(self.urdf_model, urdf_path, pin.GeometryType.VISUAL, model_path)

        self.leg_joints = {
            "front_left": ["FL_hip_joint", "FL_thigh_joint", "FL_calf_joint"],
            "front_right": ["FR_hip_joint", "FR_thigh_joint", "FR_calf_joint"],
            "rear_left": ["RL_hip_joint", "RL_thigh_joint", "RL_calf_joint"],
            "rear_right": ["RR_hip_joint", "RR_thigh_joint", "RR_calf_joint"]
        }


    def _compute_leg_jacobian(self, leg_name, xi):
        '''计算单腿Jacobian    针对单个rollout'''
        q = xi[-24:-12].detach().cpu().numpy()

        # 获取该腿末端 frame 的 ID
        end_effector_name = self.leg_joints[leg_name][-1].replace("_joint", "")  # 获取末端 link 名称
        frame_id = self.urdf_model.getFrameId(end_effector_name)

        # 计算雅可比矩阵
        pin.forwardKinematics(self.urdf_model, self.urdf_data, q)
        pin.updateFramePlacements(self.urdf_model, self.urdf_data)

        # 获取雅可比矩阵（LOCAL 坐标系）
        jacobian = pin.computeFrameJacobian(self.urdf_model, self.urdf_data, q, frame_id, pin.LOCAL)

        # 提取 3x3 的线速度部分
        if leg_name == 'front_left':
            jacobian_linear = jacobian[:3, :3]
        elif leg_name == 'front_right':
            jacobian_linear = jacobian[:3, 3:6]
        elif leg_name == 'rear_left':
            jacobian_linear = jacobian[:3, 6:9]
        elif leg_name == 'rear_right':
            jacobian_linear = jacobian[:3, 9:12]
        else:
            raise NotImplementedError
        return jacobian_linear


    def _compute_jacobian(self, xi):
        '''针对单个rollout'''
        jacobian = torch.zeros((3, 12), device=self.d)
        for i in range(4):
            leg_name = list(self.leg_joints.keys())[i]  # 获取字典的第i个键
            leg_jacobian = self._compute_leg_jacobian(leg_name, xi)
            jacobian[:, 3*i:3*i+3] = torch.tensor(leg_jacobian, device=self.d)
        return jacobian


    def _compute_rotation_matrix(self, x):
        '''针对所有rollout'''
        gravity_vec = torch.tensor([[0., 0., -1.]] * self.R, device=self.d)
        projected_gravity = torch.zeros((self.R, 3), device=self.d)
        projected_gravity = x[:, 6:9]
        # 归一化向量
        gravity_vec = gravity_vec / gravity_vec.norm(p=2, dim=-1, keepdim=True)
        projected_gravity = projected_gravity / projected_gravity.norm(p=2, dim=-1, keepdim=True)
        
        # 计算旋转轴（叉积）
        rotation_axis = torch.cross(gravity_vec, projected_gravity, dim=-1)
        axis_norm = rotation_axis.norm(p=2, dim=-1, keepdim=True)
        rotation_axis = rotation_axis / axis_norm  # 单位旋转轴
        
        # 计算旋转角度（点积）
        dot_product = torch.sum(gravity_vec * projected_gravity, dim=-1, keepdim=True)
        theta = torch.acos(dot_product)
        
        # 构造四元数
    
        q_w = torch.cos(theta / 2.0)
        q_xyz = rotation_axis * torch.sin(theta / 2.0)
        
        # 四元数：[w, x, y, z]
        quat = torch.cat([q_w, q_xyz], dim=-1)
        
        # 提取四元数分量
        q_w, q_x, q_y, q_z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        
        # 计算旋转矩阵
        Rotation_Matrix = torch.stack([
            1 - 2 * (q_y**2 + q_z**2), 2 * (q_x * q_y - q_z * q_w), 2 * (q_x * q_z + q_y * q_w),
            2 * (q_x * q_y + q_z * q_w), 1 - 2 * (q_x**2 + q_z**2), 2 * (q_y * q_z - q_x * q_w),
            2 * (q_x * q_z - q_y * q_w), 2 * (q_y * q_z + q_x * q_w), 1 - 2 * (q_x**2 + q_y**2)
        ], dim=-1)
        
        return Rotation_Matrix.view(-1, 3, 3)  # 输出旋转矩阵 [batch_size, 3, 3]



    def _PD_gain(self, x, du):
        '''针对多个rollouts'''
        action_scale = 0.25
        actions_scaled = du * action_scale
        p_gains = 20.
        d_gains = 0.5
        dof_pos = x[:, -24:-12]
        dof_vel = x[:, -12:]
        default_dof_pos = torch.tensor([[ 0.1000,  0.8000, -1.5000, -0.1000,  0.8000, -1.5000, 
                            0.1000,  1.0000, -1.5000, -0.1000,  1.0000, -1.5000]] * self.R, device=self.d)
        torques = p_gains*(actions_scaled + default_dof_pos - dof_pos) - d_gains*dof_vel
        return torques



    def form_controller(self):
        ''' form the controller after update process '''
        # 此处定义了控制量的约束
        if FULL_MODE:
            self.ctrl = mppi.MPPI(self.dynamics, self.running_cost, self.n, self.noise_sigma, 
                                num_samples=self.R, horizon=self.T, lambda_=self.lambda_,
                                device=self.d,
                                u_min=torch.tensor(U_MIN, dtype=torch.double, device=self.d), 
                                u_max=torch.tensor(U_MAX, dtype=torch.double, device=self.d)
            )
        else:
            self.ctrl = mppi.MPPI(self.dynamics, self.running_cost, self.n, self.noise_sigma, 
                                num_samples=self.R, horizon=self.T, lambda_=self.lambda_,
                                device=self.d, 
                                u_min=torch.tensor(dU_MIN, dtype=torch.double, device=self.d), 
                                u_max=torch.tensor(dU_MAX, dtype=torch.double, device=self.d)
            )
    

    def dynamics(self, x, perturbed_u):  # for NNDMDc
        ''' x_t+1 = dynamics(x_t, u_nominal, du) 
            perturbed_du = action in mppi
            delta du given by sampling noise
        '''
        if FULL_MODE:
            # print("### perturbed_u = ", perturbed_u)
            perturbed_u = torch.clamp(perturbed_u, U_MIN, U_MAX)
            # compute u_nominal from RL policy, requires full observation
            with torch.no_grad():
                xT = x.cpu().numpy()
                uT = perturbed_u.cpu().numpy()
                self.next_x = torch.tensor(self.kmodel.predict(x=xT, u=uT), device=self.d, dtype=self.dtype)
                ### 其他约束:
                # 1. ground force constraint

        else:
            # here, perturbed_u is actually perturbed_du
            perturbed_u = torch.clamp(perturbed_u, dU_MIN, dU_MAX)
            with torch.no_grad():
                xT = x.detach().cpu().numpy()
                perturbed_u = perturbed_u.detach().cpu().numpy()
                obs = self.obs.clone().detach()
                obs[:, :self.n] = x[:, :self.n]
                u_nominal = self.policy(self.obs).cpu().numpy()         # (1000, 48) --- > (1000, 12)
                self.next_x = torch.tensor(self.kmodel.predict(x=xT, u=u_nominal + perturbed_u)) 

        return self.next_x


    #######################
    ##### udpate info #####
    #######################


    def update_koopman_model_dmdc(self, A, B):   # DMDc
        ''' load koopman model to tensor '''
        self.A = torch.from_numpy(A).cuda().type(torch.float32)
        self.B = torch.from_numpy(B).cuda().type(torch.float32)
    
    def update_koopman_model_dmd(self, A):  # DMD
        ''' load koopman model to tensor '''
        self.A = torch.from_numpy(A).cuda().type(torch.float32)
    
    def update_koopman_model_edmd(self, model):  # EDMDc
        ''' load koopman model to tensor '''
        self.kmodel = model
    
    def update_koopman_model_nndmdc_with_trained_K(self, model):  # NNDMDc
        ''' load koopman model to tensor '''
        self.kmodel = model

    def update_u_nominal(self, u_nominal):
        ''' update the u preset or given by present low-level controller '''
        self.u_nominal = u_nominal.detach().expand(self.R, self.m)

    def update_r(self, r):
        ''' update the command '''
        self.r = r.detach().cuda()
    
    def update_foot_force(self, foot_force):
        self.foot_force = foot_force.to(self.d)
        self.contact = torch.zeros((self.foot_force.shape[0], self.foot_force.shape[1]))
        self.contact = self.foot_force[:, :, 2] > 1.    # 有实际接触力的脚

    def update_friction(self, static, dynamic):
        self.sf = static
        self.df = dynamic


    #################
    ##### utils #####
    #################

    def load_full_obs(self, obs):
        self.obs = obs.expand(self.R, 48)   #(1000,48)

    def load_policy(self):
        ''' load policy from RL model '''
        self.policy = None
        actor_critic_rl: ActorCritic = ActorCritic(
            num_actor_obs=48,
            num_critic_obs=48,
            num_actions=self.m,
            actor_hidden_dims=[512, 256, 128],
            critic_hidden_dims=[512, 256, 128],
            activation='elu',
            init_noise_std=1.0,
        )
        loaded_dict = torch.load(f'{LEGGED_GYM_ROOT_DIR}/logs/kmppi/A/model_15000.pt', map_location='cuda:0')
        actor_critic_rl.load_state_dict(loaded_dict['model_state_dict'])
        actor_critic_rl.to('cuda:0')
        actor_critic_rl.eval()
        self.policy = actor_critic_rl.act_inference
        print("### policy loaded ###")

    def remove_cache(self):
        del self.ctrl        
        # torch.cuda.empty_cache()

    ##########################
    ##### dynamics bakup #####
    ##########################

    # def dynamics(self, x, perturbed_du):  # for du, DMDc
    #     ''' x_t+1 = dynamics(x_t, u_nominal, du) 
    #         perturbed_du = action in mppi
    #         delta du given by sampling noise
    #     '''
    #     # print("### perturbed_du = ", perturbed_du)
    #     perturbed_du = torch.clamp(perturbed_du, dU_MIN, dU_MAX)

    #     # print("###")
    #     # print('counting x = ', x)
    #     # print('counting perturbed_u = ', perturbed_du)
    #     # print("shape of x = ", x.shape)
    #     # print("shape of u = ", perturbed_du.shape)
    #     # print("###")

    #     # compute u_nominal from RL policy, requires full observation
    #     with torch.no_grad():
    #         x = x.detach()
    #         perturbed_du = perturbed_du.detach()
    #         obs = self.obs.clone().detach()
    #         obs[:, :self.n] = x[:, :self.n]
    #         u_nominal = self.policy(self.obs)  # (1000, 48) --- > (1000, 12)
    #         next_x = torch.matmul(x, self.A.T) + torch.matmul(u_nominal + perturbed_du, self.B.T)
    #     return next_x

    # def dynamics(self, x, perturbed_u):  # for full u command, DMDc
    #     ''' x_t+1 = dynamics(x_t, u_nominal, du) 
    #         perturbed_du = action in mppi
    #         delta du given by sampling noise
    #     '''
    #     print("### perturbed_u = ", perturbed_u)
    #     perturbed_u = torch.clamp(perturbed_u, U_MIN, U_MAX)

    #     # print("###")
    #     # print('counting x = ', x)
    #     # print('counting perturbed_u = ', perturbed_du)
    #     # print("shape of x = ", x.shape)
    #     # print("shape of u = ", perturbed_du.shape)
    #     # print("###")

    #     # compute u_nominal from RL policy, requires full observation
    #     with torch.no_grad():
    #         x = x.detach().clone()
    #         next_x = torch.matmul(x, self.A.T) + torch.matmul(perturbed_u, self.B.T)
    #     return next_x

    # def dynamics(self, x, perturbed_u):  # for full u command, EDMDc
    #     ''' x_t+1 = dynamics(x_t, u_nominal, du) 
    #         perturbed_du = action in mppi
    #         delta du given by sampling noise
    #     '''
    #     # print("### perturbed_u = ", perturbed_u)
    #     perturbed_u = torch.clamp(perturbed_u, U_MIN, U_MAX)

    #     # compute u_nominal from RL policy, requires full observation
    #     with torch.no_grad():
    #         xT = x.cpu().numpy()
    #         uT = perturbed_u.cpu().numpy()
    #         self.next_x = torch.tensor(self.kmodel.predict(x=xT, u=uT), device=self.d, dtype=self.dtype)
    #     return self.next_x
    

### load koopman model: EDMDc:
        # EDMDc = pk.regression.EDMDc()
        # self.kmodel = pk.Koopman(regressor=EDMDc)