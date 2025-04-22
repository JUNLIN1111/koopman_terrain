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

HARM_POLICY = False

# hard constraints
dU_MAX = 1.0
dU_MIN = -dU_MAX


class MPPIController():
    def __init__(self, n, m, nr, cfg, T=10, R=3000):
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
        iden = 0.5 * torch.ones(self.m)   # for sampling delta_u for u
        # self.noise_sigma = torch.diag(torch.tensor(iden, dtype=self.dtype, device=self.d))  # noise set
        self.noise_sigma = torch.diag(iden.clone().detach().to(device=self.d, dtype=self.dtype))

        self.lambda_ = 10.   # λ in mppi

        self.sf = self.cfg.terrain.static_friction
        self.df = self.cfg.terrain.dynamic_friction

        self.u_nominal = torch.zeros((self.R, self.m),device=self.d)
        self.next_x = torch.zeros((self.R, self.n), device=self.d)
        self.last_action = torch.zeros((self.R, self.m), device=self.d)
        self.last_generated_u = torch.zeros((self.R, self.m), device=self.d)


        ### load policy & urdf
        self.load_policy()

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
       
    

    ########################
    ##### compute cost #####
    ########################

    def running_cost(self, x, du): 
        cost = torch.zeros(self.R, device=self.d)
        lin_vel = x[:, :2].to(self.d)
        lin_vel_command = self.obs[:, 9:11]

        zero_vel_command = torch.zeros_like(lin_vel_command, device=self.d)
        one_vel_command = torch.ones_like(lin_vel_command, device=self.d)

        projected_gravity = x[:, 6:9].to(self.d)
        
        u_cost = torch.sum(torch.square(du), dim=-1)
        delta_u_cost = torch.sum(torch.square(du - self.last_generated_u), dim=-1)
        self.last_generated_u = du

        ori_cost = torch.sum(torch.square(projected_gravity[:]), dim=-1)
        ref_cost = torch.sum(torch.square(lin_vel - lin_vel_command), dim=-1)
        # vx_cost = torch.square(lin_vel[:, 0] - lin_vel_command[:, 0])
        # vy_cost = torch.square(lin_vel[:, 1] - lin_vel_command[:, 1])
        vx_cost = torch.square(lin_vel[:, 0] - one_vel_command[:, 0])
        vy_cost = torch.square(lin_vel[:, 1] - zero_vel_command[:, 1])
        
        

        # cost = 100 * vx_cost + 0.5 * u_cost + ori_cost + 0.5 * delta_u_cost
        # cost = 1000 * vy_cost + 0.5 * u_cost + ori_cost + 0.5 * delta_u_cost
        # cost = ref_cost + 0.5 * u_cost
        # print(f"### u_cost = {u_cost[0]} ### delta_u_cost = {delta_u_cost[0]} ### vx_cost = {vx_cost[0]} ### ori_cost = {ori_cost[0]} ###")
        return cost



    def form_controller(self):
        ''' 补偿控制器
            form the controller after update process
        '''
        
        self.ctrl = mppi.MPPI(self.dynamics, self.running_cost, self.n, self.noise_sigma, 
                            num_samples=self.R, horizon=self.T, lambda_=self.lambda_,
                            device=self.d, 
                            u_min=torch.tensor(dU_MIN, dtype=torch.double, device=self.d), 
                            u_max=torch.tensor(dU_MAX, dtype=torch.double, device=self.d)
        )
    

    def dynamics(self, x, perturbed_u):  # for NNDMDc
        ''' 补偿控制器
            x_t+1 = dynamics(x_t, u_nominal, du) 
            perturbed_du = action in mppi
            delta du given by sampling noise
        '''
        # here, perturbed_u is actually perturbed_du
        perturbed_u = torch.clamp(perturbed_u, dU_MIN, dU_MAX)
        # print("### perturbed_u = ", perturbed_u[:, 0])
        with torch.no_grad():
            xT = x.detach().cpu().numpy()
            perturbed_u = perturbed_u.detach().cpu().numpy()

            obs = torch.zeros((self.R, 48), device=self.d)
            obs[:, :9] = x[:, :9]
            obs[:, 9:12] = self.obs[:, 9:12]
            obs[:, 12:36] = x[:, -24:]
            obs[:, -12:] = self.last_action[:, :]
            
            # u_nominal = self.policy(self.obs).cpu().numpy()         # (1000, 48) --- > (1000, 12)
            u_nominal = self.policy(obs).cpu().numpy()
            ### harm the RL policy on purpose to test MPPI
            if HARM_POLICY:
                # u_nominal[:, [2, 5, 8, 11]] += 0.5
                u_nominal[:, [2, 5]] -= 0.4

            self.next_x = torch.tensor(self.kmodel.predict(x=xT, u=u_nominal + perturbed_u))
            self.last_action = torch.tensor(u_nominal + perturbed_u)

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
        
    def update_foot_force(self, foot_force):
        self.foot_force = foot_force.to(self.d)
        self.contact = torch.zeros((self.foot_force.shape[0], self.foot_force.shape[1]))
        self.contact = self.foot_force[:, :, 2] > 1.    # 有实际接触力的脚

    def update_friction(self, static, dynamic):
        self.sf = static
        self.df = dynamic       

    def update_full_obs(self, obs):
        self.obs = obs.clone().detach().expand(self.R, 48)   #(1000,48)
        self.last_action[:, :] = self.obs[:, -self.m:]

    #################
    ##### utils #####
    #################

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
        actor_critic_rl.to(self.d)
        actor_critic_rl.eval()
        self.policy = actor_critic_rl.act_inference
        print("### policy loaded ###")

    def remove_cache(self):
        del self.ctrl        
        # torch.cuda.empty_cache()

