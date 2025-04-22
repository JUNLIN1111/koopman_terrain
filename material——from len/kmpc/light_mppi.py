# dir import
from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
from legged_gym.envs import *
# lib import
import torch
import numpy as np
# from legged_gym.kmpc.koopman import NNDMDc_Module


class LIGHT_MPPIController:

    def __init__(self, m, dynamics, cost_fn, u_min=-1, u_max=1, nr=0, lb=1., T=15, R=1000, device="cuda"):
        self.T = T
        self.R = R
        self.lambda_ = lb
        self.m = m
        self.nr = nr
        self.u_min = u_min
        self.u_max = u_max
        self.device = device
        self.dynamics = dynamics
        self.cost_fn = cost_fn

        self.control_sequence = torch.zeros((T, self.m), device=self.device)


    def sample_traj(self, state):
        # control_noise = torch.randn((self.R, self.T, self.m), device=self.device)
        control_noise = torch.empty((self.R, self.T, self.m), device=self.device).uniform_(self.u_min, self.u_max)
        # sampled_controls = self.control_sequence.unsqueeze(0) + control_noise
        sampled_controls = control_noise
        sampled_controls = torch.clamp(sampled_controls, self.u_min, self.u_max)

        sampled_states = torch.zeros((self.R, self.T+1, state.size(0)), device=self.device)
        sampled_states[:, 0] = state
        
        for t in range(self.T):
            sampled_states[:, t+1] = self.dynamics(sampled_states[:, t], sampled_controls[:, t])
        
        return sampled_states, sampled_controls
    
    def update_controls(self, state):
        sampled_states, sampled_controls = self.sample_traj(state)
        
        ### the cost for each trajectory
        costs = torch.zeros(self.R, device=self.device)
        
        for t in range(self.T):
            costs += self.cost_fn(sampled_states[:, t], sampled_controls[:, t])
        
        costs_min = costs.min()
        weights = torch.exp(-1 / self.lambda_ * (costs - costs_min))
        weights /= weights.sum()

        weighted_controls = weights.unsqueeze(1).unsqueeze(2) * sampled_controls
        self.control_sequence = weighted_controls.sum(dim=0)

    def get_action(self):
        return self.control_sequence[0]
    

    ### example of dynamics and cost_fn
    # def dynamics(self, state, control):
    #     ''' input:  state (R, n)
    #                 sampled u (R, m)
    #         output: propagated state (R, n)
    #     '''
    #     next_state = torch.zeros((self.R, state.size(0)), device=self.device)
    #     return next_state

    # def cost_fn(self, state, control):
    #     ''' input:  state (R, n)
    #                 sampled u (R, m)
    #         output: cost (R)
    #     '''
    #     step_cost = torch.zeros(self.R, device=self.device)
    #     return step_cost

