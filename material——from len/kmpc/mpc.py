# dir import
from rsl_rl.env import VecEnv
from rsl_rl.runners import OnPolicyRunner
from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR

# lib import
import os
import torch
import numpy as np
import pykoopman as pk
from pykoopman.common  import advance_linear_system
import cvxpy as cp

class Mpc():
    def __init__(self, n, m, nr, rank=3):
        self.n = n
        self.m = m
        self.nr = nr
        self.rank = rank
        self.C = torch.zeros((m, n+m))  # observation matrix
        self.constraints_p = torch.zeros(n)
        self.constraints_r = torch.zeros(m)


    def load_constrains(self):
        
        self.constraints_p = [
            [-1, 1],
            [-1, 1],
            [-1, 1],
            [-1, 1],
            [-1, 1],
            [-1, 1],
            [-1, 1],
            [-1, 1],
            [-1, 1]
        ]
        self.constraints_r = [
            [-1, 1],
            [-1, 1],
            [-1, 1]
        ]



    def form_problem_from_dmdc(self, A, B, p0, r0, r, T=50):
        # 这里的p实际上应该是x
        P = cp.Variable((self.rank, T+1))
        R = cp.Variable((self.m, T))
        cost = 0.0
        constraints = []

        for t in range(T):
            # cost 中使用原有的指令
            cost += cp.sum_squares(P[:2, t] - r[:2])
            constraints += [ P[:,t+1] == A @ P[:,t] + B @ R[:,t]]
            # for i in range(self.rank):
            #     constraints += [ P[i,t+1]>=self.constraints_p[i,0], P[i,t+1]<=self.constraints_p[i,1] ]
            # for i in range(self.m):
            #     constraints += [ R[i,t]>=self.constraints_r[i,0], R[i,t]<=self.constraints_r[i,1] ]
            for i in range(self.rank):
                constraints += [ P[i,t+1] >= -5., P[i,t+1] <= 5. ]
            for i in range(self.m):
                constraints += [ R[i,t] >= -5., R[i,t] <= 5. ]
            

        # constraints 中使用r_hat信息作为起始
        constraints += [P[:,0] == p0, R[:,0] == r0]
        objective = cp.Minimize(cost)
        problem = cp.Problem(objective, constraints)
        problem.solve()

        # get the predicted r
        r_hat= R.value[:, 1]
        return r_hat, P.value, R.value



    def form_problem_from_edmd(self, model, p0, r0, r, T=50):
        # 这里的p实际上不知道是啥
        P = cp.Variable((self.rank, T+1))
        R = cp.Variable((self.m, T))
        cost = 0.0
        constraints = []

        for t in range(T):
            cost += cp.sum_squares(P[:2, t] - r[:2])
            constraints += [ P[:,t+1] == model.predict]
            # constraints += [ R[] ]
            # constraints += [ P[] ]
        
        constraints += [P[:,0] == p0, R[:,0] == r]
        objective = cp.Minimize(cost)
        problem = cp.Problem(objective, constraints)
        problem.solve()

        # get the predicted r
        r_hat= R.value[:, 1]
        return r_hat, P.value, R.value



    def form_problem_from_dmdc_low(self, A, B, x0, du0, r, u, T=5):
        # 这里的p实际上应该是x
        X = cp.Variable((self.rank, T+1))   # 48完整的obs,不收敛
        dU = cp.Variable((self.m, T))
        U = cp.Variable((12, T))
        u = u.reshape((12,))
        cost = 0.0
        constraints = []
        
        for t in range(T):
            # cost 中使用原有的指令
            cost += cp.sum_squares(X[:2, t] - r[:2])    # || x - r ||
            cost += cp.sum_squares(dU[:, t])             # || du ||
            constraints += [ X[:,t+1] == A @ X[:,t] + B @ (u + dU[:, t])]

            # for i in range(self.rank):
            #     constraints += [ P[i,t+1]>=self.constraints_p[i,0], P[i,t+1]<=self.constraints_p[i,1] ]
            # for i in range(self.m):
            #     constraints += [ R[i,t]>=self.constraints_r[i,0], R[i,t]<=self.constraints_r[i,1] ]
            # for i in range(self.rank):
            #     constraints += [ X[i,t+1] >= -5., X[i,t+1] <= 5. ]
            # for i in range(self.m):
            #     constraints += [ U[i,t] >= -5., U[i,t] <= 5. ]
            

        # constraints 中使用r_hat信息作为起始
        constraints += [X[:,0] == x0, dU[:,0] == du0]
        objective = cp.Minimize(cost)
        problem = cp.Problem(objective, constraints)
        problem.solve()

        # get the modified u
        du_hat = dU.value[:, T//2]
        return du_hat, X.value, dU.value


