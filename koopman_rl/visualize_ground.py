###############################################
#                                             #
#                    Will                     #
#               Date: 2025/4/14               #
#                 VIsualize policy            #
#                                             #
###############################################


# Initialize environment to get obs/act space sizes

from go2_sim import Go2Sim    
import time 

env = Go2Sim(render_mode="human") # Create Env
env.reset()

obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]

step = 0
while  step <= 10000:
    env.render()
    action = env.action_space.sample()
    env.step(action)
    step += 1
    time.sleep(0.01)
