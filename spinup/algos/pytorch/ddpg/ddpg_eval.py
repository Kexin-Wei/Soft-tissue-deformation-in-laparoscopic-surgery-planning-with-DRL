import spinup.algos.pytorch.ddpg.core as core
import os
import torch
import numpy as np
import sys
sys.path.append('.')
print(sys.path)
from env_pyrep.env_laparo_aty import Laparo_Sim_artery
from env_pyrep.EvaluateDisplay import human_evaluate

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--hid', type=int, default=256)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--num_test_episodes', type=int, default=10)
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--random_start', action='store_false')
    args = parser.parse_args()

    ac_kwargs=dict(hidden_sizes=[args.hid]*args.l)
    actor_critic=core.MLPActorCritic
    env= Laparo_Sim_artery(random_start=True)

    ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)    
    intel_path = '2021-12-11_ddpg/2021-12-11_15-18-42-ddpg_s0'

    ac = torch.load(os.path.join("/home/test/spinnup/output",intel_path,"pyt_save/model.pt"))

    human_evaluate(ac,env,epochs=1)
