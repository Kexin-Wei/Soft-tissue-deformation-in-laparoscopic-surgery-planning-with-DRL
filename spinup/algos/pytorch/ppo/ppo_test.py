import spinup.algos.pytorch.ppo.core as core

import sys
sys.path.append('.')
print(sys.path)
from env_pyrep.env_laparo_aty import Laparo_Sim_artery
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--hid', type=int, default=64)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--num_test_episodes', type=int, default=10)
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--random_start', action='store_false')
    args = parser.parse_args()

    ac_kwargs=dict(hidden_sizes=[args.hid]*args.l)
    actor_critic=core.MLPActorCritic
    env = Laparo_Sim_artery(bounded=True,headless=args.headless,random_start=args.random_start)

    ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)

    ac = (torch.load("/home/test/spinningup/data/ppo/ppo_s0/pyt_save/model.pt"))
    ac_limit = env.action_space.high[0]
    act_dim = env.action_space.shape[0]

    print("start")
    reward_list = []
    writer = SummaryWriter(log_dir='test')
    for i in range(10):
        ob, reward_sum = env.reset(), 0
        while 1:
            act = ac.act(torch.as_tensor(ob, dtype=torch.float32))
            act = np.clip(act, -ac_limit, ac_limit)
            ob, reward, done,_  = env.step(act)
            reward_sum += reward
            if done:
                break
        print(f"Ep: {i},\treward:{reward_sum}")    
        reward_list.append(reward_sum)
        writer.add_scalar('Reward/test', reward_sum, i)
    
    print(f"Average reward: {sum(reward_list)/10}")