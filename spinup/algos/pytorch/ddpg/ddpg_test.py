import spinup.algos.pytorch.ddpg.core as core
import sys
sys.path.append('/home/test/spinnup/')
print(sys.path)
from env_pyrep.env_laparo_aty import Laparo_Sim_artery
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter

# class Test:
#     def __init__(self, args, env):
#         self.num_test_episodes=args.num_test_episodes
#         self.env=env
#         self.max_ep_len=1000
#         self.ac_limit = self.env.action_space.high[0]
#         self.act_dim = env.action_space.shape[0]

#     def get_action(self, o, noise_scale):
#         a = ac.act(torch.as_tensor(o, dtype=torch.float32))
#         a += noise_scale * np.random.randn(self.act_dim)
#         return np.clip(a, -self.ac_limit, self.ac_limit)

#     def test_agent(self):
#         for j in range(self.num_test_episodes):
#             o, d, ep_ret, ep_len = env.reset(), False, 0, 0
#             while not(d or (ep_len == self.max_ep_len)):
#                 # Take deterministic actions at test time (noise_scale=0)
#                 o, r, d, _ = env.step(self.get_action(o, 0))
#                 ep_ret += r
#                 ep_len += 1
#             logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)

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
    env = Laparo_Sim_artery(bounded=True,headless=args.headless,random_start=args.random_start)

    ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)    
    
    ac = (torch.load("/home/test/spinnup/output/2021-12-10_ddpg/2021-12-10_14-19-14-ddpg_s0/pyt_save/model.pt"))
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