import pdb
import time
import numpy as np
import sys
import os
from pyrep import PyRep
from pyrep.robots.robot_component import RobotComponent
from pyrep.objects.dummy import Dummy
from pyrep.objects.shape import Shape

from pyrep.backend import sim,utils
from gym.spaces import Box

class Laparo_Sim_artery():
    
    def __init__(self,random_start = True, headless = False, bounded = True, time_episode=7):
        
        # intializing variables
        self.scene = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'laparoscopic_single_pure.ttt')
        
        self.pr = PyRep()
        self.pr.launch(self.scene, headless=headless)        
        # set engine ode, timestep 5ms, edited at 2021.10.23, *(2ms)
        self.timestep     = 0.005
        self.time_episode = time_episode
        sim.simSetInt32Parameter(sim.sim_intparam_dynamic_engine,sim.sim_physics_ode)
        self.pr.set_simulation_timestep(self.timestep)
        self.pr.start()
        # define dof name, handle, force
        self.dof_joint_name = ['Rotation_1_active',
                               'Rotation_3_active',
                               'Rotation_5_active_shaft',
                               'Translation_1_active']
        self.dof_max_forces = [1,1,1,40]                               
        self.dof = RobotComponent(0,"Base",self.dof_joint_name)
        self.dof.set_motor_locked_at_zero_velocity(True)
        self.initial_joint_positions = self.dof.get_joint_positions()        
        # tip and target
        self.tt_name = ["tip","target"]        
        self.tt      = [Dummy(name) for name in self.tt_name]        
        
        self.action_limit = 0.5
        self.RANDOM_START = random_start
        self.BOUNDED = bounded
        self.action_space = Box(-self.action_limit, self.action_limit, (self.act_dim,),dtype=np.float32)
        self.observation_space = Box(-2, 2, (self.ob_dim,),dtype=np.float32)
                                
    def _get_state(self):
        return np.concatenate([self.dof.get_joint_positions(),
                              np.concatenate([self.tt[i].get_position() for i in range(len(self.tt_name))]),
                              ]) 
              
    def reset(self):  
        self.pr.stop()  
        utils.script_call("random_target_pos@Base",
                          sim.sim_scripttype_customizationscript,
                          [1 if self.RANDOM_START else 0])           

        self.dof.set_joint_positions(self.initial_joint_positions)
        self.dof.set_motor_locked_at_zero_velocity(True)
        self.dof.set_joint_forces(self.dof_max_forces)        
        self.reward  = 0        
        self.done    = 0        
        self.total_time = 0
        self.pr.start()
        return self._get_state()

    def step(self, action):      
        action = np.clip(action,-self.action_limit,self.action_limit)                           
        self.dof.set_joint_target_velocities([action[0],action[1],0,action[2]])
        self.pr.step()
        self.calc_reward(action)                                      
        self.total_time += self.timestep 
        return self._get_state(), self.reward, self.done, None

    def calc_reward(self,action):
        tt_dist = np.linalg.norm(self.tt[0].get_position()-self.tt[1].get_position())
        reward_dist = - tt_dist        
        self.reward = np.round(reward_dist,2)
             
        if np.isnan(self.reward) or reward_dist < -10\
            or np.isnan(np.sum(action)) \
                or np.isnan(np.sum(self._get_state())):
            print(f"************************************************"
                    f"\t reward_dist:{reward_dist:.2f}"
                    f"\treward_sum:{self.reward}")           
            self.reward = -100
            self.done = 1
            
        if tt_dist <= 0.01:
            self.reward = 10
            self.done = 1
                                    
        if self.BOUNDED and self.total_time >= self.time_episode:
            self.done = 1             

    @property
    def ob_dim(self):
        # dof pos : 4
        # target + tip pos: 6
        return 10
    
    @property
    def act_dim(self):
        # action for 3 controlled dofs
        return 3
    
    def shutdown(self):
        self.pr.stop()
        self.pr.shutdown()
        
if __name__ == "__main__":
    #benchmark reward
    
    env = Laparo_Sim_artery(20000)
    
    r_t = 0
    for i in range(20):
        ob,r_sum = env.reset() ,0
        while 1:
            act = (np.random.rand(3)*2-1).tolist()
            #print(act)
            ob_, reward, done,_ = env.step(act)
            r_sum += reward
            
            if done:
                break
        r_t += r_sum
        print(f"ep {i},reward:{r_sum}")
    
    print(f"Total reward:{r_t/20}")    
        
    
