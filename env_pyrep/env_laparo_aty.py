import pdb
import time
import numpy as np
import sys
import os
from pyrep import PyRep
from pyrep.objects.shape import Shape
from pyrep.robots.robot_component import RobotComponent
from pyrep.objects.dummy import Dummy

from pyrep.backend import sim,utils
from gym.spaces import Box



from .utils import *
from .LiverCollision import LiverCollision
from .SurgMesh import SurgMesh

class Laparo_Sim_artery():    
    def __init__(self,random_start = True, headless = False, bounded = True, 
                      time_episode=3, timestep = 0.002, 
                      mesh_filename='liver_iras_bottom_meter.json',
                      E=1e4, v=0.4, gamma=-4, dt=1e-4,lamu_times=1e-2):
        
        # intializing variables
        self.scene = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'laparoscopic_single_toolmesh.ttt')
        
        self.pr = PyRep()
        self.pr.launch(self.scene, headless=headless)        
        # set engine ode, timestep 2ms
        self.timestep     = timestep
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
        # attached surgical tool mesh
        self.sg = SurgMesh()
        self.sg_shape = Shape('SurgicalTool') # pre imported surgical tool mesh
        # liver
        self.liver = LiverCollision(mesh_filename=mesh_filename,E=E, v=v, gamma=gamma,
                                    dt=dt,lamu_times=lamu_times)
        self.liver_vertices = self.liver.vertices.copy()  
        self.target_attach_idx = 169 # surface vertice the tool should reach      
        self.m_to_cm = 1e2

        self.action_limit = 0.5
        self.RANDOM_START = random_start
        self.BOUNDED = bounded
        self.action_space = Box(-self.action_limit, self.action_limit, (self.act_dim,),dtype=np.float32)
        self.observation_space = Box(-2, 2, (self.ob_dim,),dtype=np.float32)    

    def reset(self):  
        self.pr.stop()  
        # random start for target point
        self.random_target_pos()  
        self._init_liver_show()    
        # reset dof
        self.dof.set_joint_positions(self.initial_joint_positions)
        self.dof.set_motor_locked_at_zero_velocity(True)
        self.dof.set_joint_forces(self.dof_max_forces)  
        # reset surgical tools
        self._get_sg_vertices()
        self._update_sg_x()        
        self.sg.xp = self.sg.x.copy()
        self.col_flag = False
        # reset env 
        self.reward  = 0
        self.done    = 0
        self.total_time = 0
        self.T_ttdist = 0.01
        self.pr.start()
        return self._get_state()

    def random_target_pos(self):
        R = 266e-3 #RMC to upper workspace boundary
        init_r = 40e-3 #266+init_r
        dr = 70e-3-init_r       #range 0-(70-init_r)
        init_theta = 5*np.pi/180
        dtheta = 15*np.pi/180 -init_theta #range [-15,-init_theta] + [init_theta,15]
        tip_pos = self.tt[0].get_position()

        if self.RANDOM_START:
            r = np.random.rand()*dr + R + init_r
            theta_x = (np.random.rand()*dtheta + init_theta)*np.random.choice([-1,1])
            theta_y = (np.random.rand()*dtheta + init_theta)*np.random.choice([-1,1])
        else:
            r = 0*dr + R + init_r
            theta_x = 0*dtheta + init_theta
            theta_y = 0*dtheta + init_theta
        dx = r*np.sin(theta_y)
        dy = r*np.cos(theta_y)*np.sin(theta_x)
        dz = -r*np.cos(theta_y)*np.cos(theta_x) + R
        self.target_pos = tip_pos + np.array([dx,dy,dz])
        self.tt[1].set_position(self.target_pos)
        # rotate the liver and face the tool
        self.dangle_z = np.arctan2(self.target_pos[1],self.target_pos[0]) *180/np.pi
        #self.tt[1].set_orientation([0,0,self.dangle_z],reset_dynamics=False) # this no need, if parent of liver is world
        
    def _init_liver_show(self):
        self.liver.vertices = self.liver_vertices.copy()
        liver_dpos = np.array([0.12,0.08,0.08])
        dx_target = self.liver.vertices/self.m_to_cm - liver_dpos   # center to target #cm to m
        dx_target = (rotation_matrix(self.dangle_z + 90,axis='z') @ dx_target.T).T
        self.liver.vertices = (dx_target + self.target_pos)*self.m_to_cm # m to cm
        self.liver.reset_x() # reset crash_flag   
        self.liver.update_AABB()
        self._update_liver()
        #self.liver_show.set_parent(self.tt[1])        
        #self.liver_show.set_position(self.liver_dpos,relative_to=self.tt[1])

    def _get_sg_vertices(self):
        # vertices, indices, normals
        self.sg.vertices, self.sg.tri_elements,_ = self.sg_shape.get_mesh_data()
        self.sg._get_ns()
        self.sg.surf_vindex = np.unique(self.sg.tri_elements)
    
    def step(self, action):   
        # steps
        #   1. action move dof
        #   2. sg_xp, sg_x update
        #   3. liver collision handle, move vertices to target place
        #   4. update liver new place
        #   5. calculate reward
        #   6. send back ob_n, reward, done,
        action = np.clip(action,-self.action_limit,self.action_limit)                           
        self.dof.set_joint_target_velocities([action[0],action[1],0,action[2]])
        self.pr.step()
        self.sg.xp = self.sg.x.copy()
        self._update_sg_x()
        self.liver.update_AABB()
        #self._collision_handle()
        self._update_liver()
        self.calc_reward(action)                                      
        self.total_time += self.timestep         
        return self._get_state(), self.reward, self.done, None

    def _update_sg_x(self):
        # H_matrix * [vertices,1].T        
        self.sg.x = np.matmul(self.sg_shape.get_matrix(),
                        np.c_[self.sg.vertices,np.ones(self.sg.vertices.shape[0])].T).T[:,:3]*self.m_to_cm # m to cm
    
    def _collision_handle(self):
        # pyrep step 5e-3, liver step 1e-4
        # 1. find possible col_pair
        # 2. find move_tri and move_t
        # 3. update liver per step to the target place (move_t)

        # 1. find possible col_pair
        col_points=self.liver.check_tet_aabb_collision(self.sg.x)
        # 2. find move_tri and move_t
        move_vindex, tg_disp = [],0
        if len(col_points) != 0: # find collision pair
            move_v_disp_dict = self.liver.collision_response_ray(col_points,self.sg)
            #move_v_disp_dict = self.liver.collision_response_cotin(col_points,self.sg_xp,self.sg_x)
            if move_v_disp_dict !={}:
                move_vindex = np.array(list(move_v_disp_dict.keys()))
                tg_disp =  np.array(list(move_v_disp_dict.values()))
                self.col_flag = True
        
        if self.col_flag: # liver volumne change starts when collision happens
            steps = int(self.timestep/self.liver.dt) 
            # move with tg_disp*1.5 to ensure no collision
            for _ in range(steps):
                self.liver.x[move_vindex] += tg_disp/steps
                self.liver._step(move_vindex=move_vindex)
                if self.liver.crash_flag:
                    return

    def _update_liver(self):
        if Shape.exists('liver'): self.liver_show.remove()
        self.liver_show = Shape.create_mesh((self.liver.x[self.liver.surf_vindex]/self.m_to_cm).flatten().tolist(),self.liver.tri_elements.flatten().tolist())
        self.liver_show.set_name('liver')
        self.liver_show.set_color([0.9,0.1,0.1])
        self.liver_show.set_transparency(0.5)
        self.tt[1].set_position(self.liver.x[self.target_attach_idx]/self.m_to_cm) #cm to m        
                        
    def _get_state(self):
        return np.concatenate([self.dof.get_joint_positions(),
                              np.concatenate([self.tt[i].get_position() for i in range(len(self.tt_name))]),
                              self.liver.aabb
                              ])

    def calc_reward(self,action):        
        # t1:tau1, tau2, tau3 = 5e-3, 1, 5e-3 too small
        # t2:tau1, tau2, tau3 = np.array([5e-3, 0.1, 5e-3])*1e1
        tau1, tau2, tau3 = np.array([5e-3, 1, 5e-3])*1e0
        self.tt_dist = np.linalg.norm(self.tt[0].get_position()-self.tt[1].get_position())
        liver_x_disp = np.linalg.norm(self.liver.x-self.liver.vertices,axis=-1).mean() #cm        
        tem1 = - tau1 
        # t2:tem2 = - np.power(self.tt_dist,1/3)*tau2 
        tem2 = - self.tt_dist * tau2 
        tem3 = - liver_x_disp * tau3        
        self.reward = np.round(tem1 + tem2 + tem3,2)

        reward_string = f"*** {tem1:.4f} / {tem1/self.reward:.4f} " \
                        f"*** {tem2:.4f} / {tem2/self.reward:.4f} " \
                        f"*** {tem3:.4f} / {tem3/self.reward:.4f} " \
                        f"*** {self.reward:.4f}"
        # error handle     
        if np.isnan(self.reward) \
            or np.isnan(np.sum(action)) \
                or np.isnan(np.sum(self._get_state())):
            #print(f"************************************************"
            #        f"\t reward_dist:{reward_dist:.2f}"
            #        f"\treward_sum:{self.reward}")           
            print(f"{reward_string} ***** error")                    
            self.reward = -100
            self.done = 1
            return 

        # liver crash, can't avoid at first exploring
        if self.liver.crash_flag:
            #self.reward += -0.1
            self.done = 1
            print(f"{reward_string} ***** crash")

        # reach target            
        if self.tt_dist <= self.T_ttdist:
            self.reward = 10
            self.done = 1
            print(f"{reward_string} ***** reach")
        
        # reach limited timesteps
        if self.BOUNDED and self.total_time >= self.time_episode:
            self.done = 1    

    @property
    def ob_dim(self):
        # dof pos : 4
        # target + tip pos: 6
        # liver aabb: 6 
        return 16
    
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
        
    
