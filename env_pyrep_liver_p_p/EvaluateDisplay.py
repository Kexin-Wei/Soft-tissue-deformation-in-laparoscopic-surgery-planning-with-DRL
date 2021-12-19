import numpy as np
from pyrep.backend import sim
from pyrep.objects.shape import Shape
from .utils import *
from .Mesh import Mesh
from .env_laparo_aty import Laparo_Sim_artery

class Trajectory():
    def __init__(self, listofps = None):
        self.points = []
        if listofps is not None:
            self.append(listofps)
        self.handle = sim.simAddDrawingObject(sim.sim_drawing_lines,4,0,-1,999,ambient_diffuse=[0.2, 1, 0.2])

    def reset(self):
        self.points = []
        sim.simAddDrawingObjectItem(self.handle,None)

    def append(self, listofps):        
        if not isinstance(listofps, list): # is np.array 
            listofps = np.array(listofps).tolist()
        for i in range(len(listofps)):
            self._append(listofps[i])

    def _append(self,ps):
        # ps 1x3 np.array or list with len 3
        self.points.append(list(ps))
    
    def disp(self):
        if len(self.points) !=0:
            for i in range(len(self.points)-1):
                sim.simAddDrawingObjectItem(self.handle,self.points[i]+self.points[i+1])

class Arrow(Mesh):
    def __init__(self,mesh_filename = 'arrow.msh.json'):
        Mesh.__init__(self)
        self.read_mesh_json(mesh_filename=mesh_filename)
        self.tet_elements = None
        self._handle_tri_elements(self.vertices)
        self.reset_x()    
        #self.top_vindex = np.argwhere(self.vertices[:,-1].max()==self.vertices[:,-1]).flatten()
        #self.direction = self.vtx_normal_vec[self.top_vindex]
        self.bash_vindex = np.argwhere(self.vertices[:,-1].min()==self.vertices[:,-1]).flatten()
        #self.fixed_vindex = np.r_[self.top_vindex,self.bash_vindex]
    
    def reset_x(self):
        self.x = self.vertices.copy()

    @staticmethod
    def skewsymmetric_matrix(vec):
        v1 = vec[0]
        v2 = vec[1]
        v3 = vec[2]
        return np.array([
              0, -v3,  v2, 
             v3,   0, -v1,
            -v2,  v1,   0
        ]).reshape(-1,3)

    def show(self):
        if Shape.exists('arrow'): self.handle.remove()
        self.handle = Shape.create_mesh((self.x*1e-2).flatten().tolist(),self.tri_elements.flatten().tolist())
        self.handle.set_name('arrow')
        self.handle.set_color([0.9,0.7,0.1])

    def update_x(self,pos,act):
        # act is the direction of the arrow
        # pos is the base of the arrow
        vec = np.cross([0,0,1],act) # rotate axis
        vec = vec/np.linalg.norm(vec)
        theta = np.arccos(np.dot([0,0,1],act))
        skw_vec = self.skewsymmetric_matrix(vec)
        rotate_matrix = np.identity(3) + skw_vec * np.sin(theta) + (1-np.cos(theta)) * skw_vec@ skw_vec
        temp = self.vertices- self.vertices[self.bash_vindex].mean(axis=0)
        temp = (rotate_matrix@temp.T).T
        self.x = temp + np.array(pos)*1e2
        
# run one round then display the trajectory and each acts for human assistance
def keyboard_input():
    delta = 0.5
    act_insert = [    0,     0, -delta]
    act_row    = [delta,     0,      0]
    act_pitch  = [    0, delta,      0]
    act_map = np.c_[act_row,act_insert,act_pitch].T
    act_map = np.r_[act_map,-act_map]
    act_code = input("Please select move direction:"
                     "\t 1. +y"
                     "\t 2. -z"
                     "\t 3. -x"
                     "\t 4. -y"
                     "\t 5. +z"
                     "\t 6. +x")    

    try:
        act_num = int(act_code)
        if act_num<=6 and act_num >=1:
            act = act_map[act_num-1]
        else:
            act = [0,0,0]
    except:
        act = [0,0,0]    
    return np.array(act)

def human_evaluate(ac,env,epochs = 20):
    import torch
    ac_limit = env.action_space.high[0]
    act_dim = env.action_space.shape[0]
    
    print("DRL planning...")
    
    reward_list = np.zeros(epochs)
    traj_list = []
    reach_target_ep_flag = False
    for ep in range(epochs):
        ob, reward_sum, traj,acts = env.reset(), 0, [], []

        while 1:
            act = ac.act(torch.as_tensor(ob, dtype=torch.float32))            
            ob, reward, done,_  = env.step(act)
            reward_sum += reward
            traj.append(env.tt[0].get_position().tolist())
            acts.append(act.tolist())
            if done:
                break

        print(f"Ep: {ep},\treward:{reward_sum:.3f}")    
        reward_list[ep]=reward_sum
        traj_list.append(traj)  

        if reward == 10: # reach target
            reach_target_ep_flag = True
            break # use current trajectory to guide
    
    if not reach_target_ep_flag:
        # ep is the reward_biggest one
        ep = reward_list.argmax()
    
    print(f"Planning finished...\nStart guiding...")
    # human guidance, traj[ep], acts[ep]
    ob = env.reset()
    eval_traj = Trajectory(traj_list[ep])
    eval_traj.disp()
    arrow = Arrow()
    while 1:
        # suggest
        base = env.tt[0].get_position()
        suggest = ac.act(torch.as_tensor(ob, dtype=torch.float32))
        temp = np.sign(suggest)
        direction = temp.copy()
        direction[0] = -temp[1] # pitch  -> x
        direction[1] =  temp[0] # row    -> y
        direction[2] =  temp[2] # insert -> z
        direction = direction / np.linalg.norm(direction)
        print(f"Suggest Act: {suggest}")
        arrow.update_x(base,direction)
        arrow.show()

        # select    
        act = keyboard_input()                        
        ob,reward, done, _ = env.step(act)
        if done:
            break
    print("Guiding done")
    env.shutdown()

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    arrow = Arrow()
    ax=  arrow.plt_vtx(text_opt='on')
    plt_show_equal(ax)
