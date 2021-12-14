import numpy as np
import os
import matplotlib.pyplot as plt
from .utils import *
from .Mesh import Mesh

class SurgMesh(Mesh):
    def __init__(self, mesh_filename='surgical_tool_tri.msh.json'):
        Mesh.__init__(self)
        mesh_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),mesh_filename)
        self.read_mesh_json(mesh_file)       
        
        self.tet_elements = None 
        self._handle_tri_elements(self.vertices)
        self.vertices = np.c_[-self.vertices[:,2],self.vertices[:,1],self.vertices[:,0]]
        self.vertices *= 1e-1
        
        self.x = self.vertices.copy()
        self.p0 = self.vertices.mean(axis=0)
        
        self.z = 0 # insert
        self.xangle = 0 # row
        self.yangle = 0 # pitch
        

    def plt_sg_vtx(self,draw_type='tri', text_opt='off', scl=1, **kwargs):
        ax = self._plt_msh(self.vertices*scl, draw_type=draw_type,node_opt='off', 
                            text_opt=text_opt, **kwargs)
        limits = np.c_[self.vertices.min(axis=0), self.vertices.max(axis=0)]
        plt_equal(ax,limits = limits)                              
        return ax

    def plt_sg_x(self,draw_type='tri', text_opt='off', scl=1, **kwargs):
        ax = self._plt_msh(self.x*scl, draw_type=draw_type,node_opt='off', 
                            text_opt=text_opt, **kwargs)
        limits = np.c_[self.x.min(axis=0), self.x.max(axis=0)]
        plt_equal(ax,limits = limits)                            
        return ax
    
    def set_dof(self,z,xangle,yangle):
        z = np.clip(z,0,8)
        xangle = np.clip(xangle,-30,30)
        yangle = np.clip(yangle,-30,30)

        m1 = rotation_matrix(xangle,axis='x')
        m2 = rotation_matrix(yangle,axis='y')

        pp0 = self.vertices-self.p0
        pp0[:,-1] += -z
        pp0 = (m1@m2@pp0.T).T
        self.x = pp0 + self.p0
        self.update_normal_vecs()


if __name__ == '__main__':
    sg = SurgMesh()
    # ax = sg.plt_vtx(text_opt='on')
    # plt_equal(ax)    

    z = np.random.rand()*8-4
    xangle = np.random.rand()*60-30
    yangle = np.random.rand()*60-30
    
    sg.set_dof(z,xangle,yangle)

    ax = sg.plt_sg_vtx()
    ax = sg.plt_sg_x(ax=ax)
    ax.scatter(sg.p0[0],sg.p0[1],sg.p0[2]) 
    plt.show()
    print('done')