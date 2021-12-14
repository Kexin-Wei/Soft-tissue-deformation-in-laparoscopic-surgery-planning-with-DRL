import numpy as np
import matplotlib.pyplot as plt
from .utils import *
from .MTM import MTM


class LiverDeform(MTM):
    def __init__(self, mesh_filename='liver_matdata_meter.json',E=1e4, v=0.4, gamma=-5, timeinterval=0.1,dt=1e-4,lamu_times=1e-1):
        MTM.__init__(self,E=E,v=v,gamma=gamma,timeinterval=timeinterval,dt=dt)
        self.read_mesh_json(mesh_filename)
        self.vertices *= 1e2 # cm
        # 1e-2 too soft, change to 1e-1
        self.lamu_times = lamu_times
        self.la *= self.lamu_times # kg/cm/s^2 # Pa = N/m^2 = m/s^2 * kg /m^2 = kg/m/s^2
        self.mu *= self.lamu_times # kg/cm/s^2 # Pa = N/m^2 = m/s^2 * kg /m^2 = kg/m/s^2
        self._handle_tri_elements(self.vertices)
        self.reset_x()
        #self.surf_vindex = np.unique(self.tri_elements)        
        #self.fixed_vindex = np.arange(self.n_v)[np.isin(np.arange(self.n_v),self.surf_vindex,invert=True)]
        #self.Fos = np.array([0, 0, -1]) * self.dm * 9.8 # gravity in mm

    def step(self):
        for _ in range(int(self.T_interval / self.dt)):
            self._step()

    def _step(self,move_vindex =[]): # replace old with Ficp
        self._update_volume()
        if self.crash_flag: 
            return
        self._update_aj_set()
        self._update_BCD_set()
        self._update_Fes()
        self._update_Fincompressible()
        self.Fis = -self.Fes + self.Ficp
               
        self._pre_fixed_handle(move_vindex) # can replace
        self._explicit_step()       
        self._post_fixed_handle(move_vindex) # can replace
        self.t += self.dt

    def _pre_fixed_handle(self,move_vindex=[]): # replace old
        self.Fis[move_vindex] = 0
        self.Fis[self.fixed_vindex] = 0
    
    def _post_fixed_handle(self,move_vindex=[]): # replace old
        self.x[move_vindex] = self.xp[move_vindex]
        self.x[self.fixed_vindex] = self.xp[self.fixed_vindex]

if __name__ == "__main__":
    liver = LiverDeform()
    ax = liver.plt_x(text_opt='off')
    ax.text(liver.vertices[190,0],liver.vertices[190,1],liver.vertices[190,2],'190')
    find = np.argwhere(liver.tet_elements==190)[:,0]
    ax = plt_tet(liver.vertices[liver.tet_elements[find[0]]],ax=ax)
    plt_show_equal(ax,block=False)
    print("done")