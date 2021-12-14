import pdb
import numpy as np
from .Mesh import Mesh
from .utils import *

class MTM(Mesh):
    def __init__(self,E=2500,v=0.4,gamma=-0.01,timeinterval=0.1,dt=1e-4):
        Mesh.__init__(self)        

        self.la, self.mu = self.lame_param(E,v) # meter  
        self.j_index = np.array([1, 2, 3, 2, 3, 0, 3, 0, 1, 0, 1, 2])
        self.j2_index = np.c_[self.j_index.reshape(4, 3)[:, 1:3],
                              self.j_index.reshape(4, 3)[:, 0]].flatten()
        self.jng = np.array([1, -1, 1, -1])

        self.t = 0
        self.dt = dt #5e-5
        self.T_interval = timeinterval
        self.gamma = gamma

        self.dm = 0.005615 # kg
        self.Fes = None
        self.Fis = None
        self.Fos = 0
        
        self.tet_index = None
        self.top_vtx_index = None
        self.init_volume6 = None
    
    @staticmethod
    def lame_param(E, v):
        la = E * v / (1 + v) / (1 - 2 * v)
        mu = E / 2 / (1 + v)
        return la, mu

    def reset_x(self):
        self.t=0
        self.x = self.vertices.copy()   # position at t
        self.xp = self.x.copy()         # position at t-dt
        self.sp = np.zeros(self.x.shape)  # speed at t-dt
        self.crash_flag = False
        self._update_volume()

    '''
    paper: https://hal.archives-ouvertes.fr/inria-00072611
    changing variables at each simulation step
        x: vertices position
        xp: vertices last position
        sp: vertices speed
        Fo: vertices outer force
        Ficp: vertices against compression
    each simulation steps:
        1. self._update_volumnes()
        2. self._update_aj_set()            # need volumnes
        3. self._update_BCD_set()           # need aj_set
        4. self._update_Fes()               # need BCD_set, x, xp
        5. self.Fis = -self.Fes + self.Fos
        6. fixed point handle
            self.Fis[fixed] = 0
        7. self._explicit_step()
        8. fixed point handle
            8.1 self.x[fixed]  = self.xp[fixed]
            8.2 self.sp[fixed] = 0 
    '''
    
    def _step(self): 
        self._update_volume()
        self._update_aj_set()
        self._update_BCD_set()
        self._update_Fes()
        #self._update_Fincompressible()
        self.Fis = -self.Fes + self.Fos
               
        self._pre_fixed_handle() # can replace
        self._explicit_step()       
        self._post_fixed_handle() # can replace
        self.t += self.dt
    
    def _update_volume(self):
        self.volumes6 = np.linalg.det(np.c_[np.ones(self.n_tet * 4),
                                                self.x[self.tet_elements.flatten()]].reshape(-1, 4, 4))
        if self.init_volume6 is None:
            self.init_volume6 = np.linalg.det(np.c_[np.ones(self.n_tet * 4),
                                                self.vertices[self.tet_elements.flatten()]].reshape(-1, 4, 4))                                                
        if (self.volumes6 < 0).any():
            #pdb.set_trace()
            self.crash_flag = True
            #raise ValueError("Tetrahedron is crashed")
    
    def _update_aj_set(self):
        vj = self.x[self.tet_elements[:, self.j_index]]
        vj2 = self.x[self.tet_elements[:, self.j2_index]]        
        self.aj_set = np.cross(vj, vj2).reshape(self.n_tet, -1, 3, 3).sum(axis=2) * \
                      self.jng[None, :, None]  # (n_tet,4,3)
        v6dvd = 1.0 / self.volumes6 # n_tet x 1
        self.aj_set = self.aj_set * v6dvd[:,None,None] # (n_tet,4,3)
    
    def _update_BCD_set(self):
        # Bjk = la/2 *(aj tensorx ak) mu/2*[ak tensorx aj + aj.dot(ak)Id3]
        aj_t_k = np.einsum('ijl,ikn->ijkln',self.aj_set,self.aj_set) # aj tensorx ak (n_tet x4 x4 x3 x3) - (n_tet xj xk x3 x3)
        ak_t_j = np.einsum('ijl,ikn->ijknl',self.aj_set,self.aj_set) # ak tensorx aj (n_tet x4 x4 x3 x3) - (n_tet xj xk x3 x3)
        aj_d_k = np.einsum('ijn,ikn->ijk',self.aj_set,self.aj_set) # aj.dot ak (n_tet x4 x4 x1) - (n_tet xj xk x1)
        aj_d_k_Id3 = np.zeros((self.n_tet,4,4,3,3))    # aj.dot ak x Id3 (n_tet x4 x4 x3 x3) - (n_tet xj xk x3 x3)
        np.einsum('...jj->...j',aj_d_k_Id3)[...] = aj_d_k[:,:,:,None]
        self.Bjk_set = self.la/2*aj_t_k + self.mu/2*(ak_t_j + aj_d_k_Id3) # (n_tet x4 x4 x3 x3)

        # Cjkl = la/2 * aj(ak.dot(al)) + mu/2 *[al(aj.dot(ak)) +  ak(aj.dot(al))]
        #aj_d_k = np.einsum('ijn,ikn->ijk',self.aj_set,self.aj_set)
        aj_k_d_l = np.einsum('ijn,ikl->ijkln',self.aj_set,aj_d_k) # aj (ak.dot al) (n_tet x4 x4 x4 x3) (n_tet xj xk xl x3)
        al_j_d_k = np.einsum('iln,ijk->ijkln',self.aj_set,aj_d_k)
        ak_j_d_l = np.einsum('ikn,ijl->ijkln',self.aj_set,aj_d_k)
        self.Cjkl_set = self.la/2*aj_k_d_l + self.mu/2*(al_j_d_k + ak_j_d_l) # (n_tet x4 x4 x4 x3) (n_tet xj xk xl x3)

        # Djklm = la/8*aj.dot(ak)*al.dot(am) + mu/4 *aj.dot(am)*ak.dot(al)
        aj_d_k = np.einsum('ijn,ikn->ijk',self.aj_set,self.aj_set)
        aj_d_k__l_d_m = np.einsum('ijk,ilm->ijklm',aj_d_k,aj_d_k)
        aj_d_m__k_d_l = np.einsum('ijm,ikl->ijklm',aj_d_k,aj_d_k)
        self.Djklm_set = self.la/8*aj_d_k__l_d_m + self.mu/4*aj_d_m__k_d_l
    
    def _update_Fes(self):
        self.u = self.x - self.vertices
        self.Fes = np.zeros((self.n_v, 3))
        for p in range(self.n_v):
            r_tet,p_index = np.where(self.tet_elements==p)
            vtx_index = self.tet_elements[r_tet] # (r_tet x 4)
            uj = self.u[vtx_index].reshape(-1,4,3) # Uj (r_tet x4  x3 ) - (r_tet xj x3)
            
            # F1p (r_tet x4 x3) - (r_tet xj x3)
            # F1p = 2 * sum Bpj * Uj
            Bpj = self.Bjk_set[r_tet,p_index] # Bpj (r_tet x4 x3 x3) - (r_tet xj x3 x3)
            # times volume , derive from scale test
            Bpj = Bpj * self.volumes6[r_tet][:,None,None,None] / 6 # (r_tet x4 x3 x3) * (r_tet x1 x1 x1)
            F1p = 2*( Bpj * uj[:,:,None,:]).sum(axis = -1) #print(Bpj.shape); print(F1p.shape)
            
            # F2p (r_tet x4 x4 x3) - (r_tet xj xk x3)
            # F2p = sum 2*(Uk tensorx Uj) Cjkp + Uj.dot(Uk) Cpjk
            # Cjkp (r_tet x4 x4 x3) - (r_tet xj xk x3)
            # Cpjk (r_tet x4 x4 x3) - (r_tet xj xk x3)
            uk_t_j = np.einsum('ijl,ikn->ijknl',uj,uj) # Uk tensorx Uj (r_tet x4 x4 x3 x3) - (r_tet xj xk x3 x3)
            uj_d_k = np.einsum('ijn,ikn->ijk',uj,uj) # Uj.dot(Uk)    (r_tet x4 x4 x3)    - (r_tet xj xk x3)
            Cjkp = self.Cjkl_set[r_tet,:,:,p_index]
            Cpjk = self.Cjkl_set[r_tet,p_index]
            # times volume, derive from scale test
            Cjkp = Cjkp * self.volumes6[r_tet][:,None,None,None] / 6 # (r_tet x4 x4 x3) * (r_tet x1 x1 x1)
            Cpjk = Cpjk * self.volumes6[r_tet][:,None,None,None] / 6 # (r_tet x4 x4 x3) * (r_tet x1 x1 x1)
            F2p1 = (uk_t_j*Cjkp[:,:,:,None]).sum(axis=-1)
            F2p2 = uj_d_k[:,:,:,None] * Cpjk  # print(uk_t_j.shape);print(uj_d_k.shape);print(Cjkp.shape);print(Cpjk.shape);print(F2p1.shape);print(F2p2.shape)
            F2p = 2* F2p1 + F2p2 # (r_tet x4 x4 x3) - (r_tet xj xk x3)
            
            # F3p (r_tet x4 x4 x4 x3) - (r_tet xj xk xl x3)
            Djklp = self.Djklm_set[r_tet,:,:,:,p_index] # Djklp (r_tet x4 x4 x4) - (r_tet xj xk xl)
            # times volume , derive from scale test
            Djklp = Djklp * self.volumes6[r_tet][:,None,None,None] / 6 # (r_tet x4 x4 x4) * (r_tet x1 x1 x1)
            uj_t_k_j = np.einsum('ijn,ikn,ilm->ijklmn',uj,uj,uj).sum(axis=-1) # (Ul tensorx Uk )Uj (r_tet xj xk xl x3) - (r_tet x4 x4 x4 x3)
            F3p = 4*Djklp[:,:,:,:,None]*uj_t_k_j #print(Djklp.shape);print(uj_t_k_j.shape);print(F3p.shape)

            Fp = F1p.reshape(-1,3).sum(axis=0) + F2p.reshape(-1,3).sum(axis=0) + F3p.reshape(-1,3).sum(axis=0)
            self.Fes[p] = Fp

    def _update_Fincompressible(self):
        # the original not obey the scale principle, remove tri_nv normalization, and change factor
        tri_vtx = self.x[self.tri_set] # find all triangle vertices, not only surface, (n_tet x4) x3
        tri_nv = np.cross(tri_vtx[:,1]-tri_vtx[:,0],tri_vtx[:,2]-tri_vtx[:,0])
        #tri_nv = tri_nv *(1/np.linalg.norm(tri_nv,axis=-1))[:,None] # find Nip, (n_tet x4) x3        
        if self.top_vtx_index is None:
            self.top_vtx_index = self.tet_elements[:,self.top_vtx].flatten() # F performs vertices index, (n_tet x4) x1
        #factor = np.power((self.volumes6 - self.init_volume6)*(1/self.init_volume6),2) * np.sign(self.volumes6 - self.init_volume6)# find tet index, n_tet x1
        factor = np.power((self.volumes6 - self.init_volume6)*(1/self.init_volume6),1)
        if self.tet_index is None: 
            self.tet_index = np.repeat(np.arange(self.n_tet),4) # (n_tet x4) x1
        Ficp = factor[self.tet_index][:,None]*tri_nv
        self.Ficp = np.zeros((self.n_v,3))# add force to vertices 
        for p in range(self.n_v):
            idx = np.argwhere(self.top_vtx_index==p).flatten()
            self.Ficp[p] = Ficp[idx].sum(axis=-2)

    def _pre_fixed_handle(self):
        self.Fis[self.fixed_vindex] = 0

    def _post_fixed_handle(self):
        self.x[self.fixed_vindex] = self.xp[self.fixed_vindex]
        self.sp[self.fixed_vindex] = 0

    def _explicit_step(self):              
        # Mode 1 : Euler
        x_next = self._explicit_Euler(self.x,self.xp,self.dt,self.gamma,self.dm,self.Fis)
        self.xp = self.x.copy()
        self.x = x_next.copy()
        
        # Mode 2: RK4
        #self.x, self.sp = self._update_RK4(self.x, self.sp, self.dt, self.gamma, self.dm, self.Fis)

    def _explicit_Euler(self,xp,xpp,dt,gamma,dm,Fis):
        # (dm/dt^2-gamma/2/dt)*x = Fis + 2*dm/dt^2*xp-(dm/dt^2+gamma/2/dt)*xpp
        x = (Fis + 2*dm/np.power(dt,2)*xp-(dm/np.power(dt,2)+gamma/2/dt)*xpp)/(dm/np.power(dt,2)-gamma/2/dt)
        return x

    def _explicit_RK4(self, x, sp, dt, gamma, dm, Fis):
        # dv /dt = F(t,scl_x,v) = gamma/dm* v + Fi/dm
        k1dt = (gamma / dm * sp + Fis / dm) * dt
        k2dt = (gamma / dm * (sp + k1dt / 2) + Fis / dm) * dt
        k3dt = (gamma / dm * (sp + k2dt / 2) + Fis / dm) * dt
        k4dt = (gamma / dm * (sp + k3dt) + Fis / dm) * dt
        spn = sp + (k1dt + 2 * k2dt + 2 * k3dt + k4dt) / 6

        # dx /dt = F(t,scl_x,v) = v
        k1dt = sp * dt
        k2dt = (sp + k1dt / 2) * dt
        k3dt = (sp + k2dt / 2) * dt
        k4dt = (sp + k3dt) * dt
        xn = x + (k1dt + 2 * k2dt + 2 * k3dt + k4dt) / 6
        return xn, spn
    
    ''' Plot function
    4. plt_Ficp # vtx:     dark blue  , nv: light blue
    3. plt_Fis  # vtx:     dark pink  , nv: light pink
    2. plt_Fes  # tri_mid: dark purple, nv: light purple
    1. _plt_Fs  # vtx:     dark pink  , nv: light pink 
    '''

    def plt_Ficp(self,vtx_scl=1,vec_to_scl=5,**kwargs):
        ax = ax3d_handle(**kwargs)
        ax = self._plt_Fs(self.Ficp,vtx_scl=vtx_scl,vec_scl=vec_to_scl,
                          cp='#003380',cnv='#92b7f0',
                          ax = ax)
        return ax

    def plt_Fis(self,vtx_scl=1,vec_to_scl=5,**kwargs):
        ax = ax3d_handle(**kwargs)
        ax = self._plt_Fs(self.Fis,vtx_scl=vtx_scl,vec_scl=vec_to_scl,
                          cp='#f57dd7',cnv='#c20091',
                          ax = ax)
        return ax 

    def plt_Fes(self,vtx_scl=1,vec_to_scl=5,**kwargs):
        ax = ax3d_handle(**kwargs)
        ax = self._plt_Fs(-self.Fes,vtx_scl=vtx_scl,vec_scl=vec_to_scl,
                          cp='#7D75FE',cnv='#1D1788',
                          ax = ax)
        return ax

       

    def _plt_Fs(self,Fs,vtx_scl=1,vec_to_scl=5,cp='#f57dd7',cnv='#c20091',**kwargs):
        ax = ax3d_handle(**kwargs)
        F_max = np.linalg.norm(Fs,axis=-1).max()
        vec_scl = vec_to_scl / F_max
        ax = self._plt_ps_normal_vecs(self.x*vtx_scl,Fs*vec_scl,
                        cp=cp,cnv=cnv,    
                        ax = ax)
        ax.title.set_text(f"Fes max: {F_max:.3E}")
        return ax