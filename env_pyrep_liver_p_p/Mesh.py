import numpy as np
import json
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from mpl_toolkits import mplot3d
from .utils import ax3d_handle

class Mesh:
    def __init__(self):
        self.vertices = None # in meter
        self.tri_elements = None
        self.tet_elements = None
        self.fixed_vindex = None
        self.n_v = None
        self.n_tri = None
        self.n_tet = None

        self.x = None 
        self.xp = None
        self.sp = None

        self.surf_vindex = None
        self.tri_line_segs = None
        self.tet_line_segs = None        

        self.tri_ele = np.array([1,3,2,1,2,4,1,4,3,2,3,4]).reshape(-1,3)-1 #in outer normal vector order 
        self.top_vtx = np.array([4,3,2,1])-1
        self.tri_line_comb = np.array([1, 2, 1, 3, 2, 3]).reshape(-1, 2) - 1
        self.tet_line_comb = np.array([1, 2, 1, 3, 1, 4, 2, 3, 2, 4, 3, 4]).reshape(-1, 2) - 1
        
    @staticmethod
    def lame_param(E, v):
        la = E * v / (1 + v) / (1 - 2 * v)
        mu = E / 2 / (1 + v)
        return la, mu
    '''
    basic self.XXX update
    1. read_mesh_json
    2. _get_ns
    3. _handle_tri_elements
        3.1 _update_tri_elements
        3.2 _update_tri_tet
        3.3 _update_tri_normal_vecs
        3.4 _update_vtx_normal_vecs
    4. _get_tet_line_segs 
    5. _get_tri_line_segs
    6. _update_normal_vecs
    7. update_normal_vecs
    '''
    def read_mesh_json(self, mesh_filename):
        mesh_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),mesh_filename)
        with open(mesh_file, 'r') as file:
            liver_mat = json.load(file)
        for key, value in liver_mat.items():
            exec('self.' + key + '=np.array(value)')

    def _get_ns(self):
        self.n_v = self.vertices.shape[0]
        self.n_tri = self.tri_elements.shape[0]
        if self.tet_elements is not None:
            self.n_tet = self.tet_elements.shape[0]
    
    def _update_tri_elements(self):
        # find tri_elements order base on tet_elements         
        self.tri_set = self.tet_elements[:,self.tri_ele].reshape(-1,3) # all the triangle, has repeating triangle
        temp = [tuple(row) for row in np.sort(self.tri_set,axis=1)]
        temp2,indices, counts = np.unique(temp,axis=0,return_index=True,return_counts=True)
        self.tri_elements = self.tri_set[indices[np.argwhere(counts==1)]].reshape(-1,3)  

    def _update_tri_tet(self):
        # find the tri elements relationship to tet elements, [ n_tri x 2 ]
        tri_tet = np.zeros((self.tri_elements.shape[0],3))  
        i = 0
        for tri_ele in self.tri_elements:
            match_tet = np.argwhere((self.tri_set == tuple(tri_ele)).all(axis=1)).squeeze()
            tri_tet[i] = np.r_[match_tet//4,match_tet%4,self.top_vtx[match_tet%4]].astype(int) # ith tet elements, which face in tri_ele order
            i+=1
        self.tri_tet = tri_tet
        
    def _update_tri_normal_vecs(self,vertices):
        # find center points and the normal vector of each triangle element
        tri_vtx = vertices[self.tri_elements]
        self.tri_mid = tri_vtx.mean(axis=1)
        tri_normal_vec = np.cross(tri_vtx[:,1]-tri_vtx[:,0],tri_vtx[:,2]-tri_vtx[:,0])
        self.tri_normal_vec = tri_normal_vec*(1.0/np.linalg.norm(tri_normal_vec,axis=1))[:,np.newaxis]        

    def _update_vtx_normal_vecs(self):
        # find vertices normal vector as sum of adjacent triangle elements
        if self.surf_vindex is None:
            self.surf_vindex = np.unique(self.tri_elements)
        vtx_normal_vec = [self.tri_normal_vec[
                    np.argwhere(self.tri_elements==iv)[:,0]
                    ].sum(axis=0) for iv in self.surf_vindex]
        # equal to
        # vtx_normal_vec = np.zeros((self.surf_vindex.size,3))
        # i = 0
        # for iv in self.surf_vindex:
        #     iv_tri_index = np.argwhere(self.tri_elements==iv)[:,0]
        #     iv_tri_nv = self.tri_normal_vec[iv_tri_index]    
        #     vtx_normal_vec[i] = iv_tri_nv.sum(axis=0)
        #     i+=1
        self.vtx_normal_vec = vtx_normal_vec *(1/np.linalg.norm(vtx_normal_vec,axis=-1))[:,np.newaxis]
        
    def _handle_tri_elements(self,vertices):
        # vertices : self.vertices or self.x
        if self.tet_elements is not None:
            self._update_tri_elements()
            self._update_tri_tet()            
        self._update_tri_normal_vecs(vertices)
        self._update_vtx_normal_vecs()
        self._get_ns()
        self._get_tri_line_segs()
        if self.tet_elements is not None:
            self._get_tet_line_segs()

    def _get_tet_line_segs(self):
        tet_line_segs = np.zeros((self.tet_line_comb.shape[0], self.n_tet, 2))
        for i in range(self.tet_line_comb.shape[0]): tet_line_segs[i] = self.tet_elements[:, self.tet_line_comb[i]]
        temp = [tuple(row) for row in np.sort(tet_line_segs.reshape(-1, 2), axis=1)]
        self.tet_line_segs = np.unique(temp, axis=0).astype(int)
    
    def _get_tri_line_segs(self):
        tri_line_segs = np.zeros((self.tri_line_comb.shape[0], self.n_tri, 2))
        for i in range(self.tri_line_comb.shape[0]): tri_line_segs[i] = self.tri_elements[:, self.tri_line_comb[i]]
        temp = [tuple(row) for row in np.sort(tri_line_segs.reshape(-1, 2), axis=1)]
        self.tri_line_segs = np.unique(temp, axis=0).astype(int)
    
    def _update_normal_vecs(self,vertices):
        self._update_tri_normal_vecs(vertices)
        self._update_vtx_normal_vecs()

    def update_normal_vecs(self):
        self._update_normal_vecs(self.x)
    '''
    plot function section
    
    7. plt_tri_normal_vec  # tri_mid: light purple, nv: dark purple
    6. _plt_vtx_normal_vec # vtx:       light pink, nv: dark pink    
    5. plt_vtx 
    4. plt_x
    3. _plt_fixed_vtx      # vtx:     purple
    2. _plt_ps_normal_vecs # tri_mid: light purple, nv: dark purple
    1. _plt_msh            # tet:     blue       , tri: green         vtx:orange    
    '''
    def _plt_ps_normal_vecs(self,base_ps,nvs,cp='#7D75FE',cnv='#1D1788',**kwargs):
        ax = ax3d_handle(**kwargs)
        if len(base_ps.shape) == 1: base_ps = base_ps.reshape(-1,3)
        if len(nvs.shape) == 1:     nvs = np.matlib.repmat(nvs,base_ps.shape[0],1)
        ax.scatter(base_ps[:, 0], base_ps[:, 1], base_ps[:, 2], c=cp) 
        ax.quiver(base_ps[:, 0], base_ps[:, 1], base_ps[:, 2],
              nvs[:, 0], nvs[:, 1], nvs[:, 2], color=cnv) 
        return ax

    def plt_tri_normal_vec(self,vtx_scl=1,vec_scl=1,**kwargs):
        ax = ax3d_handle(**kwargs)
        ax = self._plt_ps_normal_vecs(self.tri_mid*vtx_scl,self.tri_normal_vec*vec_scl,
                    cp='#7D75FE',cnv='#1D1788',ax=ax)  #light purple  # dark purple      
        return ax

    def _plt_vtx_normal_vec(self,vertices,vtx_scl=1,vec_scl=1,**kwargs):
        ax = ax3d_handle(**kwargs)
        ax = self._plt_ps_normal_vecs(vertices*vtx_scl,self.vtx_normal_vec*vec_scl,
                    cp='#f57dd7',cnv='#c20091',ax=ax)  #light pink    # dark pink
        return ax
    
    def _plt_fixed_vtx(self, vertices, fixed_vindex,**kwargs):
        ax = ax3d_handle(**kwargs)
        ax.scatter(vertices[fixed_vindex, 0],
                   vertices[fixed_vindex, 1],
                   vertices[fixed_vindex, 2], c='#780EB1') #purple
        return ax

    def plt_x(self, draw_type='tri', text_opt='off', scl=1, **kwargs):
        ax = self._plt_msh(self.x*scl, draw_type=draw_type, text_opt=text_opt, **kwargs)
        if self.fixed_vindex is not None:
            ax = self._plt_fixed_vtx(self.x*scl,self.fixed_vindex, ax=ax)
        return ax

    def plt_vtx(self, draw_type='tri', text_opt='off', scl = 1,**kwargs): 
        ax = self._plt_msh(self.vertices*scl, draw_type=draw_type, text_opt=text_opt, **kwargs)
        return ax

    def _plt_msh(self, vertices, draw_type='tri', node_opt = 'on', text_opt='off', **kwargs):
        # draw_type = 'all', 'tri', 'tet', 'node'
        # text_opt  = 'off', 'on'
        if draw_type == 'all':
            tri_opt = 'on'
            tet_opt = 'on'

        elif draw_type == 'tri':
            tri_opt = 'on'
            tet_opt = 'off'

        elif draw_type == 'tet':
            tri_opt = 'off'
            tet_opt = 'on'

        else:
            tri_opt = 'off'
            tet_opt = 'off'

        ax = ax3d_handle(**kwargs)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

        # draw line
        if tet_opt == 'on': #blue
            if self.tet_line_segs is None:
                self._get_tet_line_segs()
            line_vt = np.hstack((vertices[self.tet_line_segs[:, 0]], vertices[self.tet_line_segs[:, 1]]))
            lc = Line3DCollection(line_vt.reshape(-1, 2, 3), colors='#6784ed')
            ax.add_collection(lc)

        if tri_opt == 'on': #green
            if self.tri_line_segs is None:
                self._get_tri_line_segs()
            line_vt = np.hstack((vertices[self.tri_line_segs[:, 0]], vertices[self.tri_line_segs[:, 1]])).copy()
            lc = Line3DCollection(line_vt.reshape(-1, 2, 3), colors='#6EBA58')
            ax.add_collection(lc)

        # draw node
        if node_opt == 'on':
            if tri_opt == 'on': #orange
                if self.surf_vindex is None:
                    self.surf_vindex = np.unique(self.tri_elements)
                ax.scatter(vertices[self.surf_vindex, 0], vertices[self.surf_vindex, 1], vertices[self.surf_vindex, 2],
                        marker='o', c='#eb8c23') 
                if text_opt == 'on':
                    for i in self.surf_vindex: ax.text(vertices[i, 0], vertices[i, 1], vertices[i, 2], f'{i}')
            else:
                ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], marker='o', c='#eb8c23')
                if text_opt == 'on':
                    for i in range(self.n_v): ax.text(vertices[i, 0], vertices[i, 1], vertices[i, 2], f'{i}')
        return ax
