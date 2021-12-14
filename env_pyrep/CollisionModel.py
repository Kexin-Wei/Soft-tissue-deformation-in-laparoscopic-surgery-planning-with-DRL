import numpy as np
from itertools import product
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from mpl_toolkits import mplot3d
from .utils import ax3d_handle


class CollisionModel:
    def __init__(self, x=None, tet_elements=None):
        if x is None or tet_elements is None:
            self.aabb = None
            self.tet_aabb = None
        else:      
            self._update_AABB(x, tet_elements)

    def _update_AABB(self, x, tet_elements):
        # x: n_vx3
        self.aabb = np.r_[x.min(axis=0), x.max(axis=0)]
        self.tet_aabb = np.c_[x[tet_elements].min(axis=1), x[tet_elements].max(axis=1)]

    @staticmethod
    def xyzminmax(aabb):
        # xmin, ymin, zmin, xmax, ymax, zmax = aabb[0], aabb[1], aabb[2], aabb[3], aabb[4], aabb[5]
        return aabb[0], aabb[1], aabb[2], aabb[3], aabb[4], aabb[5]

    def _check_aabb_collision(self, b_aabb):
        # self.aabb as a_aabb
        def check_overlap(a_aabb, b_aabb):
            axmin, aymin, azmin, axmax, aymax, azmax = self.xyzminmax(a_aabb)
            bxmin, bymin, bzmin, bxmax, bymax, bzmax = self.xyzminmax(b_aabb)
            if axmin < bxmax and bxmin < axmax and aymin < bymax and bymin < aymax and azmin < bzmax and bzmin < azmax:
                return True
            return False

        if check_overlap(self.aabb, b_aabb) or check_overlap(b_aabb, self.aabb):
            return True
        return False
    
    def _check_aabb_p(self, points):
        # find points in aabb
        # self_vertices is usage for self.x / self.vertices
        check1= points - self.aabb[:3]
        check2= self.aabb[-3:] - points
        return np.argwhere(np.all(check1>0,axis=1)&np.all(check2>0,axis=1)).flatten()

    def _check_tet_aabb_collision(self,self_vertices,self_tet_elements,self_volumes,points):      
        # 1. find points in aabb
        # self_vertices is usage for self.x / self.vertices
        # so is self_volumes
        vin_index = self._check_aabb_p(points)

        # 2. find potential tet_aabb colliding with p
        check1 = points[vin_index,np.newaxis]-self.tet_aabb[:,:3]
        check2 = self.tet_aabb[:,-3:]-points[vin_index,np.newaxis]
        pot_col_tet = np.argwhere(np.all(check1>0,axis=2)&np.all(check2>0,axis=2))
        col_tet_n = [pot_col_tet[np.argwhere(pot_col_tet[:,0]==i),1].flatten() for i in np.unique(pot_col_tet[:,0])]
        col_p_n = vin_index[np.unique(pot_col_tet[:,0])]

        # 3. check barycentric, find points really in tet
        col_pair = {} # col_pair = {index of tet:ps, ..: ..}
        col_points = [] # col_points = {ps,,,,}  a collection for all p in tets
        for col_tet_ni,p_n in zip(col_tet_n,col_p_n):
            col_tet,p = self_vertices[self_tet_elements[col_tet_ni]],points[p_n]
            
            vs = np.tile(col_tet,(1,4)).reshape(-1,4,4,3).transpose(0,2,1,3)
            vs[:,[0,1,2,3],[0,1,2,3]] = p
            
            bc_matrix = np.ones((vs.shape[0],4,4,4))
            bc_matrix[:,:,:,1:] = vs            
            bc_co = (np.linalg.det(bc_matrix).T*(1/self_volumes[col_tet_ni]).T).T

            check1 = np.where(np.all(bc_co>0,axis=1))[0] # bc_co_i >0
            check2 = np.where(abs(np.sum(bc_co,axis=1)-1)<1e-3)[0] # sum bc_co_i =1

            col_tet_p = np.intersect1d(check1,check2)
            if len(col_tet_p) >0 : 
                col_pair.setdefault(f'{col_tet_ni[col_tet_p][0]}',[]).append(p_n)
                col_points.append(p_n)
                #print(f'Point {p} in tet {col_tet_ni[col_tet_p]}')
        return col_points #,col_pair 

    
    def plt_AABB(self, **kwargs):
        ax = self._plt_AABB(self.aabb, **kwargs)
        return ax

    def _plt_AABB(self, aabb, **kwargs):
        c_line = '#9467bd'
        c_p    = '#e377c2'
        if 'c' in kwargs.keys():
            colors = kwargs['c']
            if type(colors) is list:
                c_line = colors[0]
                c_p = colors[1]
            elif type(colors) is str:
                c_line = colors
        ax = ax3d_handle(kwargs)
                        
        # aabb: 1x6, xmin, ymin, zmin, xmax, ymax, zmax
        xmin, ymin, zmin, xmax, ymax, zmax = self.xyzminmax(aabb)
        xyz = np.array([xmin, ymin, zmin, xmax, ymin, zmin, xmax, ymax, zmin, xmin, ymax, zmin,
                        xmin, ymin, zmax, xmax, ymin, zmax, xmax, ymax, zmax, xmin, ymax, zmax]).reshape(-1, 3)
        line_segs = np.array([1, 2, 2, 3, 3, 4, 4, 1,
                              1, 5, 2, 6, 3, 7, 4, 8,
                              5, 6, 6, 7, 7, 8, 8, 5]).reshape(-1, 2) - 1
        
        line_vt = np.hstack((xyz[line_segs[:, 0]], xyz[line_segs[:, 1]])).copy()
        lc = Line3DCollection(line_vt.reshape(-1, 2, 3), colors=c_line, linestyles='--')
        ax.add_collection(lc)
        ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], marker='o', c=c_p)
        return ax

