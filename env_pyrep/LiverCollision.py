import numpy as np
import itertools
import matplotlib.pyplot as plt
from .utils import *
from .LiverMTM import LiverDeform
from .CollisionModel import CollisionModel

class LiverCollision(LiverDeform, CollisionModel):
    def __init__(self, mesh_filename='liver_iras_meter.json',E=1e4, v=0.4, gamma=-5, timeinterval=0.1,dt=1e-4,lamu_times=1e-1):
        LiverDeform.__init__(self,mesh_filename=mesh_filename,E=E,v=v,gamma=gamma,
                                    timeinterval=timeinterval,dt=dt,lamu_times=lamu_times)
        CollisionModel.__init__(self)        
        self.update_AABB()
    
    def update_AABB(self):
        self._update_AABB(self.x, self.tet_elements)

    def check_tet_aabb_collision(self, points):
        # return col_pair
        self.update_normal_vecs()
        return self._check_tet_aabb_collision(self.x, self.tet_elements, self.volumes6, points)
    
    def collision_response_ray(self,collision_points, tool):
        # tool based on Mesh
        tool.update_normal_vecs()
        move_v_disp_dict = {}
        move_tri_indexs = []
        p_indexs = np.array(collision_points)
        p_n = p_indexs.size
        ray = tool.vtx_normal_vec[p_indexs]

        # compute ray and normal vector, d= ray,n=normal_vec
        dn = ray@self.tri_normal_vec.T # p_n x n_tri
        ap = self.x[self.tri_elements[:,0]][None,:] - tool.x[p_indexs][:,None] # p_n x n_tri x 3 #choose first point as a 
        apn = (ap * self.tri_normal_vec[None,:]).sum(axis=-1) # p_n x n_tri x 3 -> p_n x n_tri
        ts = apn * (1/dn) # p_n x n_tri
        int_p = ts[:,:,None]*ray[:,None]+tool.x[p_indexs][:,None] # p_n x n_tri x3 <- p_n x n_tri x1 * p_n x1 x3  + p_n x1 x3

        # compute barycentric coordinates of intersection points
        v1 = self.x[self.tri_elements[:,1]]-self.x[self.tri_elements[:,0]] # n_tri x3
        v2 = self.x[self.tri_elements[:,2]]-self.x[self.tri_elements[:,0]]
        tri_areax2 = np.linalg.norm(np.cross(v1,v2,axis=-1),axis=-1) # n_tri

        bc_temp = np.zeros((p_n,self.n_tri,3,3,3))
        bc_temp[:] = np.tile(self.x[self.tri_elements], 3).reshape(-1, 3, 3, 3).transpose(0, 2, 1, 3)  # p_n x n_tri x 3area x 3ps x 3
        for itemp in range(p_n):
            bc_temp[itemp, :, [0, 1, 2], [0, 1, 2]] = int_p[itemp]
        v1 = bc_temp[:, :, :, 1] - bc_temp[:, :, :, 0]  # p_n x n_tri x 3area x 3xyz
        v2 = bc_temp[:, :, :, 2] - bc_temp[:, :, :, 0]
        areax2 = np.linalg.norm(np.cross(v1, v2, axis=-1), axis=-1)  # p_n x n_tri x 3area
        bc_co = areax2 * (1.0 / tri_areax2)[np.newaxis, :,
                                np.newaxis]  # p_n x n_tri x 3area<- p_n x n_tri x 3area * 1 x n_tri x 3area

        # check bc_co for all surface tri_element
        # add dn to decide            
        for itemp in range(p_n):
            # check bc_co
            check1 = np.argwhere(abs(bc_co[itemp].sum(axis=-1) - 1) < 1e-3).flatten() # each p should have at least 1
            check2 = np.argwhere(dn[itemp] < 0).flatten()
            psb_tri_index = np.intersect1d(check1,check2) # all possible tri_elements satisfies the bc_co and the negative normal vector
            if psb_tri_index.size!=0:
                psb_ts = ts[itemp,psb_tri_index] # n_psb_tri_index
                # if np.any(psb_ts<0):
                #     raise ValueError("liver shape error")

                # only 1 the tri_elements should move, the biggest negative
                move_tri_index = psb_tri_index[np.where(psb_ts<0,psb_ts,-np.inf).argmax()] 
                move_t = tool.x[p_indexs[itemp]] - int_p[itemp,move_tri_index]
                move_v_index_p = self.tri_elements[move_tri_index]
                for ividx in move_v_index_p: # same points may move multiple times.
                    if ividx not in move_v_disp_dict.keys(): 
                        move_v_disp_dict[ividx] = move_t # move_t put in for new vindex
                    else:# compare move_t for old vindex
                        if np.linalg.norm(np.c_[move_v_disp_dict[ividx],move_t].T,axis=-1).argmax() == 1 : # older move closer than new
                            move_v_disp_dict[ividx] = move_t
                move_tri_indexs.append(move_tri_index.tolist())
        #print('\t',move_tri_indexs,end='')
        return move_v_disp_dict


    def collision_response_cotin(self,collision_points,past_p,current_p):
        # works for past_p outside surface, current_p inside.
        
        move_v_disp_dict = {}
        move_tri_indexs = []
        p_indexs = np.array(collision_points)
        p_n = p_indexs.size
        ray = current_p[p_indexs]-past_p[p_indexs] 
        ray = ray*(1/np.linalg.norm(ray,axis=-1))[:,None] # p_n x3

        # compute ray and normal vector, d= ray,n=normal_vec
        dn = ray@self.tri_normal_vec.T # p_n x n_tri
        ap = self.x[self.tri_elements[:,0]][None,:] - past_p[p_indexs][:,None] # p_n x n_tri x 3 #choose first point as a 
        apn = (ap * self.tri_normal_vec[None,:]).sum(axis=-1) # p_n x n_tri x 3 -> p_n x n_tri
        ts = apn * (1/dn) # p_n x n_tri
        int_p = ts[:,:,None]*ray[:,None]+past_p[p_indexs][:,None] # p_n x n_tri x3 <- p_n x n_tri x1 * p_n x1 x3  + p_n x1 x3

        # compute barycentric coordinates of intersection points
        v1 = self.x[self.tri_elements[:,1]]-self.x[self.tri_elements[:,0]] # n_tri x3
        v2 = self.x[self.tri_elements[:,2]]-self.x[self.tri_elements[:,0]]
        tri_areax2 = np.linalg.norm(np.cross(v1,v2,axis=-1),axis=-1) # n_tri

        bc_temp = np.zeros((p_n,self.n_tri,3,3,3))
        bc_temp[:] = np.tile(self.x[self.tri_elements], 3).reshape(-1, 3, 3, 3).transpose(0, 2, 1, 3)  # p_n x n_tri x 3area x 3ps x 3
        for itemp in range(p_n):
            bc_temp[itemp, :, [0, 1, 2], [0, 1, 2]] = int_p[itemp]
        v1 = bc_temp[:, :, :, 1] - bc_temp[:, :, :, 0]  # p_n x n_tri x 3area x 3xyz
        v2 = bc_temp[:, :, :, 2] - bc_temp[:, :, :, 0]
        areax2 = np.linalg.norm(np.cross(v1, v2, axis=-1), axis=-1)  # p_n x n_tri x 3area
        bc_co = areax2 * (1.0 / tri_areax2)[np.newaxis, :,
                                np.newaxis]  # p_n x n_tri x 3area<- p_n x n_tri x 3area * 1 x n_tri x 3area

        # check bc_co for all surface tri_element
        # add dn to decide            
        for itemp in range(p_n):
            # check bc_co
            check1 = np.argwhere(abs(bc_co[itemp].sum(axis=-1) - 1) < 1e-3).flatten() # each p should have at least 1
            check2 = np.argwhere(dn[itemp] < 0).flatten()
            psb_tri_index = np.intersect1d(check1,check2) # all possible tri_elements satisfies the bc_co and the negative normal vector
            if psb_tri_index.size!=0:
                psb_ts = ts[itemp,psb_tri_index] # n_psb_tri_index
                # if np.any(psb_ts<0):
                #     raise ValueError("liver shape error")
                move_tri_index = psb_tri_index[psb_ts.argmin()] # only 1 the tri_elements should move
                move_t = current_p[p_indexs[itemp]] - int_p[itemp,move_tri_index]
                move_v_index_p = self.tri_elements[move_tri_index]
                for ividx in move_v_index_p: # same points may move multiple times.
                    if ividx not in move_v_disp_dict.keys(): 
                        move_v_disp_dict[ividx] = move_t # move_t put in for new vindex
                    else:# compare move_t for old vindex
                        if np.linalg.norm(np.c_[move_v_disp_dict[ividx],move_t].T,axis=-1).argmax() == 1 : # older move closer than new
                            move_v_disp_dict[ividx] = move_t
                move_tri_indexs.append(move_tri_index.tolist())
        #print('\t',move_tri_indexs,end='')
        return move_v_disp_dict

if __name__ == "__main__":
    from Mesh import Mesh

    liver = LiverCollision()
    test = Mesh()
    test.vertices = np.random.rand(10, 3) * 200
    ax = liver.plt_vtx(draw_type='node')
    ax.scatter(test.vertices[:, 0], test.vertices[:, 1], test.vertices[:, 2], c='blueviolet')
    plt.show()

    print(liver.check_tet_aabb_collision(test.vertices))