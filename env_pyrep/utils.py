import os
import json
import numpy as np
import itertools
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from mpl_toolkits import mplot3d

def liver_dump_init(env, name = None):
    liver = {'x':[],'Fes':[],'Fis':[],'Ficp':[],'volume':[],'col_p_n':[],'crash':[]} 
    liver['vtx'] = env.liver.x.copy()        
    if name is not None:
        liver['name'] = name
    else:
        liver['name'] = f"_dt{env.timestep}_down_gm{env.liver.gamma}"
    return liver
    
def liver_dump_step(liver,env):
    liver['x'].append(env.liver.x)
    liver['Fes'].append(env.liver.Fes)
    liver['Fis'].append(env.liver.Fis)
    liver['Ficp'].append(env.liver.Ficp)
    liver['volume'].append(np.round(env.liver.volumes6.sum() / env.liver.init_volume6.sum(),3))
    liver['col_p_n'].append(len(env.liver.check_tet_aabb_collision(env.sg.x)))
    liver['crash'].append(env.liver.crash_flag)
    return liver

def liver_dump(liver,ep = None):
    liver_save ={}
    liver_save['vtx'] = liver['vtx'].tolist()
    liver_save['x']    = np.array(liver['x']).tolist()
    liver_save['Fes']  = np.array(liver['Fes']).tolist()
    liver_save['Fis']  = np.array(liver['Fis']).tolist()
    liver_save['Ficp'] = np.array(liver['Ficp']).tolist()
    liver_save['volume'] = np.array(liver['volume']).tolist()
    liver_save['col_p_n']= np.array(liver['col_p_n']).tolist()
    liver_save['crash']   = np.array(liver['crash']).tolist()
    if ep is None:
        with open(os.path.join('liver_json',f"liver_record{liver['name']}.json"),'w') as f:
            json.dump(liver_save,f)
    else:
        with open(os.path.join('liver_json',f"liver_record_{int(ep)}.json"),'w') as f:
            json.dump(liver_save,f)

def liver_dump_load(liver):
    vtx  = np.array(liver['vtx'])
    x    = np.array(liver['x'])
    Fes  = np.array(liver['Fes'])
    Fis  = np.array(liver['Fis'])
    Ficp = np.array(liver['Ficp'])
    volume = np.array(liver['volume'])
    col_p_n = np.array(liver['col_p_n'])
    crash   = np.array(liver['crash'])
    return vtx, x, Fes, Fis, Ficp, volume, col_p_n, crash
'''
temp:
    1. collision_response_cotin
    2. collision_response_self
'''
def collision_response_cotin(pair,liver,past_p,current_p):
    # check bc_co for all surface tri_element
    # add dn to decide
    move_v_disp_dict = {}
    move_tri_indexs = []
    flat_list = [item for sublist in list(pair.values()) for item in sublist]
    p_indexs = np.array(flat_list).reshape(-1)
    p_n = p_indexs.shape[0]
    ray = current_p[p_indexs]-past_p[p_indexs] 
    ray = ray*(1/np.linalg.norm(ray,axis=-1))[:,None] # p_n x3
    
    # compute ray and normal vector, d= ray,n=normal_vec
    dn = ray@liver.tri_normal_vec.T # p_n x n_tri
    ap = liver.x[liver.tri_elements[:,0]][None,:] - past_p[p_indexs][:,None] # p_n x n_tri x 3 #choose first point as a 
    apn = (ap * liver.tri_normal_vec[None,:]).sum(axis=-1) # p_n x n_tri x 3 -> p_n x n_tri
    ts = apn * (1/dn) # p_n x n_tri
    int_p = ts[:,:,None]*ray[:,None]+past_p[p_indexs][:,None] # p_n x n_tri x3 <- p_n x n_tri x1 * p_n x1 x3  + p_n x1 x3

    # compute barycentric coordinates of intersection points
    v1 = liver.x[liver.tri_elements[:,1]]-liver.x[liver.tri_elements[:,0]] # n_tri x3
    v2 = liver.x[liver.tri_elements[:,2]]-liver.x[liver.tri_elements[:,0]]
    tri_areax2 = np.linalg.norm(np.cross(v1,v2,axis=-1),axis=-1) # n_tri
    
    bc_temp = np.zeros((p_n,liver.n_tri,3,3,3))
    bc_temp[:] = np.tile(liver.x[liver.tri_elements], 3).reshape(-1, 3, 3, 3).transpose(0, 2, 1, 3)  # p_n x n_tri x 3area x 3ps x 3
    for itemp in range(p_n):
        bc_temp[itemp, :, [0, 1, 2], [0, 1, 2]] = int_p[itemp]
    v1 = bc_temp[:, :, :, 1] - bc_temp[:, :, :, 0]  # p_n x n_tri x 3area x 3xyz
    v2 = bc_temp[:, :, :, 2] - bc_temp[:, :, :, 0]
    areax2 = np.linalg.norm(np.cross(v1, v2, axis=-1), axis=-1)  # p_n x n_tri x 3area
    bc_co = areax2 * (1.0 / tri_areax2)[np.newaxis, :,
                         np.newaxis]  # p_n x n_tri x 3area<- p_n x n_tri x 3area * 1 x n_tri x 3area
       
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
            move_v_index_p = liver.tri_elements[move_tri_index]
            for ividx in move_v_index_p: # same points may move multiple times.
                if ividx not in move_v_disp_dict.keys(): 
                    move_v_disp_dict[ividx] = move_t # move_t put in for new vindex
                else:# compare move_t for old vindex
                    if np.linalg.norm(np.c_[move_v_disp_dict[ividx],move_t].T,axis=-1).argmax() == 1 : # older move closer than new
                        move_v_disp_dict[ividx] = move_t
            move_tri_indexs.append(move_tri_index.tolist())
    print(move_tri_indexs)
    return move_v_disp_dict

def collision_response_self(pair, liver, tool):
    # not so good when the deform is bigger
    # change to old fixed to test, problem still, try cotin methods
    new_vtx_delta = None
    move_tris = {}
    nv_aves = {}
    new_vtx_deltas = {}
    
    for key, value in pair.items():
        new_vtx_delta = np.zeros(liver.x.shape)
        i_tet, p_index = int(key), np.array(value)
        p_n = p_index.shape[0]

        # find potential collpaision surface tri_element
        col_tri_index = np.argwhere(liver.tri_tet[:, 0] == i_tet).flatten()
        if col_tri_index.size == 0: raise ValueError(
            "Update time step too big, vertices skip the surface tetrahedron elements")
        col_tri_n = col_tri_index.shape[0]
        col_tri_nv = liver.tri_normal_vec[col_tri_index]
        col_tri_p = liver.x[liver.tri_elements[col_tri_index].T[0]]  # chose the first points

        # compute nv_ave
        nv_ave = tool.vtx_normal_vec[p_index].sum(axis=0)
        nv_ave = nv_ave / np.linalg.norm(nv_ave)
        nv_aves[key] = nv_ave

        # compute ts and intersection points
        dn = nv_ave.dot(col_tri_nv.T)  # col_tri_n
        ap = col_tri_p[np.newaxis, :] - tool.x[p_index, np.newaxis]  # p_n x col_tri_n x 3
        dotn = np.tile(col_tri_nv, p_n).reshape(-1, p_n, 3).transpose(1, 0, 2)
        apn = (ap * dotn).sum(axis=-1)  # p_n x col_tri_n
        ts = apn * (1 / dn)  # p_n x col_tri_n
        int_col_p = ts[:, :, np.newaxis] * nv_ave[np.newaxis, np.newaxis, :] \
                    + tool.vertices[p_index][:, np.newaxis, :]  # p_n x col_tri_n x 1 * 1 x 1 x 3 + p_n x 1 x 3

        # compute barycentric coordinates of intersection points
        tri_vertices = liver.x[liver.tri_elements[col_tri_index]]  # n_tri x 3 x 3
        v1 = tri_vertices[:, 1] - tri_vertices[:, 0]
        v2 = tri_vertices[:, 2] - tri_vertices[:, 0]
        tri_areax2 = np.linalg.norm(np.cross(v1, v2, axis=-1), axis=-1)  # n_tri

        bc_temp = np.zeros((p_n, col_tri_n, 3, 3, 3))
        bc_temp[:] = np.tile(tri_vertices, 3).reshape(-1, 3, 3, 3).transpose(0, 2, 1, 3)  # p_n x col_tri_n x 3 x 3 x 3
        for itemp in range(p_n):
            bc_temp[itemp, :, [0, 1, 2], [0, 1, 2]] = int_col_p[itemp]
        v1 = bc_temp[:, :, :, 1] - bc_temp[:, :, :, 0]  # p_n x col_tri_n x 3area x 3xyz
        v2 = bc_temp[:, :, :, 2] - bc_temp[:, :, :, 0]
        areax2 = np.linalg.norm(np.cross(v1, v2, axis=-1), axis=-1)  # p_n x col_tri_n x 3area
        bc_co = areax2 * (1.0 / tri_areax2)[np.newaxis, :,
                         np.newaxis]  # p_n x col_tri_n x 3area * 1 x col_tri_n x 3area = p_n x col_tri_n x 3area

        # Move tri to point with tmax           
        check1 = np.argwhere(abs(bc_co.sum(axis=-1) - 1) < 1e-3)
        check2 = np.argwhere(dn < 0)
        inter_tri_index = np.intersect1d(check1[:, 1], check2) # find colliable surface tri_elements index
        # no colliable tri_elements
        if inter_tri_index.size == 0: 
            the_best_tri = dn.argmin()  # chose one of most collidable tri
            move_tri = liver.tri_elements[col_tri_index[the_best_tri]]
            tri_nv = liver.tri_normal_vec[col_tri_index[the_best_tri]].flatten()
            tri_vtx = liver.x[move_tri].reshape(3, 3)
            v = nv_ave - tri_nv  # find a new direction, not so sharp as nv_ave
            v = v / np.linalg.norm(v)
            dn_t = v.dot(tri_nv)  # 1
            ap_t = tri_vtx[0] - tool.x[p_index]
            t_t = ap_t.dot(tri_nv) / dn_t
            move_t = t_t.min()
            new_vtx_delta[move_tri] += - move_t * v
            new_vtx_deltas.setdefault(key, []).append(new_vtx_delta)
            move_tris.setdefault(key, []).append(move_tri.flatten())
            print(' None ',end='')
        else:
        # more than 1 colliable tri_elements
            if len(inter_tri_index) > 1:
                temp_delta = np.zeros((liver.x.shape[0], len(inter_tri_index)))  # n_v * n_inter
                itemp = 0
                for inter_tri_i in inter_tri_index:
                    part_p_index = check1[ check1[:, 1] == inter_tri_i, 0]  # p index of each tri_element that satisfies bc_co condition
                    move_t = ts[part_p_index, inter_tri_i].min()
                    move_tri = liver.tri_elements[col_tri_index[inter_tri_i]]
                    temp_delta[move_tri, itemp] = - move_t  # collect all possible move_t for all vertices
                    move_tris.setdefault(key, []).append(move_tri.flatten())
                    itemp += 1
                new_vtx_delta += temp_delta.max(axis=-1)[:, np.newaxis] * nv_ave[np.newaxis,:]  # move with the maximal move_t
                new_vtx_deltas.setdefault(key, []).append(new_vtx_delta)
                print(' Multi ',end='')
            else:
        # only 1 colliable tri_elements
                move_t = ts[:, inter_tri_index].min()
                move_tri = liver.tri_elements[col_tri_index[inter_tri_index]]
                new_vtx_delta[move_tri] += -move_t * nv_ave
                new_vtx_deltas.setdefault(key, []).append(new_vtx_delta)
                move_tris.setdefault(key, []).append(move_tri.flatten())
                print(' Single ',end='')
    return new_vtx_delta, move_tris, nv_aves,  new_vtx_deltas

'''
static methods:
    1. lame_param
    2. tri_mid_vec
    3. rotation_matrix
    4. flatten_list
'''
def lame_param(E, v):
    la = E * v / (1 + v) / (1 - 2 * v)
    mu = E / 2 / (1 + v)
    return la, mu

def tri_mid_vec(vertices, tri_elements):
    tri_vtx = vertices[tri_elements]
    tri_mid = tri_vtx.mean(axis=1)
    tri_normal_vec = np.cross(tri_vtx[:, 1] - tri_vtx[:, 0], tri_vtx[:, 2] - tri_vtx[:, 0])
    tri_normal_vec = tri_normal_vec * (1.0 / np.linalg.norm(tri_normal_vec, axis=1))[:, np.newaxis]
    return tri_mid, tri_normal_vec

def rotation_matrix(deg,axis='x'):
    rad = np.deg2rad(deg)
    s,c = np.sin(rad),np.cos(rad)
    if axis=='x':
        return np.array([  1,  0,  0,
                           0,  c, -s,
                           0,  s,  c]).reshape(-1,3)
    elif axis=='y':
        return np.array([  c,  0,  s,
                           0,  1,  0,
                          -s,  0,  c]).reshape(-1,3)
    elif axis=='z':
        return np.array([  c, -s,  0,
                           s,  c,  0,
                           0,  0,  1]).reshape(-1,3)
    else:
        return np.ones((3,3))

# def flatten_list(l):
#     # not work well
#     for el in l:
#         if isinstance(el, Iterable) and not isinstance(el, (str, bytes)):
#             return flatten_list(el)
#         else:
#             return el
'''
matplotlibe subplot
    1. create_axs
    2. draw_liver
    3. draw_liver_tool
'''
def create_axs(subplot_n,block=False,return_fig=False):
    r = int(np.floor(np.sqrt(subplot_n)))
    c = int(subplot_n/r)
    fig = plt.figure(figsize=plt.figaspect(0.5))
    axs = {}
    for i in range(subplot_n):
        axs[i] = fig.add_subplot(r, c, i+1, projection='3d')
    if return_fig:
        return axs,fig
    return axs

def draw_liver(liver,ax):
    ax.cla()
    ax = liver.plt_vtx(ax=ax)
    ax = liver.plt_x(ax=ax)
    plt_equal(ax)
    return ax

def draw_liver_F(liver,axs,f_scl = 5e0):
    # Fes, Ficp, Fis+ displacement
    axs[0].cla()    
    axs[0] = liver.plt_x(ax=axs[0])
    axs[0] = liver.plt_Fes(vec_to_scl=f_scl,ax=axs[0])
    plt_equal(axs[0])
    axs[1].cla()
    axs[1] = liver.plt_x(ax=axs[1])
    axs[1] = liver.plt_Ficp(vec_to_scl=f_scl,ax=axs[1])
    plt_equal(axs[1])
    axs[2].cla()
    axs[2] = liver.plt_vtx(ax=axs[2])
    axs[2] = liver.plt_x(ax=axs[2])
    axs[2] = liver.plt_Fis(vec_to_scl=f_scl,ax=axs[2])
    plt_equal(axs[2])
    return axs

def draw_liver_tool(liver,sg,axs,f_scl=5e0):
    axs[0].cla()    
    axs[0] = liver.plt_x(ax=axs[0])
    axs[0] = liver.plt_tri_normal_vec(vec_scl=f_scl/2,ax=axs[0])
    plt_equal(axs[0])
    axs[1].cla()
    axs[1] = sg.plt_sg_x(ax=axs[1])
    axs[1] = sg._plt_vtx_normal_vec(sg.x,vec_scl=f_scl/2,ax=axs[1])
    plt_equal(axs[1])
    axs[2].cla()
    axs[2] = liver.plt_x(ax=axs[2])
    axs[2] = sg.plt_sg_x(ax=axs[2])
    plt_equal(axs[2])            
    axs_l = {axs[3],axs[4],axs[5]}
    axs_l = draw_liver(liver,axs_l,f_scl=f_scl) 
    axs[3],axs[4],axs[5] = axs_l[0],axs_l[1],axs_l[2]   
    plt.draw()#plt.show(block=False)
    return axs

'''
aabb
    1. xyzminmax
    2. _plt_AABB
    3. plt_aabb_p
'''
def xyzminmax(aabb):
    # xmin, ymin, zmin, xmax, ymax, zmax = aabb[0], aabb[1], aabb[2], aabb[3], aabb[4], aabb[5]
    return aabb[0], aabb[1], aabb[2], aabb[3], aabb[4], aabb[5]


def plt_AABB(aabb, **kwargs):
    c_line = '#9467bd'
    c_p = '#e377c2'
    if 'c' in kwargs.keys():
        colors = kwargs['c']
        if type(colors) is list:
            c_line = colors[0]
            c_p = colors[1]
        elif type(colors) is str:
            c_line = colors
    ax = ax3d_handle(**kwargs)

    # aabb: 1x6, xmin, ymin, zmin, xmax, ymax, zmax
    xmin, ymin, zmin, xmax, ymax, zmax = xyzminmax(aabb)
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


def plt_aabb_p(aabb, p, **kwargs):
    ax = ax3d_handle(**kwargs)
    ax.scatter(p[0], p[1], p[2], c='#22D8C3')
    plt_AABB(aabb, ax=ax)
    return ax

'''
ax handle
    1.  1) plt_equal
        2) plt_show_equal
        3) set_axes_equal
        4) _set_axes_radius
    2. ax3d_handle
    3. plt_tet
    4. plt_tet_ps
    5. plt_normal_vecs
    6. plt_tri
    7. plt_tri_ps
'''
def plt_equal(ax,limits = None):
    ax.set_box_aspect((1, 1, 1))  # IMPORTANT - this is the new, key line
    set_axes_equal(ax,limits=limits)  # IMPORTANT - this is also required

def plt_show_equal(ax,block=False,limits = None):
    plt_equal(ax,limits=limits)
    plt.show(block=block)

def set_axes_equal(ax: plt.Axes,limits = None):
        """Set 3D plot axes to equal scale.

        Make axes of 3D plot have equal scale so that spheres appear as
        spheres and cubes as cubes.  Required since `ax.axis('equal')`
        and `ax.set_aspect('equal')` don't work on 3D.
        """
        if limits is None:
            limits = np.array([
                ax.get_xlim3d(),
                ax.get_ylim3d(),
                ax.get_zlim3d(),
            ])
        origin = np.mean(limits, axis=1)
        radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
        _set_axes_radius(ax, origin, radius)

def _set_axes_radius(ax, origin, radius):
    x, y, z = origin
    ax.set_xlim3d([x - radius, x + radius])
    ax.set_ylim3d([y - radius, y + radius])
    ax.set_zlim3d([z - radius, z + radius]) 

def ax3d_handle(return_fig=False,**kwargs):
    if 'ax' in kwargs:
        ax = kwargs['ax']
    else:
        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(projection='3d')
    if return_fig:
        return ax,fig
    return ax





def plt_tet(vs, text_opt='off', **kwargs):
    ax = ax3d_handle(**kwargs)
    ax.scatter(vs[:, 0], vs[:, 1], vs[:, 2], c='#BCB6E3')
    if text_opt == "on":
        for i in range(4): ax.text(vs[i, 0], vs[i, 1], vs[i, 2], f'{i + 1}')
    line_order = np.array([1, 2, 1, 3, 1, 4, 2, 3, 2, 4, 3, 4]).reshape(-1, 2) - 1
    line_vt = np.hstack((vs[line_order[:, 0]], vs[line_order[:, 1]]))
    lc = Line3DCollection(line_vt.reshape(-1, 2, 3), colors='#8A7BFB')
    ax.add_collection(lc)
    return ax


def plt_tet_ps(vs, p, text_opt='off', **kwargs):
    p = np.array(p)
    ax = ax3d_handle(**kwargs)
    ax = plt_tet(vs, text_opt=text_opt, ax=ax)
    if len(p.shape) == 1: p = p.reshape(1, -1)
    ax.scatter(p[:, 0], p[:, 1], p[:, 2], c='#22D8C3')
    return ax





def plt_normal_vecs(base_ps, vecs, scl=1, **kwargs):
    vesc_scl = vecs * scl
    ax = ax3d_handle(**kwargs)
    ax.scatter(base_ps[:, 0], base_ps[:, 1], base_ps[:, 2], c='#1D1788')
    ax.quiver(base_ps[:, 0], base_ps[:, 1], base_ps[:, 2],
              vesc_scl[:, 0], vesc_scl[:, 1], vesc_scl[:, 2], color='#7D75FE')
    return ax


def plt_tet_ps_vecs(vs, p, vec, scl=1, text_opt = 'off', **kwargs):
    ax = ax3d_handle(**kwargs)
    ax = plt_tet_ps(vs, p, ax=ax, text_opt = text_opt)
    if len(p.shape) == 1:     p = p.reshape(1, -1)
    if len(vec.shape) == 1: vec = vec.reshape(1, -1)
    ax = plt_normal_vecs(p, vec, scl=scl, ax=ax)
    return ax


def plt_tri(vs, text_opt='off', **kwargs):
    ax = ax3d_handle(**kwargs)
    ax.scatter(vs[:, 0], vs[:, 1], vs[:, 2], c='#ff00ff')
    if text_opt == "on":
        for i in range(3): ax.text(vs[i, 0], vs[i, 1], vs[i, 2], f'{i + 1}')
    line_order = np.array([1, 2, 1, 3, 2, 3]).reshape(-1, 2) - 1
    line_vt = np.hstack((vs[line_order[:, 0]], vs[line_order[:, 1]]))
    lc = Line3DCollection(line_vt.reshape(-1, 2, 3), colors='#9933ff')
    ax.add_collection(lc)
    return ax


def plt_tri_ps(vs, p, text_opt='off', **kwargs):
    ax = ax3d_handle(**kwargs)
    ax = plt_tri(vs, text_opt=text_opt, ax=ax)
    if len(p.shape) == 1: p = p.reshape(1, -1)
    ax.scatter(p[:, 0], p[:, 1], p[:, 2], c='#22D8C3')
    return ax
