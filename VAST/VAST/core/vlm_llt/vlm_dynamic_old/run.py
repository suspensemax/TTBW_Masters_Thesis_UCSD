from VAST.core.vlm_llt.vlm_dynamic_old.VLM_prescribed_wake_solver import UVLMSolver

from VAST.utils.generate_mesh import *
from VAST.utils.make_video_vedo import make_video as make_video_vedo
import time
import numpy as np
# Script to create optimization problem

be = 'python_csdl_backend'
# be = 'csdl_om'
make_video = 0

########################################
# define mesh here
########################################
nx = 29; ny = 5
chord = 1; span = 12
num_nodes = 20;  nt = num_nodes

alpha = np.deg2rad(5)
u_val = np.concatenate((np.array([1]), np.ones(num_nodes-1))).reshape(num_nodes,1)

states_dict = {
    'u': u_val, 'v': np.zeros((num_nodes, 1)), 'w': np.zeros((num_nodes, 1)),
    'p': np.zeros((num_nodes, 1)), 'q': np.zeros((num_nodes, 1)), 'r': np.zeros((num_nodes, 1)),
    'theta': np.ones((num_nodes, 1))*alpha, 'psi': np.zeros((num_nodes, 1)),
    'x': np.zeros((num_nodes, 1)), 'y': np.zeros((num_nodes, 1)), 'z': np.zeros((num_nodes, 1)),
    'phiw': np.zeros((num_nodes, 1)), 'gamma': np.zeros((num_nodes, 1)),'psiw': np.zeros((num_nodes, 1)),
}

mesh_dict = {"num_y": ny, "num_x": nx, "wing_type": "rect",  "symmetry": False, "span": span, "root_chord": chord,"span_cos_spacing": False, "chord_cos_spacing": False}
mesh = generate_mesh(mesh_dict)

surface_properties_dict = {'wing':(nx,ny,3)}

# mesh_val = generate_simple_mesh(nx, ny, num_nodes)
mesh_val = np.zeros((num_nodes, nx, ny, 3))
z_offset = np.array([0,1,2,3,4,5,4,3,2,1,0,-1,-2,-3,-4,-5,-4,-3,-2,-1])
for i in range(num_nodes):
    mesh_val[i, :, :, :] = mesh
    mesh_val[i, :, :, 0] = mesh.copy()[:, :, 0] 
    mesh_val[i, :, :, 1] = mesh.copy()[:, :, 1] 
    mesh_val[i, :, :, 2] += z_offset[i]

h_stepsize = delta_t = 1 

if be == 'csdl_om':
    import csdl_om
    sim = csdl_om.Simulator(UVLMSolver(num_times=nt,h_stepsize=h_stepsize), mode='rev')
if be == 'python_csdl_backend':
    import python_csdl_backend
    sim = python_csdl_backend.Simulator(UVLMSolver(num_times=nt,h_stepsize=h_stepsize,states_dict=states_dict,
                                        surface_properties_dict=surface_properties_dict,mesh_val=mesh_val), mode='rev')
    
t_start = time.time()
sim.run()
print('simulation time is', time.time() - t_start)
np.savetxt('cl12full',sim['wing_C_L'])
######################################################
# make video
######################################################

if make_video == 1:
    make_video_vedo(surface_properties_dict,num_nodes,sim)
# sim.compute_totals(of='',wrt='*')
######################################################
# end make video
######################################################

# sim.visualize_implementation()
# partials = sim.check_partials(compact_print=True)
# sim.prob.check_totals(compact_print=True)