'''Example 3 : verification of prescibed vlm with Katz and Plotkin 1991'''

from VAST.core.vlm_llt.vlm_dynamic_old.VLM_prescribed_wake_solver import UVLMSolver

from VAST.utils.generate_mesh import *
from VAST.utils.make_video_vedo import make_video as make_video_vedo
import time
import numpy as np
# Script to create optimization problem

be = 'python_csdl_backend'
make_video = 0
plot_cl = 1
########################################
# define mesh here
########################################
nx = 15; ny = 5
chord = 1; span = 4
num_nodes = 8;  nt = num_nodes
n_period = 4
omg=1
h=0.1
alpha = - np.deg2rad(5)

t_vec = np.linspace(0, n_period*np.pi*2, num_nodes)

u_val = (np.ones(num_nodes) * np.cos(alpha)).reshape((num_nodes,1))
w_vel = np.ones((num_nodes,1)) * np.sin(alpha) - h * np.cos(omg*t_vec).reshape((num_nodes,1))

alpha_equ = np.arctan2(w_vel, u_val)

states_dict = {
    'u': u_val, 'v': np.zeros((num_nodes, 1)), 'w': w_vel,
    'p': np.zeros((num_nodes, 1)), 'q': np.zeros((num_nodes, 1)), 'r': np.zeros((num_nodes, 1)),
    'theta': alpha_equ, 'psi': np.zeros((num_nodes, 1)),
    'x': np.zeros((num_nodes, 1)), 'y': np.zeros((num_nodes, 1)), 'z': np.zeros((num_nodes, 1)),
    'phiw': np.zeros((num_nodes, 1)), 'gamma': np.zeros((num_nodes, 1)),'psiw': np.zeros((num_nodes, 1)),
}

mesh_dict = {"num_y": ny, "num_x": nx, "wing_type": "rect",  "symmetry": False, "span": span, "root_chord": chord,"span_cos_spacing": False, "chord_cos_spacing": False}
mesh = generate_mesh(mesh_dict)

surface_properties_dict = {'wing':(nx,ny,3)}

# mesh_val = generate_simple_mesh(nx, ny, num_nodes)
mesh_val = np.zeros((num_nodes, nx, ny, 3))
z_offset = h*sin(omg*t_vec)

for i in range(num_nodes):
    mesh_val[i, :, :, :] = mesh
    mesh_val[i, :, :, 0] = mesh.copy()[:, :, 0] 
    mesh_val[i, :, :, 1] = mesh.copy()[:, :, 1] 
    mesh_val[i, :, :, 2] += z_offset[i]

h_stepsize = t_vec[1] 

if be == 'csdl_om':
    import csdl_om
    sim = csdl_om.Simulator(UVLMSolver(num_times=nt,h_stepsize=h_stepsize,states_dict=states_dict,
                                        surface_properties_dict=surface_properties_dict,mesh_val=mesh_val), mode='rev')
if be == 'python_csdl_backend':
    import python_csdl_backend
    sim = python_csdl_backend.Simulator(UVLMSolver(num_times=nt,h_stepsize=h_stepsize,states_dict=states_dict,
                                        surface_properties_dict=surface_properties_dict,mesh_val=mesh_val), mode='rev')
    
t_start = time.time()
sim.run()
print('simulation time is', time.time() - t_start)
# print('theta',sim['theta'])

######################################################
# make video
######################################################
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
if make_video == 1:
    make_video_vedo(surface_properties_dict,num_nodes,sim)
if plot_cl == 1:
    import matplotlib.pyplot as plt
    plt.plot(t_vec,sim['wing_C_L'],'.-')
    # plt.gca().invert_yaxis()
    # plt.show()
    plt.figure()
    cl = sim['wing_C_L'][-int(num_nodes/n_period)*(n_period-2)-1:-int(num_nodes/n_period)-2]
    cl_ref = np.loadtxt('data.txt')
    plt.plot(np.linspace(0, np.pi*2,cl.shape[0]),cl,'.-')
    plt.plot(np.linspace(0, np.pi*2,cl_ref.shape[0]),cl_ref,'.-')
    plt.legend(['VAST','BYU_UVLM'])
    plt.gca().invert_yaxis()
    # plt.show()

cl_ref = cl_ref.flatten()
cl = cl.flatten()
# print('the error is', np.linalg.norm(cl-cl_ref)/np.linalg.norm(cl_ref)*100,'%')
# sim.compute_totals(of='',wrt='*')
######################################################
# end make video
######################################################

# sim.visualize_implementation()
# partials = sim.check_partials(compact_print=True)
# sim.prob.check_totals(compact_print=True)

# sim.check_totals(compact_print=True)

shape = (num_nodes, nx - 1, ny - 1,)

# using bernoulli equation to compute the pressure, and compare with the result from Kutta-joukowski theorem
# \Nabla p = p_l - p_u = \rho [(Q_t ^ 2 / 2)_u - (Q_t ^ 2 / 2)_l + (\partial \PHI /\partial t)_u - (\partial \PHI /\partial t)_l]
gamma_ij = sim['gamma_b'].reshape(shape)
mesh = sim['wing_bd_vtx_coords']

rho = 1
# delta_c_ij = mesh[:,1:,:,:] - mesh[:,:-1,:,:]
delta_c_ij = 0.5 # hardcode for now as we use uniform mesh
delta_b_ij = 0.5 # hardcode for now as we use uniform mesh
dgamma_di = np.zeros(shape)
dgamma_di[:,0,:] = gamma_ij[:,0,:]
dgamma_di[:,1:,:] = gamma_ij[:,1:,:] - gamma_ij[:,:-1,:]

dgamma_dj = np.zeros(shape)
dgamma_dj[:,:,0] = gamma_ij[:,:,0]
dgamma_dj[:,:,1:] = gamma_ij[:,:,1:] - gamma_ij[:,:,:-1]

# pphi_ptau_i_upper = dgamma_di/(delta_c_ij*2)
# pphi_ptau_i_lower = -dgamma_di/(delta_c_ij*2)

# pphi_ptau_j_upper = dgamma_dj/(delta_b_ij*2)
# pphi_ptau_j_lower = -dgamma_dj/(delta_b_ij*2)

# pphi_ptime_upper = np.zeros((num_nodes, nx-1, ny-1))
# pphi_ptime_upper[1:,:,:] = (gamma_ij[1,:,:] - gamma_ij[0,:,:])/(t_vec[1]*2)
# pphi_ptime_upper[1:,:,:] = (gamma_ij[1:,:,:] - gamma_ij[:-1,:,:])/(t_vec[1]*2)
# pphi_ptime_lower = -pphi_ptime_upper

pphi_ptau_i = dgamma_di/(delta_c_ij)
pphi_ptau_j = dgamma_dj/(delta_b_ij)

pphi_ptime = np.zeros((num_nodes, nx-1, ny-1))
pphi_ptime[0,:,:] = (gamma_ij[1,:,:] - gamma_ij[0,:,:])/(t_vec[1])
pphi_ptime[1:,:,:] = (gamma_ij[1:,:,:] - gamma_ij[:-1,:,:])/(t_vec[1])

# which frame is tau_i, and tau_j in?
tau_i = np.zeros(shape+(3,))
tai_i = (mesh[:,1:,:,:] - mesh[:,:-1,:,:])[:,:,1:,:]

tau_j = np.zeros(shape+(3,))
tau_j = (mesh[:,:,1:,:] - mesh[:,:,:-1,:])[:,1:,:,:]

# delta_p_ij = rho( (first_term . \tau_i) pphi_ptau_i +\
#                   (first_term . \tau_j) pphi_ptau_j +\      
#                    pphi_ptime_upper*2


# first_term = kinematic velocity + induced velocity by the wake panels
wing_kinematic_vel = sim['wing_kinematic_vel'].reshape(shape+(3,))
wake_ind_vel = (np.einsum('ijkl,ik->ijl',sim['aic_M'],sim['gamma_w'].reshape((num_nodes, -1)))).reshape(shape+(3,))
first_term = wing_kinematic_vel + wake_ind_vel
# v_total = sim['wing_eval_total_vel'].reshape(shape+(3,))

delta_p_ij = rho*(np.einsum('ijkl,ijkl->ijk',first_term,tau_i))*pphi_ptau_i +\
                rho*(np.einsum('ijkl,ijkl->ijk',first_term,tau_j))*pphi_ptau_j +\
                rho*pphi_ptime
panel_pressure_all = np.linalg.norm(sim['panel_forces_all']/s_panel,axis=-1)
# s_panel = sim['wing_s_panel'] 
s_panel = 1/6 # hardcode for now as we use uniform mesh
F_sum = -np.sum(delta_p_ij,axis=(1,2)) * s_panel / (0.5 * rho * 1.03**2 *s_panel*24)

F_mag = -np.linalg.norm(sim['F'],axis=1) / (0.5 * rho * 1.03**2 *s_panel*24)
F_mag_s = -np.linalg.norm(sim['F_s'],axis=1) / (0.5 * rho * 1.03**2 *s_panel*24)
plt.figure()
plt.plot(F_sum)
plt.plot(F_mag)
plt.plot(F_mag_s)
plt.gca().invert_yaxis()
plt.show()

from scipy.spatial.transform import Rotation as R

r = R.from_euler('y', np.rad2deg(sim['theta'].flatten()), degrees=True)

Rot_mat = r.as_matrix()

wing_bd_vtx_normals = sim['wing_bd_vtx_normals'].reshape((num_nodes, nx-1, ny-1,3))

rot_normals= np.einsum('ijk,iopk->iopj',Rot_mat, wing_bd_vtx_normals)

F = - np.einsum('ijk,ijkl->ijkl',delta_p_ij,rot_normals)*s_panel


F_total = np.sum(F,axis=(1,2))