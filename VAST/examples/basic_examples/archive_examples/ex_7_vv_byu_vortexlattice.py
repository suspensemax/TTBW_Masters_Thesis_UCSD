'''Example 7 : vnv with BYU VortexLattice'''
import csdl
import numpy as np
from VAST.core.fluid_problem import FluidProblem
from VAST.utils.generate_mesh import *
from VAST.core.submodels.input_submodels.create_input_model import CreateACSatesModel
from VAST.core.vlm_llt.vlm_solver import VLMSolverModel
from python_csdl_backend import Simulator

make_video = 0
plot_cl = 1

########################################
# load mesh from file 
########################################




import cProfile
profiler = cProfile.Profile()

def ex1_generate_model_vlm_fixed_wake(num_nodes,):
    fluid_problem = FluidProblem(solver_option='VLM', problem_type='fixed_wake')

    model_1 = csdl.Model()
    ####################################################################
    # 1. add aircraft states
    ####################################################################
    v_inf = np.ones((num_nodes,1))*1
    theta = np.deg2rad(np.ones((num_nodes,1))*10)  # pitch angles

    submodel = CreateACSatesModel(v_inf=v_inf, theta=theta, num_nodes=num_nodes)
    model_1.add(submodel, 'InputsModule')
    ####################################################################
    # 2. add VLM meshes
    ####################################################################
    # single lifting surface 
    # (nx: number of points in streamwise direction; ny:number of points in spanwise direction)

    x_coords = np.loadtxt('vnv_meshes/byu_vortex_lattice/x.txt')
    y_coords = np.loadtxt('vnv_meshes/byu_vortex_lattice/y.txt')
    z_coords = np.loadtxt('vnv_meshes/byu_vortex_lattice/z.txt')
    mesh = np.stack((x_coords, y_coords, z_coords), axis=-1)

    nx = x_coords.shape[0]; ny = x_coords.shape[1]

    surface_names = ['wing']
    surface_shapes = [(num_nodes, nx, ny, 3)]

    wing = model_1.create_input('wing', val=np.einsum('i,jkl->ijkl', np.ones((num_nodes)), mesh))

    ####################################################################
    # 3. add VAST solver
    ####################################################################
    if fluid_problem.solver_option == 'VLM':
        eval_pts_shapes = [(num_nodes, x[1] - 1, x[2] - 1, 3) for x in surface_shapes]
        submodel = VLMSolverModel(
            surface_names=surface_names,
            surface_shapes=surface_shapes,
            num_nodes=num_nodes,
            eval_pts_shapes=eval_pts_shapes,
            AcStates='dummy',
        )
    # wing_C_L_OAS = np.array([0.4426841725811703]).reshape((num_nodes, 1))
    # wing_C_D_i_OAS = np.array([0.005878842561184834]).reshape((num_nodes, 1))
    model_1.add(submodel, 'VLMSolverModel')
    ####################################################################
    profiler.enable()
    rep = csdl.GraphRepresentation(model_1)
    profiler.disable()
    profiler.dump_stats('output_1')
    sim = Simulator(model_1) # add simulator
    sim.run()
    gamma = sim['gamma_b'].flatten()
    gamma_ref = np.loadtxt('vnv_meshes/byu_vortex_lattice/gamma_10_aoa.txt').flatten()
    print(np.linalg.norm(gamma - gamma_ref)/np.linalg.norm(gamma_ref)*100,'%')
    print((gamma - gamma_ref)/gamma_ref*100,'%')
    print(gamma_ref)
    print()
    print(gamma)
    return sim

# sim = ex1_generate_model_vlm_fixed_wake(num_nodes=1,)




sim.run()

import pyvista as pv

mesh = sim['wing'].reshape(7,13,3)
pv_mesh = pv.StructuredGrid(mesh[:,:,0], mesh[:,:,1], mesh[:,:,2])
pv_mesh.plot(show_edges=True, color='w', line_width=1, show_scalar_bar=False, background='k',)


# # print('the error is', np.linalg.norm(cl-cl_ref)/np.linalg.norm(cl_ref)*100,'%')
# # sim.compute_totals(of='',wrt='*')
# ######################################################
# # end make video
# ######################################################

# # sim.visualize_implementation()
# # partials = sim.check_partials(compact_print=True)
# # sim.prob.check_totals(compact_print=True)

# # sim.check_totals(compact_print=True)

# shape = (num_nodes, nx - 1, ny - 1,)

# # using bernoulli equation to compute the pressure, and compare with the result from Kutta-joukowski theorem
# # \Nabla p = p_l - p_u = \rho [(Q_t ^ 2 / 2)_u - (Q_t ^ 2 / 2)_l + (\partial \PHI /\partial t)_u - (\partial \PHI /\partial t)_l]
# gamma_ij = sim['gamma_b'].reshape(shape)
# mesh = sim['wing_bd_vtx_coords']

# rho = 1
# delta_c_ij = (mesh[:,1:,:,:] - mesh[:,:-1,:,:])[0][0,0,0]
# # delta_c_ij = 0.5 # hardcode for now as we use uniform mesh
# delta_b_ij = (mesh[:,:,1,:] - mesh[:,:,0,:])[0][0,1]
# dgamma_di = np.zeros(shape)
# dgamma_di[:,0,:] = gamma_ij[:,0,:]
# dgamma_di[:,1:,:] = gamma_ij[:,1:,:] - gamma_ij[:,:-1,:]

# dgamma_dj = np.zeros(shape)
# dgamma_dj[:,:,0] = gamma_ij[:,:,0]
# dgamma_dj[:,:,1:] = gamma_ij[:,:,1:] - gamma_ij[:,:,:-1]

# # pphi_ptau_i_upper = dgamma_di/(delta_c_ij*2)
# # pphi_ptau_i_lower = -dgamma_di/(delta_c_ij*2)

# # pphi_ptau_j_upper = dgamma_dj/(delta_b_ij*2)
# # pphi_ptau_j_lower = -dgamma_dj/(delta_b_ij*2)

# # pphi_ptime_upper = np.zeros((num_nodes, nx-1, ny-1))
# # pphi_ptime_upper[1:,:,:] = (gamma_ij[1,:,:] - gamma_ij[0,:,:])/(t_vec[1]*2)
# # pphi_ptime_upper[1:,:,:] = (gamma_ij[1:,:,:] - gamma_ij[:-1,:,:])/(t_vec[1]*2)
# # pphi_ptime_lower = -pphi_ptime_upper
# wing_kinematic_vel = sim['wing_kinematic_vel'].reshape(shape+(3,))

# pphi_ptau_i = dgamma_di/(delta_c_ij)
# pphi_ptau_j = dgamma_dj/(delta_b_ij)

# upper_surface_velocity = np.zeros((num_nodes, nx-1, ny-1, 3))
# upper_surface_velocity[:,:,:,0] = pphi_ptau_i/2 + wing_kinematic_vel[:,:,:,0]
# upper_surface_velocity[:,:,:,1] = pphi_ptau_j/2 + wing_kinematic_vel[:,:,:,1]
# upper_surface_velocity[:,:,:,2] = wing_kinematic_vel[:,:,:,2]

# lower_surface_velocity = np.zeros((num_nodes, nx-1, ny-1, 3))
# lower_surface_velocity[:,:,:,0] = -pphi_ptau_i/2 + wing_kinematic_vel[:,:,:,0]
# lower_surface_velocity[:,:,:,1] = -pphi_ptau_j/2 + wing_kinematic_vel[:,:,:,1]
# lower_surface_velocity[:,:,:,2] = wing_kinematic_vel[:,:,:,2]

# pphi_ptime = np.zeros((num_nodes, nx-1, ny-1))
# pphi_ptime[0,:,:] = (gamma_ij[1,:,:] - gamma_ij[0,:,:])/(t_vec[1])
# pphi_ptime[1:,:,:] = (gamma_ij[1:,:,:] - gamma_ij[:-1,:,:])/(t_vec[1])

# upper_surface_velocity_norm_sq = np.linalg.norm(upper_surface_velocity,axis=-1)**2
# lower_surface_velocity_norm_sq = np.linalg.norm(lower_surface_velocity,axis=-1)**2
# delta_p = rho*(upper_surface_velocity_norm_sq/2 - lower_surface_velocity_norm_sq/2 + pphi_ptime)
# panel_forces = np.sum(delta_p*delta_c_ij*delta_b_ij,axis=(1,2))/(0.5*1.01362278*1.01362278**2/(delta_c_ij*delta_b_ij))

# delta_p_static = rho*(upper_surface_velocity_norm_sq/2 - lower_surface_velocity_norm_sq/2)
# panel_forces_static = np.sum(delta_p_static*delta_c_ij*delta_b_ij,axis=(1,2))/(0.5*1.01362278*1.01362278**2/(delta_c_ij*delta_b_ij))


# plt.plot(panel_forces_static)
# plt.plot(panel_forces-panel_forces_static)
# plt.plot(panel_forces)
# # plt.plot(sim['wing_C_L'])
# plt.show()

# u_val = (np.ones(num_nodes) * np.cos(alpha)).reshape((num_nodes,1))
# w_vel = np.ones((num_nodes,1)) * np.sin(alpha) - h * np.cos(omg*t_vec).reshape((num_nodes,1))
# w_vel_dot = h * omg * np.sin(omg*t_vec).reshape((num_nodes,1))
# plt.plot(panel_forces-panel_forces_static,'.-')
# plt.plot(-w_vel_dot,'.-')
# plt.show()

# # # which frame is tau_i, and tau_j in?
# # tau_i = np.zeros(shape+(3,))
# # tai_i = (mesh[:,1:,:,:] - mesh[:,:-1,:,:])[:,:,1:,:]

# # tau_j = np.zeros(shape+(3,))
# # tau_j = (mesh[:,:,1:,:] - mesh[:,:,:-1,:])[:,1:,:,:]

# # # delta_p_ij = rho( (first_term . \tau_i) pphi_ptau_i +\
# # #                   (first_term . \tau_j) pphi_ptau_j +\      
# # #                    pphi_ptime_upper*2


# # # first_term = kinematic velocity + induced velocity by the wake panels
# # wake_ind_vel = (np.einsum('ijkl,ik->ijl',sim['aic_M'],sim['gamma_w'].reshape((num_nodes, -1)))).reshape(shape+(3,))
# # first_term = wing_kinematic_vel + wake_ind_vel
# # # v_total = sim['wing_eval_total_vel'].reshape(shape+(3,))

# # delta_p_ij = rho*(np.einsum('ijkl,ijkl->ijk',first_term,tau_i))*pphi_ptau_i +\
# #                 rho*(np.einsum('ijkl,ijkl->ijk',first_term,tau_j))*pphi_ptau_j +\
# #                 rho*pphi_ptime
# # panel_pressure_all = np.linalg.norm(sim['panel_forces_all']/s_panel,axis=-1)
# # # s_panel = sim['wing_s_panel'] 
# # s_panel = 1/6 # hardcode for now as we use uniform mesh
# # F_sum = -np.sum(delta_p_ij,axis=(1,2)) * s_panel / (0.5 * rho * 1.03**2 *s_panel*24)

# # F_mag = -np.linalg.norm(sim['F'],axis=1) / (0.5 * rho * 1.03**2 *s_panel*24)
# # F_mag_s = -np.linalg.norm(sim['F_s'],axis=1) / (0.5 * rho * 1.03**2 *s_panel*24)
# # plt.figure()
# # plt.plot(F_sum)
# # plt.plot(F_mag)
# # plt.plot(F_mag_s)
# # plt.gca().invert_yaxis()
# # plt.show()

# # from scipy.spatial.transform import Rotation as R

# # r = R.from_euler('y', np.rad2deg(sim['theta'].flatten()), degrees=True)

# # Rot_mat = r.as_matrix()

# # wing_bd_vtx_normals = sim['wing_bd_vtx_normals'].reshape((num_nodes, nx-1, ny-1,3))

# # rot_normals= np.einsum('ijk,iopk->iopj',Rot_mat, wing_bd_vtx_normals)

# # F = - np.einsum('ijk,ijkl->ijkl',delta_p_ij,rot_normals)*s_panel


# # F_total = np.sum(F,axis=(1,2))