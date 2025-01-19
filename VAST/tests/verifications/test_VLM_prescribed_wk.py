import csdl
import numpy as np
from VAST.core.fluid_problem import FluidProblem
from VAST.utils.generate_mesh import *
from VAST.core.vlm_llt.vlm_solver import VLMSolverModel
from python_csdl_backend import Simulator


from VAST.core.vlm_llt.vlm_dynamic_old.VLM_prescribed_wake_solver import UVLMSolver

from VAST.utils.generate_mesh import *
import numpy as np

'''
This example demonstrates the basic VLM simulation 
with a single lifting surface with internal function to generate evaluation pts
Please see vlm_scipt_mls.py for how to use user defined evaluation pts
'''


# self.lifting_surface_dict = dict()
# self.non_lifting_surface_dict = dict()
# self.wake_surface_dict = dict()

##test random
def test_generate_model_vlm_fixed_wake():
    solver_option = 'VLM'
    problem_type = 'prescibed_wake'
    fluid_problem = FluidProblem(solver_option=solver_option, problem_type=problem_type)
    ####################################################################
    # 1. Define VLM inputs that share the common names within CADDEE
    ####################################################################
    # nx = 15; ny = 5
    nx = 7; ny = 3
    chord = 1; span = 4
    num_nodes = 99;  nt = num_nodes
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

    h_stepsize = delta_t = 1 
    if fluid_problem.solver_option == 'VLM' and fluid_problem.problem_type == 'prescibed_wake':
        sim = Simulator(UVLMSolver(num_times=nt,h_stepsize=h_stepsize,states_dict=states_dict,
                                        surface_properties_dict=surface_properties_dict,mesh_val=mesh_val), mode='rev')
    sim.run()
    
    cl = sim['wing_C_L'][75:-1].flatten()
    try:
        cl_ref = np.loadtxt('/Users/jyan/Documents/packages/VAST/tests/verifications/uvlm_plunging.txt').flatten()
    # try:
    #     # cl_ref = np.loadtxt('/home/lsdo/Documents/packages/VAST/tests/verifications/uvlm_plunging.txt').flatten()
    #     import os
    #     print(os.getcwd())
    #     cl_ref = np.loadtxt(os.getcwd()+'/tests/verifications/uvlm_plunging.txt').flatten()
    except:
        cl_ref = np.loadtxt('/home/lsdo/Documents/packages/VAST/tests/verifications/uvlm_plunging.txt').flatten()

    print('cl',cl)
    print('cl_ref',cl_ref)
    print(np.linalg.norm(
            (cl-cl_ref)) / np.linalg.norm(cl_ref))

    np.testing.assert_array_almost_equal(np.linalg.norm(
            (cl-cl_ref)) / np.linalg.norm(cl_ref),
                                                0,
                                                decimal=1)   

    # t_start = time.time()
    return 0

test_generate_model_vlm_fixed_wake()




