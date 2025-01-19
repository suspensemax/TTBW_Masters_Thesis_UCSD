import csdl
import numpy as np
from VAST.core.fluid_problem import FluidProblem
from VAST.utils.generate_mesh import *
from VAST.core.vlm_llt.vlm_solver import VLMSolverModel
from python_csdl_backend import Simulator
import pytest


'''
This example demonstrates the basic VLM simulation 
with a single lifting surface with internal function to generate evaluation pts
Please see vlm_scipt_mls.py for how to use user defined evaluation pts
'''


# self.lifting_surface_dict = dict()
# self.non_lifting_surface_dict = dict()
# self.wake_surface_dict = dict()

##test aero_tests_1
def test_generate_model_vlm_fixed_wake():
    solver_option = 'VLM'
    problem_type = 'fixed_wake'
    fluid_problem = FluidProblem(solver_option=solver_option, problem_type=problem_type)
    ####################################################################
    # 1. Define VLM inputs that share the common names within CADDEE
    ####################################################################
    num_nodes = 1
    create_opt = 'create_inputs'
    model_1 = csdl.Model()
    alpha = np.deg2rad(np.ones((num_nodes,1))*5)

    vx = 248.136
    vz = 0

    u = model_1.create_input('u',val=np.ones((num_nodes,1))*vx)
    v = model_1.create_input('v',val=np.zeros((num_nodes, 1)))
    w = model_1.create_input('w',val=np.ones((num_nodes,1))*vz)
    p = model_1.create_input('p',val=np.zeros((num_nodes, 1)))
    q = model_1.create_input('q',val=np.zeros((num_nodes, 1)))
    r = model_1.create_input('r',val=np.zeros((num_nodes, 1)))
    phi = model_1.create_input('phi',val=np.zeros((num_nodes, 1)))
    theta = model_1.create_input('theta',val=alpha)
    psi = model_1.create_input('psi',val=np.zeros((num_nodes, 1)))
    x = model_1.create_input('x',val=np.zeros((num_nodes, 1)))
    y = model_1.create_input('y',val=np.zeros((num_nodes, 1)))
    z = model_1.create_input('z',val=np.ones((num_nodes, 1))*1000)
    phiw = model_1.create_input('phiw',val=np.zeros((num_nodes, 1)))
    gamma = model_1.create_input('gamma',val=np.zeros((num_nodes, 1)))
    psiw = model_1.create_input('psiw',val=np.zeros((num_nodes, 1)))

    ####################################################################
    # 2. add VLM meshes
    ####################################################################
    # single lifting surface
    nx = 3  # number of points in streamwise direction
    ny = 11  # number of points in spanwise direction

    surface_names = ['wing']
    surface_shapes = [(num_nodes, nx, ny, 3)]

    # chord = 1.49352
    # span = 16.2 / chord

    mesh_dict = {
        "num_y": ny,
        "num_x": nx,
        "wing_type": "rect",
        "symmetry": False,
        "span": 10.0,
        "chord": 1,
        "span_cos_sppacing": 1.0,
        "chord_cos_sacing": 1.0,
    }

    # Generate mesh of a rectangular wing
    mesh = generate_mesh(mesh_dict) #(nx,ny,3)
    offset = 0

    mesh_val = np.zeros((num_nodes, nx, ny, 3))

    for i in range(num_nodes):
        mesh_val[i, :, :, :] = mesh
        mesh_val[i, :, :, 0] = mesh.copy()[:, :, 0]
        mesh_val[i, :, :, 1] = mesh.copy()[:, :, 1] + offset

    wing = model_1.create_input('wing', val=mesh_val)
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
    wing_C_L_OAS = np.array([0.4426841725811703]).reshape((num_nodes, 1))
    wing_C_D_i_OAS = np.array([0.005878842561184834]).reshape((num_nodes, 1))

    model_1.add(submodel, 'VLMSolverModel')
    sim = Simulator(model_1)
    sim.run()
    np.testing.assert_array_almost_equal(np.linalg.norm(
            (wing_C_L_OAS - sim['wing_C_L'])) / np.linalg.norm(sim['wing_C_L']),
                                                0,
                                                decimal=2)
    np.testing.assert_array_almost_equal(np.linalg.norm(
            (wing_C_D_i_OAS - sim['wing_C_D_i'])) / np.linalg.norm(sim['wing_C_D_i']),
                                                0,
                                                decimal=2)
    return sim

# sim = test_generate_model_vlm_fixed_wake(fluid_problem=fluid_problem)




