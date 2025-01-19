'''Example 7 : vnv with BYU VortexLattice'''
import csdl
import numpy as np
from VAST.core.fluid_problem import FluidProblem
from VAST.utils.generate_mesh import *
from VAST.core.submodels.input_submodels.create_input_model import CreateACSatesModel
from VAST.core.vlm_llt.vlm_solver import VLMSolverModel
from python_csdl_backend import Simulator
import pytest

make_video = 0
plot_cl = 1

########################################
# load mesh from file 
########################################


def test_generate_model_vlm_fixed_wake():
    fluid_problem = FluidProblem(solver_option='VLM', problem_type='fixed_wake')

    model_1 = csdl.Model()
    num_nodes = 1
    ####################################################################
    # 1. add aircraft states
    ####################################################################
    v_inf = np.ones((num_nodes,1))*1
    theta = np.deg2rad(np.ones((num_nodes,1))*10)  # pitch angles

    submodel = CreateACSatesModel(v_inf=v_inf, theta=theta, num_nodes=num_nodes)
    model_1.add(submodel, 'InputsModel')
    ####################################################################
    # 2. add VLM meshes
    ####################################################################
    # single lifting surface 
    # (nx: number of points in streamwise direction; ny:number of points in spanwise direction)

    x_coords = np.loadtxt('/home/lsdo/Documents/packages/VAST/tests/verifications/byu_vortex_lattice/x.txt')-1.1
    y_coords = np.loadtxt('/home/lsdo/Documents/packages/VAST/tests/verifications/byu_vortex_lattice/y.txt')
    z_coords = np.loadtxt('/home/lsdo/Documents/packages/VAST/tests/verifications/byu_vortex_lattice/z.txt')
    mesh = np.stack((x_coords, y_coords, z_coords), axis=-1)
    print('mesh shape: ', mesh.shape)
    print(x_coords)

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

    sim = Simulator(model_1) # add simulator
    sim.run()
    
    gamma = sim['gamma_b'].flatten()
    gamma_ref = np.loadtxt('/home/lsdo/Documents/packages/VAST/tests/verifications/byu_vortex_lattice/gamma_10_aoa.txt').flatten()

    wing_C_L_ref = 0.6305970178500268
    wing_C_D_i_ref = 0.031877414928509776

    # decimal = 1 means rel error under 10 percent

    # np.testing.assert_array_almost_equal(np.linalg.norm(
    #         (gamma - gamma_ref)) / np.linalg.norm(gamma_ref),
    #                                             0,
    #                                             decimal=1)

    print('wing_C_L', sim['wing_C_L'])
    print('wing_C_D_i', sim['wing_C_D_i'])
    print(x_coords.shape)

    np.testing.assert_array_almost_equal(np.abs(np.linalg.norm(
            (wing_C_L_ref - sim['wing_C_L'])) / np.linalg.norm(sim['wing_C_L'])),
                                                0,
                                                decimal=1)
    # np.testing.assert_array_almost_equal(np.abs(np.linalg.norm(
    #         (wing_C_D_i_ref - sim['wing_C_D_i'])) / np.linalg.norm(sim['wing_C_D_i'])),
    #                                             0,
    #                                             decimal=1)


    return sim

sim = test_generate_model_vlm_fixed_wake()
