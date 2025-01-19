import csdl
from VAST.core.submodels.input_submodels.create_input_model import CreateACSatesModel
from VAST.core.submodels.input_submodels.create_input_module import CreateACSatesModule
from VAST.core.vlm_llt.vlm_solver import VLMSolverModel

from VAST.core.fluid_problem import FluidProblem
import m3l
from VAST.core.vlm_llt.NodalMapping import NodalMap,RadialBasisFunctions
from VAST.core.generate_mappings_m3l import VASTNodelDisplacements,VASTNodalForces
# from generate_mappings_m3l import VASTNodalForces



import numpy as np
from VAST.utils.generate_mesh import *
from python_csdl_backend import Simulator

# def test_disp_map():
#     fluid_problem = FluidProblem(solver_option='VLM', problem_type='fixed_wake')

#     num_nodes=1; nx=3; ny=11

#     v_inf = np.ones((num_nodes,1))*248.136
#     theta = np.deg2rad(np.ones((num_nodes,1))*5)  # pitch angles


#     surface_names = ['wing']
#     surface_shapes = [(num_nodes, nx, ny, 3)]
#     mesh_dict = {
#         "num_y": ny, "num_x": nx, "wing_type": "rect", "symmetry": False, "span": 10.0,
#         "chord": 1, "span_cos_sppacing": 1.0, "chord_cos_sacing": 1.0,
#     }
#     # Generate mesh of a rectangular wing
#     mesh = generate_mesh(mesh_dict)

#     ###########################################
#     # 1. Create a dummy m3l.Model()
#     ###########################################
#     dummy_model = m3l.Model()

#     # fluid_model.set_module_input('wing_undef_mesh', val=np.einsum('i,jkl->ijkl', np.ones((num_nodes)), mesh))

#     input_dicts = {}
#     input_dicts['v_inf'] = v_inf
#     input_dicts['theta'] = theta
#     input_dicts['undef_mesh'] = [np.einsum('i,jkl->ijkl', np.ones((num_nodes)), mesh)]
#     input_dicts['displacements'] = [np.zeros((num_nodes, nx, ny, 3))]


#     airfoil = np.loadtxt(fname='/home/lsdo/Documents/packages/VAST/VAST/core/vlm_llt/naca0012.txt')
#     z_val = np.linspace(-5, 5, 13)
#     oml_mesh = np.zeros((21, 13, 3))
#     oml_mesh[:, :, 0] = np.outer(airfoil[:, 0],np.ones(13))-0.5
#     oml_mesh[:, :, 1] = np.outer(np.ones(21),z_val)
#     oml_mesh[:, :, 2] = np.outer(airfoil[:, 1],np.ones(13))

#     ###########################################
#     # 2. Create fluid_model as VASTFluidSover 
#     # (msl.explicit operation)
#     ###########################################
#     fluid_model = VASTNodelDisplacements(
#                                     surface_names=surface_names,
#                                     surface_shapes=surface_shapes,
#                                     initial_meshes=[input_dicts['undef_mesh'][0].reshape(-1,3)],)


#     displacements = []
#     for i in range(len(surface_names)):
#         surface_name = surface_names[i]
#         surface_shape = oml_mesh.shape
#         displacement = m3l.Variable(f'{surface_name}_nodal_displacements',shape=surface_shape,value=np.zeros(oml_mesh.shape))
#         fluid_model.set_module_input(f'{surface_name}_nodal_displacements', val=np.zeros(oml_mesh.shape))
#         displacements.append(displacement)

#     ###########################################
#     # 4. call fluid_model.evaluate to get
#     # surface panel forces
#     ###########################################
#     vlm_displacements = fluid_model.evaluate(displacements,[oml_mesh.reshape(-1,3)])

#     ###########################################
#     # 5. register outputs to dummy_model
#     ###########################################
#     for i in range(len(surface_names)):
#         surface_name = surface_names[i]
#         dummy_model.register_output(vlm_displacements[i])

#     ###########################################
#     # 6. call _assemble_csdl to get dummy_model_csdl
#     ###########################################
#     dummy_model_csdl = dummy_model.assemble_csdl()
#     ###########################################
#     # 7. use sim.run to run the csdl model
#     ###########################################    
#     # create a random displacement on the oml mesh
#     disp_temp = np.linspace(0.6, 0,7)
#     disp_z = np.outer(np.ones(21),np.concatenate((disp_temp, disp_temp[:-1][::-1])))
#     disp = np.zeros((1, 21, 13, 3))
#     disp[0, :, :, 2] = disp_z
#     oml_disp = dummy_model_csdl.create_input('wing_nodal_displacements', val=disp)

#     sim = Simulator(dummy_model_csdl,analytics=False) # add simulator
#     sim.run()

def test_force_map():
    fluid_problem = FluidProblem(solver_option='VLM', problem_type='fixed_wake')

    num_nodes=1; nx=3; ny=11

    v_inf = np.ones((num_nodes,1))*248.136
    theta = np.deg2rad(np.ones((num_nodes,1))*5)  # pitch angles


    surface_names = ['wing']
    surface_shapes = [(num_nodes, nx, ny, 3)]
    mesh_dict = {
        "num_y": ny, "num_x": nx, "wing_type": "rect", "symmetry": False, "span": 10.0,
        "chord": 1, "span_cos_sppacing": 1.0, "chord_cos_sacing": 1.0,
    }
    # Generate mesh of a rectangular wing
    mesh = generate_mesh(mesh_dict)

    ###########################################
    # 1. Create a dummy m3l.Model()
    ###########################################
    dummy_model = m3l.Model()

    # fluid_model.set_module_input('wing_undef_mesh', val=np.einsum('i,jkl->ijkl', np.ones((num_nodes)), mesh))

    input_dicts = {}
    input_dicts['v_inf'] = v_inf
    input_dicts['theta'] = theta
    input_dicts['undef_mesh'] = [np.einsum('i,jkl->ijkl', np.ones((num_nodes)), mesh)]
    input_dicts['displacements'] = [np.zeros((num_nodes, nx, ny, 3))]


    airfoil = np.loadtxt(fname='/home/lsdo/Documents/packages/VAST/VAST/core/vlm_llt/naca0012.txt')
    z_val = np.linspace(-5, 5, 13)
    oml_mesh = np.zeros((21, 13, 3))
    oml_mesh[:, :, 0] = np.outer(airfoil[:, 0],np.ones(13))-0.5
    oml_mesh[:, :, 1] = np.outer(np.ones(21),z_val)
    oml_mesh[:, :, 2] = np.outer(airfoil[:, 1],np.ones(13))

    ###########################################
    # 2. Create fluid_model as VASTFluidSover 
    # (msl.explicit operation)
    ###########################################
    fluid_model = VASTNodalForces(  surface_names=surface_names,
                                    surface_shapes=surface_shapes,
                                    initial_meshes=[input_dicts['undef_mesh'][0].reshape(-1,3)],)


    vlm_forces = []
    for i in range(len(surface_names)):
        surface_name = surface_names[i]
        surface_shape = surface_shapes[i]
        vlm_forces_surface = m3l.Variable(f'{surface_name}_total_forces',shape=surface_shape,value=np.zeros(surface_shape))
        fluid_model.set_module_input(f'{surface_name}_total_forces', val=np.zeros(surface_shape))
        vlm_forces.append(vlm_forces_surface)

    ###########################################
    # 4. call fluid_model.evaluate to get
    # surface panel forces
    ###########################################
    oml_forces = fluid_model.evaluate(vlm_forces,[oml_mesh.reshape(-1,3)])

    ###########################################
    # 5. register outputs to dummy_model
    ###########################################
    for i in range(len(surface_names)):
        # surface_name = surface_names[i]
        dummy_model.register_output(oml_forces[i])

    ###########################################
    # 6. call _assemble_csdl to get dummy_model_csdl
    ###########################################
    dummy_model_csdl = dummy_model.assemble_csdl()
    ###########################################
    # 7. use sim.run to run the csdl model
    ###########################################    

    vlm_dummy_forces = np.random.random((1,20,3))
    vlm_forces_surface = dummy_model_csdl.create_input('wing_total_forces', val=vlm_dummy_forces)
    sim = Simulator(dummy_model_csdl,analytics=False) # add simulator
    sim.run()

    oml_mesh_flatten = oml_mesh.reshape(-1,3)
    vlm_dummy_forces[0,0] = 1000
    vlm_f = vlm_dummy_forces.reshape(-1,3)
    map = np.loadtxt('weights.txt')

    np.sum(map,axis=0)
    np.sum(map,axis=1)

    oml_f_x = vlm_f[:,0]@map
    oml_f_y = vlm_f[:,1]@map
    oml_f_z = vlm_f[:,2]@map
    error_x = np.sum(oml_f_x) - np.sum(vlm_f[:,0])
    error_y = np.sum(oml_f_y) - np.sum(vlm_f[:,1])
    error_z = np.sum(oml_f_z) - np.sum(vlm_f[:,2])
    return error_x, error_y, error_z


