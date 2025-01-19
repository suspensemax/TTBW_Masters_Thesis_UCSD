import csdl
from VAST.core.vlm_llt.vlm_solver import VLMSolverModel

from VAST.core.fluid_problem import FluidProblem
import m3l
from typing import List, Union
import numpy as np
from dataclasses import dataclass, field

def check_same_length(*lists):
    return all(len(lst) == len(lists[0]) for lst in lists)

def mirror_mesh_across_xz(original_mesh, mirror_mesh, num_spanwise):
    """
    Mirror y-coordinates of vlm mesh
    
    Parameters
    ----------
    original_mesh : csdl Variable 
        The original mesh

    mirrored_mesh : csdl Variable (create_output)
        The mirror mesh

    num_spanwise : int
        number of spanwise 
    """
    if num_spanwise %2 == 0:
        mirror_mesh[:, :, 0:int(num_spanwise/2), 0] = original_mesh[:, :, 0:int(num_spanwise/2), 0] * 1
        mirror_mesh[:, :, 0:int(num_spanwise/2), 1] = original_mesh[:, :, 0:int(num_spanwise/2), 1] * 1
        mirror_mesh[:, :, 0:int(num_spanwise/2), 2] = original_mesh[:, :, 0:int(num_spanwise/2), 2] * 1

        for j in range(int(num_spanwise/2)):
            index = int(num_spanwise/2 + j)
            symmetric_index = int(num_spanwise/2- j -1)
            mirror_mesh[:, :, index, 0] = original_mesh[:, :, symmetric_index, 0]
            mirror_mesh[:, :, index, 1] = -1* original_mesh[:, :, symmetric_index, 1]
            mirror_mesh[:, :, index, 2] = original_mesh[:, :, symmetric_index, 2]
    else:
        mirror_mesh[:, :, 0:int(num_spanwise/2), 0] = original_mesh[:, :, 0:int(num_spanwise/2), 0] * 1
        mirror_mesh[:, :, 0:int(num_spanwise/2), 1] = original_mesh[:, :, 0:int(num_spanwise/2), 1] * 1
        mirror_mesh[:, :, 0:int(num_spanwise/2), 2] = original_mesh[:, :, 0:int(num_spanwise/2), 2] * 1
        mirror_mesh[:, :, int(num_spanwise/2), :] = original_mesh[:, :, int(num_spanwise/2), :]
        
        for j in range(int(num_spanwise/2)):
            index = int(num_spanwise/2 + j + 1)
            symmetric_index = int(num_spanwise/2- j- 1)
            mirror_mesh[:, :, index, 0] = original_mesh[:, :, symmetric_index, 0]
            mirror_mesh[:, :, index, 1] = -1* original_mesh[:, :, symmetric_index, 1]
            mirror_mesh[:, :, index, 2] = original_mesh[:, :, symmetric_index, 2]

def zero_out_mesh_y(original_mesh, new_mesh):
    print(original_mesh.shape)
    print(new_mesh.shape)
    new_mesh[:, :, :, 0] = original_mesh[:, :, :, 0]
    new_mesh[:, :, :, 2] = original_mesh[:, :, :, 2]

def average_z(original_mesh, new_mesh):
    new_mesh[:, :, :, 0] = original_mesh[:, :, :, 0]
    new_mesh[:, :, :, 1] = original_mesh[:, :, :, 1]
    new_mesh[:, :, :, 2] = csdl.expand(csdl.average(original_mesh[:, :, :, 2]), shape=original_mesh.shape[0:-1] + (1, ))

def deflect_flap(original_mesh : csdl.Variable, new_mesh : csdl.Variable, rotation_mat : csdl.Variable, num_chordwise : int(1)):
    shape = original_mesh.shape
    
    new_mesh[:, 0:-num_chordwise, :, :] =  original_mesh[:, 0:-num_chordwise, :, :]
    num_nodes = shape[0]
    num_chordwise_panels = shape[1]
    nx_list = np.arange(-num_chordwise, 0, 1).tolist()
    num_spanwise = shape[2]

    if num_nodes > 1:
        rotation_mat = csdl.reshape(rotation_mat[0, :, :], (3, 3))

    for i in range(num_nodes):
        for nx in nx_list:
            for ny in range(num_spanwise):
                rot_vec = original_mesh[i, nx, ny, :]
                index = (num_chordwise + 1) + nx
                origin_vec = original_mesh[i, nx-index, ny, :]

                rotation = csdl.matvec(rotation_mat, csdl.reshape(rot_vec - origin_vec, (3, ))) 
                new_mesh[i, int(num_chordwise_panels + nx), ny, :] = csdl.expand(rotation, (1, 1, 1, 3), indices='l->ijkl') + origin_vec

    return new_mesh

class StabilityAdapterModelCSDL(csdl.Model):
    def initialize(self):
        self.parameters.declare('arguments', types=dict)
        self.parameters.declare('num_nodes', types=int)
        self.parameters.declare('neglect', types=list, default=[])
        self.parameters.declare('stability_flag', types=bool)
        self.parameters.declare('meshes', types=list)
        self.parameters.declare('surface_names', types=list)
        self.parameters.declare('deflections', types=list)
    
    def define(self):
        args = self.parameters['arguments']
        num_nodes = self.parameters['num_nodes']
        ac_states = ['u', 'v', 'w', 'p', 'q', 'r', 'theta', 'phi', 'psi', 'x', 'y', 'z']
        special_cases = self.parameters['neglect']
        stability_flag = self.parameters['stability_flag']
        meshes = self.parameters['meshes']
        surface_names = self.parameters['surface_names']
        deflections = self.parameters['deflections']

        if stability_flag:
            rotation_mat = self.create_output('rotation_mat', val=0, shape=(13, 3, 3))
            attitude_angles = {}
            for key, value in args.items():
                # Check if key is an aircraft state
                if key in ac_states:
                    csdl_var = self.declare_variable(key, shape=(num_nodes * 13, ))
                    self.register_output(name=f'{key}_exp', var=csdl_var * 1)
                    if key in ['phi', 'theta', 'psi']:
                        attitude_angles[key] = csdl_var
            
            phi = np.zeros((13, ))
            theta = np.zeros((13, ))
            psi = np.zeros((13, ))
            phi[7] = np.deg2rad(0.5)
            theta[8] = np.deg2rad(0.5)
            psi[9] = np.deg2rad(0.5)

            phi_csdl = self.create_input('phi_perturbed', val=phi)
            theta_csdl = self.create_input('theta_perturbed', val=theta)
            psi_csdl = self.create_input('psi_perturbed', val=psi)

            rotation_mat[:, 0, 0] = csdl.reshape(csdl.cos(psi_csdl) * csdl.cos(theta_csdl), (13, 1, 1))
            rotation_mat[:, 0, 1] = csdl.reshape(csdl.cos(psi_csdl) * csdl.sin(theta_csdl) * csdl.sin(phi_csdl) - csdl.sin(psi_csdl) * csdl.cos(phi_csdl) , (13, 1, 1))
            rotation_mat[:, 0, 2] = csdl.reshape(csdl.cos(psi_csdl) * csdl.sin(theta_csdl) * csdl.cos(phi_csdl) + csdl.sin(psi_csdl) * csdl.sin(phi_csdl), (13, 1, 1))
            rotation_mat[:, 1, 0] = csdl.reshape(csdl.sin(psi_csdl) * csdl.cos(theta_csdl), (13, 1, 1))
            rotation_mat[:, 1, 1] = csdl.reshape(csdl.sin(psi_csdl) * csdl.sin(theta_csdl) * csdl.sin(phi_csdl) +  csdl.cos(psi_csdl) * csdl.cos(phi_csdl), (13, 1, 1))
            rotation_mat[:, 1, 2] = csdl.reshape(csdl.sin(psi_csdl) * csdl.sin(theta_csdl) * csdl.cos(phi_csdl) - csdl.cos(psi_csdl) * csdl.sin(phi_csdl), (13, 1, 1))
            rotation_mat[:, 2, 0] = csdl.reshape(-1 * csdl.sin(theta_csdl), (13, 1, 1))
            rotation_mat[:, 2, 1] = csdl.reshape(csdl.cos(theta_csdl) * csdl.sin(phi_csdl), (13, 1, 1))
            rotation_mat[:, 2, 2] = csdl.reshape(csdl.cos(theta_csdl) * csdl.cos(phi_csdl), (13, 1, 1))

            for key, value in args.items():
                if key in ac_states:
                    pass
                
                # Check whether key is a special case (meaning we don't expand the variable)
                elif key in special_cases:
                    csdl_var = self.declare_variable(key, shape=value.shape)
                    self.register_output(name=f'{key}_exp', var=csdl_var * 1)
                
                # Check whether key is a mesh variable
                elif key in surface_names:
                    csdl_var = self.declare_variable(key, shape=value.shape)
                    mesh = meshes[surface_names.index(key)]
                    deflection = deflections[surface_names.index(key)]
                    desc = mesh.description
                    
                    # Check the length of the shape tuple (3 or 4 should be possible)
                    if len(value.shape) == 4:
                        if value.shape[0] == 1:
                            csdl_var_reshaped = csdl.reshape(csdl_var, new_shape=value.shape[1:])
                            csdl_var_exp_initial = csdl.expand(csdl_var_reshaped, shape=(13, ) + value.shape[1:], indices='ijk->hijk')
                            if deflection is not None:
                                deflection_angle = csdl.expand(self.declare_variable(f'{key}_flap_deflection', shape=(1, )), shape=(13))
                                flap_rot_mat = self.create_output(f'{key}_flap_rot_mat', shape=(13, 3, 3), val=0)
                                flap_rot_mat[:, 0, 0] = csdl.reshape(csdl.cos(deflection_angle), new_shape=(13, 1, 1))
                                flap_rot_mat[:, 0, 2] = csdl.reshape(csdl.sin(deflection_angle) * -1, new_shape=(13, 1, 1))
                                flap_rot_mat[:, 1, 1] = self.create_input(f'flap_deflection_one_entry_{key}', val=1, shape=(13, 1, 1))
                                flap_rot_mat[:, 2, 0] = csdl.reshape(csdl.sin(deflection_angle), new_shape=(13, 1, 1))
                                flap_rot_mat[:, 2, 2] = csdl.reshape(csdl.cos(deflection_angle), new_shape=(13, 1, 1))
                                flap_rot_mesh = self.create_output(f"{key}_flap_rotation_mesh", shape=csdl_var_exp_initial.shape, val=0)
                                csdl_var_exp = deflect_flap(csdl_var_exp_initial, flap_rot_mesh, flap_rot_mat, 1)
                            else:
                                csdl_var_exp = csdl_var_exp_initial * 1


                            csdl_var_exp_rotated = self.create_output(f"{key}_mesh_rotated", shape=csdl_var_exp.shape)
                            num_chordwise = value.shape[1]
                            num_spanwise = value.shape[2]

                            for i in range(13):
                                if i == 0:
                                    csdl_var_exp_rotated[i, :, :, :] = csdl_var_exp[i, :, :, :]
                                else:
                                    for j in range(num_chordwise):
                                        for k in range(num_spanwise):
                                            csdl_var_exp_rotated[i, j, k, :] = csdl.reshape(csdl.matvec(csdl.reshape(rotation_mat[i, :, :], new_shape=(3, 3)), csdl.reshape(csdl_var_exp[i, j, k, :], new_shape=(3, ))), (1, 1, 1, 3))

                            if desc is not None:
                                if desc not in ['mirror', 'zero_y', 'average_z']:
                                    raise ValueError(f"description for VLM mesh variables can only be 'mirror' or 'zero_y'. Received '{mesh.description}'")
                                elif desc == 'mirror':
                                    mirrored_mesh = self.create_output(f'{key}_exp', shape=(13, ) + value.shape[1:], val=0)
                                    mirror_mesh_across_xz(original_mesh=csdl_var_exp_rotated, mirror_mesh=mirrored_mesh, num_spanwise=num_spanwise)               
                                elif desc == 'zero_y':
                                    new_mesh = self.create_output(f'{key}_exp', shape=(13, ) + value.shape[1:], val=0)
                                    zero_out_mesh_y(original_mesh=csdl_var_exp_rotated, new_mesh=new_mesh)
                                elif desc == 'average_z':
                                    new_mesh = self.create_output(f'{key}_exp', shape=(13, ) + value.shape, val=0)
                                    average_z(original_mesh=csdl_var_exp_rotated, new_mesh=new_mesh)
                                else:
                                    raise NotImplementedError
                            else:
                                self.register_output(name=f'{key}_exp', var=csdl_var_exp_rotated* 1)
                        else:
                            print(key, value.shape)
                            raise NotImplementedError
                    
                    elif len(value.shape) == 3:
                        if value.shape[0] == 1 or value.shape[1] == 1:
                            raise ValueError("VLM meshes cannot have num_spanwise, num_chordwise<2 ")
                        else:

                            csdl_var_exp = csdl.expand(csdl_var, shape=(13, ) + value.shape, indices='ijk->hijk')
                            csdl_var_exp_rotated = self.create_output(f"{key}_mesh_rotated", shape=csdl_var_exp.shape)
                            num_chordwise = value.shape[0]
                            num_spanwise = value.shape[1]

                            

                            for i in range(13):
                                if i == 0:
                                    csdl_var_exp_rotated[i, :, :, :] = csdl_var_exp[i, :, :, :]
                                else:
                                    for j in range(num_chordwise):
                                        for k in range(num_spanwise):
                                            csdl_var_exp_rotated[i, j, k, :] = csdl.reshape(csdl.matvec(csdl.reshape(rotation_mat[i, :, :], new_shape=(3, 3)), csdl.reshape(csdl_var_exp[i, j, k, :], new_shape=(3, ))), (1, 1, 1, 3))

                            if desc is not None:
                                if desc not in ['mirror', 'zero_y', 'average_z']:
                                    raise ValueError(f"description for VLM mesh variables can only be 'mirror' or 'zero_y'. Received '{mesh.description}'")
                                elif desc == 'mirror':
                                    mirrored_mesh = self.create_output(f'{key}_exp', shape=(13, ) + value.shape, val=0)
                                    mirror_mesh_across_xz(original_mesh=csdl_var_exp_rotated, mirror_mesh=mirrored_mesh, num_spanwise=num_spanwise)               
                                elif desc == 'zero_y':
                                    new_mesh = self.create_output(f'{key}_exp', shape=(13, ) + value.shape, val=0)
                                    zero_out_mesh_y(original_mesh=csdl_var_exp_rotated, new_mesh=new_mesh)
                                elif desc == 'average_z':
                                    new_mesh = self.create_output(f'{key}_exp', shape=(13, ) + value.shape, val=0)
                                    average_z(original_mesh=csdl_var_exp_rotated, new_mesh=new_mesh)
                                else:
                                    raise NotImplementedError
                            else:
                                self.register_output(name=f'{key}_exp', var=csdl_var_exp_rotated* 1)
                            
                # Else, expand other csdl variables (e.g., density)
                else:
                    csdl_var = self.declare_variable(key, shape=value.shape)
                    if len(value.shape) == 1 and value.shape[0] == 1:
                        # print(key, value.shape)
                        csdl_var_exp = csdl.expand(csdl_var, shape=(num_nodes * 13, ))
                        self.register_output(name=f'{key}_exp', var=csdl_var_exp)
                    elif len(value.shape) == 1 and value.shape[0] != 1:
                        # print(key, (13, ) + value.shape)
                        csdl_var_exp = csdl.reshape(csdl.expand(csdl_var, shape=(13, ) + value.shape, indices='i->ji'), new_shape=(13, value.shape[0]))
                        self.register_output(name=f'{key}_exp', var=csdl_var_exp)
                    elif len(value.shape) == 2:
                        if num_nodes == value.shape[0]:
                            csdl_var_exp = csdl.reshape(csdl.expand(csdl_var, shape=(13, ) + value.shape, indices='ij->kij'), new_shape=(13*num_nodes, value.shape[1]))
                            self.register_output(name=f'{key}_exp', var=csdl_var_exp)
                        elif num_nodes == value.shape[1]:
                            csdl_var_exp = csdl.reshape(csdl.expand(csdl_var, shape=(13, ) + value.shape, indices='ij->kij'), new_shape=(13*num_nodes, value.shape[0]))
                            self.register_output(name=f'{key}_exp', var=csdl_var_exp)
                    else:
                        print(key, value.shape)
                        raise NotImplementedError
                    
        else:
            for key, value in args.items():
                # Check if key is an aircraft state
                if key in ac_states:
                    csdl_var = self.declare_variable(key, shape=(num_nodes, ))
                    self.register_output(name=f'{key}_exp', var=csdl_var * 1)
                
                # Check whether key is a special case (meaning we don't expand the variable)
                elif key in special_cases:
                    csdl_var = self.declare_variable(key, shape=value.shape)
                    self.register_output(name=f'{key}_exp', var=csdl_var * 1)
                
                # Check whether key is a mesh variable
                elif key in surface_names:
                    csdl_var = self.declare_variable(key, shape=value.shape)
                    mesh = meshes[surface_names.index(key)]
                    deflection = deflections[surface_names.index(key)]
                    desc = mesh.description
                    
                    # Check the length of the shape tuple (3 or 4 should be possible)
                    if len(value.shape) == 4:
                        if value.shape[0] == 1:
                            csdl_var_reshaped = csdl.reshape(csdl_var, new_shape=value.shape[1:])
                            csdl_var_exp_initial = csdl.expand(csdl_var_reshaped, shape=(num_nodes, ) + value.shape[1:], indices='ijk->hijk')
                            if deflection is not None:
                                deflection_angle = self.declare_variable(f'{key}_flap_deflection', shape=(1, ))
                                flap_rot_mat = self.create_output(f'{key}_flap_rot_mat', shape=(3, 3), val=0)
                                flap_rot_mat[0, 0] = csdl.reshape(csdl.cos(deflection_angle), (1, 1))
                                flap_rot_mat[0, 2] = csdl.reshape(csdl.sin(deflection_angle) * -1, (1, 1))
                                flap_rot_mat[1, 1] = self.create_input(f'flap_deflection_one_entry_{key}', val=1, shape=(1, 1))
                                flap_rot_mat[2, 0] = csdl.reshape(csdl.sin(deflection_angle), (1, 1))
                                flap_rot_mat[2, 2] = csdl.reshape(csdl.cos(deflection_angle), (1, 1))
                                flap_rot_mesh = self.create_output(f"{key}_flap_rotation_mesh", shape=csdl_var_exp_initial.shape, val=0)
                                csdl_var_exp = deflect_flap(csdl_var_exp_initial, flap_rot_mesh, flap_rot_mat, 1)
                            else:
                                csdl_var_exp = csdl_var_exp_initial * 1
                            
                            num_spanwise = value.shape[2]

                            if desc is not None:
                                if desc not in ['mirror', 'zero_y', 'average_z']:
                                    raise ValueError(f"description for VLM mesh variables can only be 'mirror' or 'zero_y'. Received '{mesh.description}'")
                                elif desc == 'mirror':
                                    mirrored_mesh = self.create_output(f'{key}_exp', shape=(num_nodes, ) + value.shape[1:], val=0)
                                    mirror_mesh_across_xz(original_mesh=csdl_var_exp, mirror_mesh=mirrored_mesh, num_spanwise=num_spanwise)               
                                elif desc == 'zero_y':
                                    new_mesh = self.create_output(f'{key}_exp', shape=(num_nodes, ) + value.shape[1:], val=0)
                                    zero_out_mesh_y(original_mesh=csdl_var_exp, new_mesh=new_mesh)
                                elif desc == 'average_z':
                                    new_mesh = self.create_output(f'{key}_exp', shape=(num_nodes, ) + value.shape, val=0)
                                    average_z(original_mesh=csdl_var_exp, new_mesh=new_mesh)
                            else:
                                self.register_output(name=f'{key}_exp', var=csdl_var_exp)
                        else:
                            print(key, value.shape)
                            raise NotImplementedError
                    
                    elif len(value.shape) == 3:
                        if value.shape[0] == 1 or value.shape[1] == 1:
                            raise ValueError("VLM meshes cannot have num_spanwise, num_chordwise<2 ")
                        else:
                            csdl_var_exp = csdl.expand(csdl_var, shape=(num_nodes, ) + value.shape, indices='ijk->hijk')
                            num_spanwise = value.shape[1]

                            if desc is not None:
                                if desc not in ['mirror', 'zero_y', 'average_z']:
                                    raise ValueError(f"description for VLM mesh variables can only be 'mirror' or 'zero_y'. Received '{mesh.description}'")
                                elif desc == 'mirror':
                                    mirrored_mesh = self.create_output(f'{key}_exp', shape=(num_nodes, ) + value.shape, val=0)
                                    mirror_mesh_across_xz(original_mesh=csdl_var_exp, mirror_mesh=mirrored_mesh, num_spanwise=num_spanwise)               
                                elif desc == 'zero_y':
                                    new_mesh = self.create_output(f'{key}_exp', shape=(num_nodes, ) + value.shape, val=0)
                                    zero_out_mesh_y(original_mesh=csdl_var_exp, new_mesh=new_mesh)
                                elif desc == 'average_z':
                                    new_mesh = self.create_output(f'{key}_exp', shape=(num_nodes, ) + value.shape, val=0)
                                    average_z(original_mesh=csdl_var_exp, new_mesh=new_mesh)

                            else:
                                self.register_output(name=f'{key}_exp', var=csdl_var_exp)
                    else:
                        raise NotImplementedError

                # Else, expand other csdl variables (e.g., density)
                else:
                    csdl_var = self.declare_variable(key, shape=value.shape)
                    if len(value.shape) == 1 and value.shape[0] == 1:
                        # print(key, value.shape)
                        csdl_var_exp = csdl.expand(csdl_var, shape=(num_nodes, ))
                        self.register_output(name=f'{key}_exp', var=csdl_var_exp)
                    elif len(value.shape) == 1 and value.shape[0] != 1:
                        # print(key, (13, ) + value.shape)
                        csdl_var_exp = csdl.reshape(csdl.expand(csdl_var, shape=(num_nodes, ) + value.shape, indices='i->ji'), new_shape=(num_nodes, value.shape[0]))
                        self.register_output(name=f'{key}_exp', var=csdl_var_exp)
                    elif len(value.shape) == 2:
                        if num_nodes == value.shape[0]:
                            csdl_var_exp = csdl.reshape(csdl.expand(csdl_var, shape=(num_nodes, ) + value.shape, indices='ij->kij'), new_shape=(num_nodes, value.shape[1]))
                            self.register_output(name=f'{key}_exp', var=csdl_var_exp)
                        elif num_nodes == value.shape[1]:
                            csdl_var_exp = csdl.reshape(csdl.expand(csdl_var, shape=(num_nodes, ) + value.shape, indices='ij->kij'), new_shape=(num_nodes, value.shape[0]))
                            self.register_output(name=f'{key}_exp', var=csdl_var_exp)
                    else:
                        print(key, value.shape)
                        raise NotImplementedError


@dataclass
class VLMOutputs:
    """
    VLM Outputs
    -----------

    forces:
        forces

    moments:
        moments

    panel_forces:
        panel forces
    """
    forces : m3l.Variable = None
    moments : m3l.Variable = None
    panel_forces : list = field(default_factory=list) 
    Total_lift:m3l.Variable = None
    Total_drag:m3l.Variable = None
    total_CD:m3l.Variable = None
    total_CL:m3l.Variable = None
    L_over_D:m3l.Variable = None
    cd_induced_drag:m3l.Variable = None

class VASTFluidSover(m3l.ExplicitOperation):
    def initialize(self, kwargs):
        self.parameters.declare('component', default=None)
        self.parameters.declare('mesh', default=None)
        self.parameters.declare('fluid_solver', True)
        self.parameters.declare('num_nodes', default=1, types=int)

        self.parameters.declare('fluid_problem', types=FluidProblem)
        self.parameters.declare('surface_names', types=list)
        self.parameters.declare('surface_shapes', types=list)
        self.parameters.declare('solve_option', default='direct')
        self.parameters.declare('mesh_unit', default='m')
        self.parameters.declare('cl0', default=None)
        self.parameters.declare('input_dicts', default=None)

        self.parameters.declare('ML', default=False)
        self._stability_flag = False
        
        super().initialize(kwargs=kwargs)

    def assign_attributes(self):
        self.name = self.parameters['name']


    def compute(self):
        '''
        Creates a CSDL model to compute the solver outputs.

        Returns
        -------
        csdl_model : csdl.Model
            The csdl model which computes the outputs (the normal solver)
        '''
        fluid_problem = self.parameters['fluid_problem']
        surface_names = self.parameters['surface_names']
        surface_shapes = self.parameters['surface_shapes']
        num_nodes = self.parameters['num_nodes'] #surface_shapes[0][0]
        mesh_unit = self.parameters['mesh_unit']
        cl0 = self.parameters['cl0']
        input_dicts = self.parameters['input_dicts']
        ML = self.parameters['ML']

        if self._stability_flag:
            csdl_model = StabilityAdapterModelCSDL(
                arguments=self.arguments,
                num_nodes=num_nodes,
                neglect=self.neglect_list,
                surface_names=surface_names,
                meshes=self.meshes,
                deflections=self.deflections,
                stability_flag=self._stability_flag,
            )
            
            vast_model = VASTCSDL(
                fluid_problem=fluid_problem,
                surface_names=surface_names,  
                surface_shapes=[(13, ) + surface_shapes[i][1:] for i in range(len(surface_shapes))], #self.surface_shapes,
                num_nodes=num_nodes*13,
                mesh_unit=mesh_unit,
                cl0=cl0,
                input_dicts=input_dicts,
                ML=ML,
            )
            
            operation_name = self.parameters['name']
            
            csdl_model.add(vast_model, operation_name, promotes=[])
            for key, value in self.arguments.items():
                csdl_model.connect(f'{key}_exp',f'{operation_name}.{key}')

        else:
            csdl_model = StabilityAdapterModelCSDL(
                arguments=self.arguments,
                num_nodes=num_nodes,
                neglect=self.neglect_list,
                surface_names=surface_names,
                meshes=self.meshes,
                deflections=self.deflections,
                stability_flag=self._stability_flag,
            )

            vast_model = VASTCSDL(
                fluid_problem=fluid_problem,
                surface_names=surface_names,  
                surface_shapes=self.surface_shapes,
                num_nodes=num_nodes,
                mesh_unit=mesh_unit,
                cl0=cl0,
                input_dicts=input_dicts,
                ML=ML,)
            
            operation_name = self.parameters['name']
            
            csdl_model.add(vast_model, operation_name, promotes=[])
            for key, value in self.arguments.items():
                csdl_model.connect(f'{key}_exp',f'{operation_name}.{key}')
    
        return csdl_model      

    def compute_derivates(self,inputs,derivatives):
        pass

    def evaluate(self, ac_states, atmosphere: List[m3l.Variable], meshes: List[m3l.Variable], 
                 displacements : List[m3l.Variable]=None, ML=False, eval_pt:m3l.Variable=None,
                 wing_AR : m3l.Variable=None, deflections : List[m3l.Variable] = None):
        '''
        Evaluates the vast model.
        
        Parameters
        ----------
        displacements : list of m3l.Variable = None
            The forces on the mesh nodes.

        Returns
        -------
        panel_forces : m3l.Variable
            The displacements of the mesh nodes.

        '''
        # Gets information for naming/shapes
        # beam_name = list(self.parameters['beams'].keys())[0]   # this is only taking the first mesh added to the solver.
        # mesh = list(self.parameters['mesh'].parameters['meshes'].values())[0]   # this is only taking the first mesh added to the solver.
        surface_shapes = self.parameters['surface_shapes']
        self._stability_flag = ac_states.stability_flag
        self.meshes = meshes
        if self._stability_flag:
            self.surface_shapes=surface_shapes #[(13, ) + surface_shapes[i][1:] for i in range(len(surface_shapes))]
            num_nodes = 13
        else:
            self.surface_shapes = surface_shapes
            num_nodes = self.parameters['num_nodes']

        surface_names = self.parameters['surface_names']
        ML = self.parameters['ML']

        if deflections is not None:
            if not check_same_length(surface_names, surface_shapes, meshes, deflections):
                raise ValueError("Lists 'meshes' (input to evaluate()), 'surface_names', 'surface_shapes', 'deflections' must be the same length")
            else:
                self.deflections = deflections
        else:
            if not check_same_length(surface_names, surface_shapes, meshes):
                raise ValueError("Lists 'meshes' (input to evaluate()), 'surface_names', 'surface_shapes' must be the same length")
            self.deflections = [None] * len(meshes)

        self.arguments = {}
        # displacements = self.displacements 
        if displacements is not None:
            for i in range(len(surface_names)):
                surface_name = surface_names[i]
                self.arguments[f'{surface_name}_displacements'] = displacements[i]
        
        if wing_AR is not None:
            self.arguments['wing_AR'] = wing_AR.copy()

        if deflections is not None:
            for i in range(len(meshes)):
                surface_name = surface_names[i]
                mesh = meshes[i]
                self.arguments[surface_name] = mesh
                deflection = deflections[i]
                if deflection is not None: #pass
                    self.arguments[f'{surface_name}_flap_deflection'] = deflection
        else:
            for i in range(len(meshes)):
                surface_name = surface_names[i]
                mesh = meshes[i]
                self.arguments[surface_name] = mesh

        # print(arguments)
        # new_arguments = {**arguments, **ac_states}
        self.arguments['u'] = ac_states.u
        self.arguments['v'] = ac_states.v
        self.arguments['w'] = ac_states.w
        self.arguments['p'] = ac_states.p
        self.arguments['q'] = ac_states.q
        self.arguments['r'] = ac_states.r
        self.arguments['phi'] = ac_states.phi
        self.arguments['theta'] = ac_states.theta
        self.arguments['psi'] = ac_states.psi
        self.arguments['density'] = atmosphere.density
        
        self.neglect_list = ['wing_AR']#, 'evaluation_pt']
        if eval_pt is None:
            if self._stability_flag:
                eval_pt = m3l.Variable(shape=(num_nodes, 3), value=np.tile(np.array([0., 0., 0.]), (num_nodes, 1)))
            else:
                eval_pt = m3l.Variable(shape=(3, ), value=np.array([0., 0., 0.]))
            self.arguments['evaluation_pt'] = eval_pt
            
            for i in range(len(meshes)):
                surface_name = surface_names[i]
                self.arguments[f'{surface_name}_rot_ref'] = eval_pt
                self.neglect_list.append(f'{surface_name}_rot_ref')
        else:
            self.arguments['evaluation_pt'] = eval_pt
            for i in range(len(meshes)):
                surface_name = surface_names[i]
                self.arguments[f'{surface_name}_rot_ref'] = eval_pt
        # self.arguments['psiw'] = ac_states['psi_w']

        
        # Create the M3L variables that are being output
        forces = []
        cl_spans = []
        re_spans = []  
        panel_areas = [] 
        evaluation_pts = []
        # if self._stability_flag:
        for i in range(len(surface_names)):
            surface_name = surface_names[i]
            surface_shapes = self.surface_shapes[i]
            # num_nodes = surface_shapes[0]
            nx = surface_shapes[1]
            ny = surface_shapes[2]

            force = m3l.Variable(name=f'{self.name}.{surface_name}_total_forces', shape=(num_nodes, int((nx-1)*(ny-1)), 3), operation=self)
            cl_span = m3l.Variable(name=f'{self.name}.{surface_name}_cl_span_total', shape=(num_nodes, int(ny-1),1), operation=self)
            re_span = m3l.Variable(name=f'{self.name}.{surface_name}_re_span', shape=(num_nodes, int(ny-1),1), operation=self)
            panel_area = m3l.Variable(name=f'{self.name}.{surface_name}_s_panel', shape=(num_nodes,nx-1,ny-1), operation=self)
            evaluation_pt = m3l.Variable(name=f'{self.name}.{surface_name}_eval_pts_coords', shape=(num_nodes,nx-1,ny-1,3), operation=self)

            forces.append(force)
            cl_spans.append(cl_span)
            re_spans.append(re_span)
            panel_areas.append(panel_area)
            evaluation_pts.append(evaluation_pt)

        total_force = m3l.Variable(name=f'{self.name}.F', shape=(num_nodes, 3), operation=self)
        total_moment = m3l.Variable(name=f'{self.name}.M', shape=(num_nodes, 3), operation=self)

        # else:
        #     for i in range(len(surface_names)):
        #         surface_name = surface_names[i]
        #         surface_shapes = self.surface_shapes[i]
        #         # num_nodes = surface_shapes[0]
        #         nx = surface_shapes[1]
        #         ny = surface_shapes[2]

        #         force = m3l.Variable(name=f'{surface_name}_total_forces', shape=(num_nodes, int((nx-1)*(ny-1)), 3), operation=self)
        #         cl_span = m3l.Variable(name=f'{surface_name}_cl_span_total', shape=(num_nodes, int(ny-1),1), operation=self)
        #         re_span = m3l.Variable(name=f'{surface_name}_re_span', shape=(num_nodes, int(ny-1),1), operation=self)
        #         panel_area = m3l.Variable(name=f'{surface_name}_s_panel', shape=(num_nodes,nx-1,ny-1), operation=self)
        #         evaluation_pt = m3l.Variable(name=f'{surface_name}_eval_pts_coords', shape=(num_nodes,nx-1,ny-1,3), operation=self)

        #         forces.append(force)
        #         cl_spans.append(cl_span)
        #         re_spans.append(re_span)
        #         panel_areas.append(panel_area)
        #         evaluation_pts.append(evaluation_pt)

        #     total_force = m3l.Variable(name='F', shape=(num_nodes, 3), operation=self)
        #     total_moment = m3l.Variable(name='M', shape=(num_nodes, 3), operation=self)

        # L_over_D = m3l.Variable(name = f'{self.name}.L_over_D', shape = (num_nodes,), operation=self)
        L_over_Drag = m3l.Variable(name = f'{self.name}.L_over_D', shape = (num_nodes,1), operation=self)
        total_lift = m3l.Variable(name = f'{self.name}.total_lift', shape = (num_nodes,1), operation=self)
        total_drag = m3l.Variable(name = f'{self.name}.total_drag', shape = (num_nodes,1), operation=self)
        total_CD = m3l.Variable(name = f'{self.name}.total_CD', shape = (num_nodes,1), operation=self)
        cd_induced_drag = m3l.Variable(name = f'{self.name}.cd_induced_drag', shape = (num_nodes,1), operation=self)
        total_CL = m3l.Variable(name = f'{self.name}.total_CL', shape = (num_nodes,1), operation=self)
        
        # L_over_Drag = m3l.Variable(name = 'L_over_D', shape = (num_nodes,1), operation=self)

        vlm_outputs = VLMOutputs()

        # return spanwise cl, forces on panels with vlm internal correction for cl0 and cdv, total force and total moment for trim
        
        if ML:
            return cl_spans, re_spans, forces, panel_areas, evaluation_pts, total_force, total_moment
        else:
            vlm_outputs.forces = total_force
            vlm_outputs.moments = total_moment
            vlm_outputs.panel_forces = forces
            vlm_outputs.Total_lift = total_lift
            vlm_outputs.Total_drag = total_drag
            vlm_outputs.total_CD = total_CD
            vlm_outputs.total_CL = total_CL
            vlm_outputs.cd_induced_drag = cd_induced_drag
            vlm_outputs.L_over_D = L_over_Drag
            return vlm_outputs        

class VASTCSDL(csdl.Model):
    def initialize(self):
        self.parameters.declare('fluid_problem',default=None)
        self.parameters.declare('surface_names', types=list)
        self.parameters.declare('surface_shapes', types=list)
        self.parameters.declare('num_nodes', types=int, default=1)
        self.parameters.declare('solve_option', default='direct')
        self.parameters.declare('mesh_unit', default='m')
        self.parameters.declare('cl0', default=None)
        self.parameters.declare('input_dicts', default=None)
        self.parameters.declare('ML', default=False)

    def define(self):
        fluid_problem = self.parameters['fluid_problem']
        solver_options = fluid_problem.solver_option
        problem_type = fluid_problem.problem_type

        surface_names = self.parameters['surface_names']
        surface_shapes = self.parameters['surface_shapes']
        num_nodes = self.parameters['num_nodes'] #surface_shapes[0][0]
        mesh_unit = self.parameters['mesh_unit']
        cl0 = self.parameters['cl0']

        ML = self.parameters['ML']

        # todo: connect the mesh to the solver
        # wing = model_1.create_input('wing', val=np.einsum('i,jkl->ijkl', np.ones((num_nodes)), mesh))
        # try:
        #     input_dicts = self.parameters['input_dicts']
        #     submodel = CreateACSatesModule(v_inf=input_dicts['v_inf'],theta=input_dicts['theta'],num_nodes=num_nodes)
        #     self.add(submodel, 'ACSates')

        for i in range(len(surface_names)):
            surface_name = surface_names[i]
            surface_shape = self.parameters['surface_shapes'][i]
            # displacements = self.declare_variable(f'{surface_name}_displacements', shape=(surface_shape),val=0.)

            undef_mesh = self.declare_variable(f'{surface_name}_mesh', val=np.zeros(surface_shape))
            # mesh = undef_mesh  #+ displacements

        # except:
        #     pass

        if fluid_problem.solver_option == 'VLM' and fluid_problem.problem_type == 'fixed_wake':
            submodel = VLMSolverModel(
                num_nodes=num_nodes,
                surface_names=surface_names,
                surface_shapes=surface_shapes,
                AcStates='dummy',
                mesh_unit=mesh_unit,
                cl0=cl0,
                ML=ML
            )
            self.add(submodel, 'VLMSolverModel')

        # TODO: make dynamic case works
        elif fluid_problem.solver_option == 'VLM' and fluid_problem.problem_type == 'prescribed_wake':
            sim = Simulator(UVLMSolver(num_times=nt,h_stepsize=h_stepsize,states_dict=states_dict,
                                            surface_properties_dict=surface_properties_dict,mesh_val=mesh_val), mode='rev')

# if __name__ == "__main__":

#     # import numpy as np
#     # from VAST.utils.generate_mesh import *
#     # from python_csdl_backend import Simulator
#     # import caddee.api as cd 

#     # fluid_problem = FluidProblem(solver_option='VLM', problem_type='fixed_wake')

#     # num_nodes=1; nx=3; ny=11

#     # v_inf = np.ones((num_nodes,1))*248.136
#     # theta = np.deg2rad(np.ones((num_nodes,1))*5)  # pitch angles


#     # surface_names = ['wing']
#     # surface_shapes = [(num_nodes, nx, ny, 3)]
#     # mesh_dict = {
#     #     "num_y": ny, "num_x": nx, "wing_type": "rect", "symmetry": False, "span": 10.0,
#     #     "chord": 1, "span_cos_sppacing": 1.0, "chord_cos_sacing": 1.0,
#     # }
#     # # Generate mesh of a rectangular wing
#     # mesh = generate_mesh(mesh_dict)

#     # ###########################################
#     # # 1. Create a dummy m3l.Model()
#     # ###########################################
#     # dummy_model = m3l.Model()
#     # # fluid_model = VASTFluidSover(fluid_problem=fluid_problem, surface_names=surface_names, surface_shapes=surface_shapes, mesh_unit='m', cl0=0.0)


#     # # submodel = CreateACSatesModel(v_inf=v_inf, theta=theta, num_nodes=num_nodes)
#     # # model_1.add(submodel, 'InputsModule')
#     # fluid_model = VASTFluidSover(fluid_problem=fluid_problem,
#     #                              surface_names=surface_names,
#     #                              surface_shapes=surface_shapes,
#     #                              input_dicts=None,)


#     # ###########################################
#     # # 3. set fluid_model inputs 
#     # ###########################################
#     # fluid_model.set_module_input('u',val=v_inf)
#     # fluid_model.set_module_input('v',val=np.zeros((num_nodes, 1)))
#     # fluid_model.set_module_input('w',val=np.ones((num_nodes,1))*0)
#     # fluid_model.set_module_input('p',val=np.zeros((num_nodes, 1)))
#     # fluid_model.set_module_input('q',val=np.zeros((num_nodes, 1)))
#     # fluid_model.set_module_input('r',val=np.zeros((num_nodes, 1)))
#     # fluid_model.set_module_input('phi',val=np.zeros((num_nodes, 1)))
#     # fluid_model.set_module_input('theta',val=theta)
#     # fluid_model.set_module_input('psi',val=np.zeros((num_nodes, 1)))
#     # fluid_model.set_module_input('x',val=np.zeros((num_nodes, 1)))
#     # fluid_model.set_module_input('y',val=np.zeros((num_nodes, 1)))
#     # fluid_model.set_module_input('z',val=np.ones((num_nodes, 1))*1000)
#     # fluid_model.set_module_input('phiw',val=np.zeros((num_nodes, 1)))
#     # fluid_model.set_module_input('phi',val=np.zeros((num_nodes, 1)))
#     # fluid_model.set_module_input('psiw',val=np.zeros((num_nodes, 1)))
 
#     # fluid_model.set_module_input('wing_undef_mesh', val=np.einsum('i,jkl->ijkl', np.ones((num_nodes)), mesh))

#     # input_dicts = {}
#     # # input_dicts['v_inf'] = v_inf
#     # # input_dicts['theta'] = theta
#     # # input_dicts['undef_mesh'] = [np.einsum('i,jkl->ijkl', np.ones((num_nodes)), mesh)]
#     # # input_dicts['displacements'] = [np.zeros((num_nodes, nx, ny, 3))]

#     # ###########################################
#     # # 2. Create fluid_model as VASTFluidSover 
#     # # (msl.explicit operation)
#     # ###########################################


#     # displacements = []
#     # for i in range(len(surface_names)):
#     #     surface_name = surface_names[i]
#     #     surface_shape = surface_shapes[i]
#     #     displacement = m3l.Variable(f'{surface_name}_displacements',shape=surface_shape,value=np.ones(surface_shape)*10)
#     #     fluid_model.set_module_input(f'{surface_name}_displacements', val=np.ones(surface_shape)*100)
#     #     displacements.append(displacement)

#     # ###########################################
#     # # 4. call fluid_model.evaluate to get
#     # # surface panel forces
#     # ###########################################
#     # forces = fluid_model.evaluate(displacements)

#     # ###########################################
#     # # 5. register outputs to dummy_model
#     # ###########################################
#     # for i in range(len(surface_names)):
#     #     surface_name = surface_names[i]
#     #     dummy_model.register_output(forces[i])
        
#     # ###########################################
#     # # 6. call _assemble_csdl to get dummy_model_csdl
#     # ###########################################
#     # dummy_model_csdl = dummy_model._assemble_csdl()
#     # ###########################################
#     # # 7. use sim.run to run the csdl model
#     # ###########################################    

#     # sim = Simulator(dummy_model_csdl,analytics=False) # add simulator
#     # sim.run()




#     import numpy as np
#     from VAST.utils.generate_mesh import *
#     from python_csdl_backend import Simulator

#     fluid_problem = FluidProblem(solver_option='VLM', problem_type='fixed_wake')

#     num_nodes=1; nx=5; ny=11

#     v_inf = np.ones((num_nodes,1))*57
#     theta = np.deg2rad(np.ones((num_nodes,1))*5)  # pitch angles

#     model_1 = ModuleCSDL()

#     # submodel = CreateACSatesModel(v_inf=v_inf, theta=theta, num_nodes=num_nodes)
#     # model_1.add(submodel, 'InputsModule')
#     # model_1.add_design_variable('InputsModule.u')
#     model_1.create_input('u', val=70, shape=(num_nodes, 1))
#     model_1.add_design_variable('u', lower=50, upper=100, scaler=1e-2)

#     surface_names = ['wing','tail']
#     surface_shapes = [(num_nodes, nx, ny, 3),(num_nodes, nx-2, ny-2, 3)]

#     mesh_dict = {
#         "num_y": ny, "num_x": nx, "wing_type": "rect", "symmetry": False, "span": 10.0,
#         "chord": 1, "span_cos_sppacing": 1.0, "chord_cos_sacing": 1.0,
#     }

#     mesh_dict_1 = {
#         "num_y": ny-2, "num_x": nx-2, "wing_type": "rect", "symmetry": False, "span": 10.0,
#         "chord": 1, "span_cos_sppacing": 1.0, "chord_cos_sacing": 1.0,
#     }

#     # Generate mesh of a rectangular wing
#     mesh = generate_mesh(mesh_dict) 
#     mesh_1 = generate_mesh(mesh_dict_1) 
#     wing = model_1.create_input('wing', val=np.einsum('i,jkl->ijkl', np.ones((num_nodes)), mesh))
#     wing = model_1.create_input('tail', val=np.einsum('i,jkl->ijkl', np.ones((num_nodes)), mesh_1))

#     # add VAST fluid solver

#     submodel = VASTCSDL(
#         fluid_problem=fluid_problem,
#         surface_names=surface_names,
#         surface_shapes=surface_shapes,
#     )
#     model_1.add(submodel, 'VASTSolverModule')
#     sim = Simulator(model_1, analytics=True) # add simulator

    
#     model_1.add_objective('VASTSolverModule.VLMSolverModel.VLM_outputs.LiftDrag.total_drag')
#     sim.run()
#     sim.check_totals()

