import csdl
from VAST.core.fluid_problem import FluidProblem
import m3l
from VAST.core.vlm_llt.NodalMapping import NodalMap,RadialBasisFunctions
import numpy as np


class VASTNodelDisplacements(m3l.ExplicitOperation):
    def initialize(self, kwargs):
        self.parameters.declare('name', types=str)
        self.parameters.declare('surface_names', types=list)
        self.parameters.declare('surface_shapes', types=list)
        self.parameters.declare('initial_meshes', default=list) # vlm mesh before optimization

    def assign_attributes(self):
        self.name = self.parameters['name']

    def compute(self, nodal_displacements, nodal_displacements_mesh):
        surface_names = self.parameters['surface_names']  
        surface_shapes = self.parameters['surface_shapes']
        initial_meshes = self.parameters['initial_meshes']

        if not isinstance(nodal_displacements, list):
            raise TypeError('nodal_displacements must be a list of m3l.Variable in VASTNodelDisplacements.compute()')
        if not isinstance(nodal_displacements_mesh, list):
            raise TypeError('nodal_displacements_mesh must be a list of am.MappedArray in VASTNodelDisplacements.compute()')


        csdl_model = csdl.Model()

        for i in range(len(surface_names)):
            surface_name = surface_names[i]
            surface_shape = surface_shapes[i]
            initial_mesh = initial_meshes[i]
            
            nodal_displacements_surface = nodal_displacements[i] # nodal displacement on the oml mesh for the current surface
            nodal_displacements_mesh_surface = nodal_displacements_mesh[i] # oml mesh for the current surface on which the displacement is defined

            displacements_map = self.disp_map(initial_mesh.reshape((-1,3)), oml=nodal_displacements_mesh_surface.reshape((-1,3)))


            displacements_map_csdl = csdl_model.create_input(name=surface_name+'_displacements_map', val=displacements_map)

            # register an input for the oml nodal displacements
            nodal_displacements_csdl = csdl_model.declare_variable(name=surface_name+'_nodal_displacements', shape=nodal_displacements_surface.shape)
            # print('shapes')
            # print('displacements_map_csdl shape', displacements_map_csdl.shape)
            # print('nodal_displacements_csdl shape', nodal_displacements_csdl.shape)
            # print('nodal_displacements_surface', nodal_displacements_surface.shape)
            # print('nodal_displacements_mesh_surface', nodal_displacements_mesh_surface.shape)
            
            flatenned_vlm_mesh_displacements = csdl.matmat(displacements_map_csdl, csdl.reshape(nodal_displacements_csdl,
                                                            (nodal_displacements_surface.shape[0]*nodal_displacements_surface.shape[1],3)))
            # print('nodal_displacements_csdl', nodal_displacements_csdl.shape)
            # print('displacements_map_csdl', displacements_map_csdl.shape)
            # print('flatenned_vlm_mesh_displacements', flatenned_vlm_mesh_displacements.shape)
            # TODO: think about how to vectorize this across num_nodes

            vlm_mesh_displacements = csdl.reshape(flatenned_vlm_mesh_displacements, new_shape=surface_shape)
            csdl_model.register_output(f'{surface_name}_displacements', vlm_mesh_displacements)
            print('vlm_mesh_displacements--------------------------name--------------------------------------------', vlm_mesh_displacements.name)

        return csdl_model

    def evaluate(self, nodal_displacements, nodal_displacements_mesh):
        '''
        Maps nodal displacements_mesh from arbitrary locations to the mesh nodes.
        
        Parameters
        ----------
        nodal_displacements : a list of m3l.Variable
            The nodal_displacements to be mapped to the mesh nodes.
        nodal_displacements_mesh : a list of am.MappedArray
            The mesh that the nodal displacements_mesh are currently defined over.

        Returns
        -------
        mesh_forces : m3l.Variable
            The forces on the mesh.
        '''

        if not isinstance(nodal_displacements, list):
            raise TypeError('nodal_displacements must be a list of m3l.Variable in VASTNodelDisplacements.evaluate()')
        if not isinstance(nodal_displacements_mesh, list):
            raise TypeError('nodal_displacements_mesh must be a list of am.MappedArray in VASTNodelDisplacements.evaluate()')

        surface_names = self.parameters['surface_names']  
        surface_shapes = self.parameters['surface_shapes']
        initial_meshes = self.parameters['initial_meshes']

        arguments = {}

        operation_csdl = self.compute(nodal_displacements=nodal_displacements, nodal_displacements_mesh=nodal_displacements_mesh)
        for i in range(len(surface_names)):
            surface_name = surface_names[i]
            arguments[surface_name+'_nodal_displacements'] = nodal_displacements[i]

        displacement_map_operation = m3l.CSDLOperation(name='vlm_disp_map', arguments=arguments, operation_csdl=operation_csdl)

        vlm_displacements = []
        for i in range(len(surface_names)):
            surface_name = surface_names[i]
            surface_shape = surface_shapes[i]
            vlm_displacement = m3l.Variable(name=f'{surface_name}_displacements', shape=surface_shape, operation=displacement_map_operation)
            vlm_displacements.append(vlm_displacement)

        return vlm_displacements


    def disp_map(self, mesh, oml):

        # project the displacement on the oml mesh to the camber surface mesh
        # print('mesh', type(mesh))
        # print('mesh', mesh)
        # print('oml', type(oml))
        weights = (NodalMap(oml.reshape((-1,3)), mesh.reshape((-1,3)), RBF_width_par=2, RBF_func=RadialBasisFunctions.Gaussian).map)


        return weights

class VASTNodalForces(m3l.ExplicitOperation):

    def initialize(self, kwargs):
        self.parameters.declare('name', types=str)
        self.parameters.declare('surface_names', types=list)
        self.parameters.declare('surface_shapes', types=list)
        self.parameters.declare('initial_meshes', default=list)

    def assign_attributes(self):
        self.name = self.parameters['name']

    def compute(self):
        surface_names = self.parameters['surface_names']  
        surface_shapes = self.parameters['surface_shapes']
        initial_meshes = self.parameters['initial_meshes']
        
        nodal_force_meshes = self.nodal_forces_meshes
        print('nodal_force_meshes', nodal_force_meshes)

        csdl_model = csdl.Model()

        for i in range(len(surface_names)):
            surface_name = surface_names[i]
            surface_shape = surface_shapes[i]
            initial_mesh = initial_meshes[i]
            nx = surface_shape[1]
            ny = surface_shape[2]
            outshape = int((nx-1)*(ny-1))
            num_nodes = surface_shape[0]
            eval_pts_location = 0.25
            if type(initial_mesh)==np.ndarray: # NOTE: this is just for testing purpose, in the long term, we should raise an error if oml_mesh is not a m3l.MappedArray
                initial_mesh_reshaped = initial_mesh.reshape((num_nodes, nx, ny, 3))
            else:
                initial_mesh_reshaped = initial_mesh.reshape((num_nodes, nx, ny, 3)).value

            force_points_vlm = (
                    (1 - eval_pts_location) * 0.5 * initial_mesh_reshaped[:, 0:-1, 0:-1, :] +
                    (1 - eval_pts_location) * 0.5 * initial_mesh_reshaped[:, 0:-1, 1:, :] +
                    eval_pts_location * 0.5 * initial_mesh_reshaped[:, 1:, 0:-1, :] +
                    eval_pts_location * 0.5 * initial_mesh_reshaped[:, 1:, 1:, :])
            # csdl_model.create_input(n
            # csdl_model.print_var(force_points_vlm)
            # self.print_var(force_points_vlm)
            print('force_points_vlm',type(force_points_vlm))
            print('force_points_vlm shape', force_points_vlm.shape)
            if type(initial_mesh)==np.ndarray:
                force_map = self.disp_map(force_points_vlm.reshape((-1,3)),nodal_force_meshes[i].reshape((-1,3)),save_map=True)   
                # force_map_1 = self.disp_map(force_points_vlm.reshape((-1,3)),nodal_force_meshes[i].reshape((-1,3)),save_map=True)   
            else:
                force_map = self.disp_map(force_points_vlm.reshape((-1,3)),nodal_force_meshes[i].value.reshape((-1,3)))   
                # force_map_1 = self.disp_map(force_points_vlm.reshape((-1,3)),nodal_force_meshes[i].value.reshape((-1,3)))   
            # this is the inverse of the displacements_map in the displacements operation
            # print('force_map',type(force_map_1))
            # print('force_map shape',force_map_1.shape)
            # force_map = 1780 * np.ones((force_map_1.shape))
            # exit()

            force_map_csdl = csdl_model.create_input(name=surface_name+'_force_map', val=force_map)
            vlm_forces_csdl = csdl_model.declare_variable(name=surface_name+'_total_forces', shape=(num_nodes, outshape, 3))
            

            # print('nodal_displacements_csdl', nodal_displacements_csdl.shape)
            # print('displacements_map_csdl', displacements_map_csdl.shape)
            # print('flatenned_vlm_mesh_displacements', flatenned_vlm_mesh_displacements.shape)
            flatenned_oml_mesh_forces = csdl.matmat(force_map_csdl,csdl.reshape(vlm_forces_csdl,(outshape*num_nodes,3)))

            oml_mesh_forces = csdl.reshape(flatenned_oml_mesh_forces, new_shape=nodal_force_meshes[i].shape)
            csdl_model.register_output(f'{surface_name}_oml_forces', oml_mesh_forces)

        return csdl_model

    def evaluate(self, vlm_forces, nodal_force_meshes):
        '''
        Maps nodal displacements_mesh from arbitrary locations to the mesh nodes.
        
        Parameters
        ----------
        nodal_displacements : a list of m3l.Variable
            The nodal_displacements to be mapped to the mesh nodes.
        nodal_force_meshes : a list of am.MappedArray
            The mesh that the nodal displacements_mesh are currently defined over.

        Returns
        -------
        mesh_forces : m3l.Variable
            The forces on the mesh.
        '''

        surface_names = self.parameters['surface_names']  
        surface_shapes = self.parameters['surface_shapes']
        initial_meshes = self.parameters['initial_meshes']

        if not isinstance(vlm_forces, list):
            raise TypeError('vlm_forces must be a list of m3l.Variable in VASTNodelForces.evaluate()')
        if not isinstance(nodal_force_meshes, list):
            raise TypeError('nodal_force_meshes must be a list of am.MappedArray in VASTNodelForces.evaluate()')     

        self.nodal_forces_meshes = nodal_force_meshes

        self.arguments = {}

        for i in range(len(surface_names)):
            surface_name = surface_names[i]
            self.arguments[surface_name+'_total_forces'] = vlm_forces[i]

        oml_forces = []
        for i in range(len(surface_names)):
            surface_name = surface_names[i]
            surface_shape = surface_shapes[i]
            nx = surface_shape[2]
            ny = surface_shape[1]
            outshape = int((nx-1)*(ny-1))
            num_nodes = surface_shape[0]  
            oml_mesh = nodal_force_meshes[i]
        
            oml_force = m3l.Variable(name=f'{surface_name}_oml_forces', shape=oml_mesh.shape, operation=self)
            oml_forces.append(oml_force)

        return oml_forces


    def disp_map(self, mesh, oml,save_map=False):

        # project the displacement on the oml mesh to the camber surface mesh
        # print('mesh', type(mesh))
        # print('oml', type(oml))
        # print('oml', type(oml))
        # print(mesh)
        # print(oml)
        weights = NodalMap(oml.reshape((-1,3)), mesh.reshape(-1,3), RBF_width_par=2, RBF_func=RadialBasisFunctions.Gaussian).map
        # np.savetxt('weights.txt',weights)
        # print(weights.shape)
        # print((np.sum(weights,axis=0)))
        # print(np.sum(weights,axis=1))
        if save_map:
            np.savetxt('weights.txt',weights)


        return weights.T



# if __name__ == "__main__":
#     import numpy as np
#     from VAST.utils.generate_mesh import *
#     from python_csdl_backend import Simulator

#     fluid_problem = FluidProblem(solver_option='VLM', problem_type='fixed_wake')

#     num_nodes=1; nx=3; ny=11
#     outshape = int((nx-1)*(ny-1))

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
#     fluid_model = VASTNodalForces(
#                                  surface_names=surface_names,
#                                  surface_shapes=surface_shapes,
#                                  initial_meshes=[input_dicts['undef_mesh'][0].reshape(-1,3)],)


#     vlm_forces = []
#     for i in range(len(surface_names)):
#         surface_name = surface_names[i]
#         surface_shape = surface_shapes[i]
#         vlm_force = m3l.Variable(f'{surface_name}_total_forces',shape=(num_nodes, outshape, 3),value=np.zeros((num_nodes, outshape, 3)))
#         fluid_model.set_module_input(f'{surface_name}_total_forces', val=np.zeros((num_nodes, outshape, 3)))
#         vlm_forces.append(vlm_force)

#     ###########################################
#     # 4. call fluid_model.evaluate to get
#     # surface panel forces
#     ###########################################
#     oml_forces = fluid_model.evaluate(vlm_forces,oml_mesh.reshape(-1,3))

#     ###########################################
#     # 5. register outputs to dummy_model
#     ###########################################
#     for i in range(len(surface_names)):
#         surface_name = surface_names[i]
#         dummy_model.register_output(oml_forces[i])
        
#     ###########################################
#     # 6. call _assemble_csdl to get dummy_model_csdl
#     ###########################################
#     dummy_model_csdl = dummy_model._assemble_csdl()
#     ###########################################
#     # 7. use sim.run to run the csdl model
#     ###########################################    
#     # create a random displacement on the oml mesh
#     disp_temp = np.linspace(0.6, 0,7)
#     disp_z = np.outer(np.ones(21),np.concatenate((disp_temp, disp_temp[:-1][::-1])))
#     disp = np.zeros((1, 21, 13, 3))
#     disp[0, :, :, 2] = disp_z
#     vlm_force = dummy_model_csdl.create_input('wing_total_forces', val=np.random.rand(1, outshape, 3))

#     sim = Simulator(dummy_model_csdl, analytics=False) # add simulator
#     sim.run()


# if __name__ == "__main__":
#     import numpy as np
#     from VAST.utils.generate_mesh import *
#     from python_csdl_backend import Simulator

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
#                                  surface_names=surface_names,
#                                  surface_shapes=surface_shapes,
#                                  initial_meshes=[input_dicts['undef_mesh'][0].reshape(-1,3)],)


#     displacements = []
#     for i in range(len(surface_names)):
#         surface_name = surface_names[i]
#         surface_shape = surface_shapes[i]
#         displacement = m3l.Variable(f'{surface_name}_nodal_displacements',shape=surface_shape,value=np.zeros(oml_mesh.shape))
#         fluid_model.set_module_input(f'{surface_name}_nodal_displacements', val=np.zeros(oml_mesh.shape))
#         displacements.append(displacement)

#     ###########################################
#     # 4. call fluid_model.evaluate to get
#     # surface panel forces
#     ###########################################
#     vlm_displacements = fluid_model.evaluate(displacements,oml_mesh.reshape(-1,3))

#     ###########################################
#     # 5. register outputs to dummy_model
#     ###########################################
#     for i in range(len(surface_names)):
#         surface_name = surface_names[i]
#         dummy_model.register_output(vlm_displacements[i])
        
#     ###########################################
#     # 6. call _assemble_csdl to get dummy_model_csdl
#     ###########################################
#     dummy_model_csdl = dummy_model._assemble_csdl()
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