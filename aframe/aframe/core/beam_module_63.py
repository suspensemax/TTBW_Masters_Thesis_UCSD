# from aframe.core.aframe import Aframe
from aframe.core.aframe_63 import Aframe
import numpy as np
import csdl
import m3l
import array_mapper as am
from typing import Tuple, Dict
from dataclasses import dataclass, field


@dataclass
class EBBeamOutputs:
    """Data class for Euler--Bernoulli beam outputs"""
    displacements : m3l.Variable
    # stresses : m3l.Variable
    stresses : list = field(default_factory=list) 
    # top_buckling : m3l.Variable
    top_buckling : list = field(default_factory=list)
    # bot_buckling : m3l.Variable
    bot_buckling : list = field(default_factory=list)
    rotations : m3l.Variable = None
    mass : m3l.Variable = None
    cg_vector : m3l.Variable = None
    inertia_tensor : m3l.Variable = None
    struct_mass : m3l.Variable = None
    cg_vector_not_list : m3l.Variable = None
    inertia_tensor_not_list : m3l.Variable = None

class EBBeam(m3l.ExplicitOperation):
    def initialize(self, kwargs):
        self.parameters.declare('name', types=str)
        self.parameters.declare('beams', default={})
        self.parameters.declare('bounds', default={})
        self.parameters.declare('joints', default={})
        self.parameters.declare('mesh_units', default='m')
        self.num_nodes = None

    def assign_attributes(self):
        self.name = self.parameters['name']
        self.beams = self.parameters['beams']
        self.bounds = self.parameters['bounds']
        self.joints = self.parameters['joints']
        self.mesh_units = self.parameters['mesh_units']

    def compute(self):
        '''
        Creates a CSDL model to compute the solver outputs.

        Returns
        -------
        csdl_model : csdl.Model
            The csdl model which computes the outputs (the normal solver)
        '''
        beams = self.parameters['beams']
        bounds = self.parameters['bounds']
        joints = self.parameters['joints']
        mesh_units = self.parameters['mesh_units']

        csdl_model = LinearBeamCSDL(
            beams=beams,  
            bounds=bounds,
            joints=joints,
            mesh_units=mesh_units)
        
        return csdl_model

    # # optional - for dynamic or conservative coupling
    # def compute_derivatives(self, arguments, derivatives):
    #     '''
    #     Creates a CSDL model to compute the derivatives of the solver outputs.


    #     '''
    #     geo_mesh = arguments['geo_mesh']
    #     displacement_nodes = arguments['displacement_nodes']
    #     derivatives['nodal_displacement', 'displacement'] = displacement_map(geo_mesh, displacement_nodes)

    # def evaluate(self, beam_mesh : m3l.Variable, t_web : m3l.Variable, t_top : m3l.Variable, t_bot : m3l.Variable,
    #              forces:m3l.Variable=None, moments:m3l.Variable=None) -> EBBeamOutputs:

    def evaluate(self, width=[], height=[], t_top=[], t_bot=[], t_web=[], beam_nodes = [],
                 forces:m3l.Variable=None, moments:m3l.Variable=None) -> EBBeamOutputs:
        
        self.arguments = {}
        beams = self.parameters['beams']

        '''
        Evaluates the beam model.
        
        Parameters
        ----------
        forces : m3l.Variable = None
            The forces on the mesh nodes.
        moments : m3l.Variable = None
            The moments on the mesh nodes.

        Returns
        -------
        displacements : m3l.Variable
            The displacements of the mesh nodes.
        rotations : m3l.Variable
            The rotations of the mesh nodes.

        '''

        displacements_list = []
        rotations_list = []
        mass_list = []
        cg_list = []
        inertia_tensor_list = []
        stresses_list = []
        top_buckling_list = []
        bot_buckling_list = []

        for i, beam_name in enumerate(beams):
            # mesh = beam_mesh[i].beam_nodes
            # beam_height = beam_mesh[i].height
            # beam_width = beam_mesh[i].width

            beam_name = list(self.parameters['beams'].keys())[i]
            n = len(self.beams[beam_name]['nodes'])

            # self.arguments = {}
            # self.arguments[f'{beam_name}_width'] = beam_width
            # self.arguments[f'{beam_name}_height'] = beam_height
            self.arguments[f'{beam_name}_width'] = width[i]
            self.arguments[f'{beam_name}_height'] = height[i]
            self.arguments[f'{beam_name}_ttop'] = t_top[i]
            self.arguments[f'{beam_name}_tbot'] = t_bot[i]
            self.arguments[f'{beam_name}_tweb'] = t_web[i]
            self.arguments[f'{beam_name}_mesh'] = beam_nodes[i]
            # self.arguments[f'{beam_name}_mesh'] = mesh


            if forces is not None:
                self.arguments[f'{beam_name}_forces'] = forces[i]
            if moments is not None:
                self.arguments[f'{beam_name}_moments'] = moments[i]

            # displacements = m3l.Variable(name=f'{beam_name}_displacement', shape=mesh.shape, operation=self)
            # rotations = m3l.Variable(name=f'{beam_name}_rotation', shape=mesh.shape, operation=self)
            displacements = m3l.Variable(name=f'{beam_name}_displacement', shape=beam_nodes[i], operation=self)
            rotations = m3l.Variable(name=f'{beam_name}_rotation', shape=beam_nodes[i], operation=self)
            mass = m3l.Variable(name='mass', shape=(1,), operation=self)
            cg = m3l.Variable(name='cg_vector', shape=(3,), operation=self)
            inertia_tensor = m3l.Variable(name='inertia_tensor', shape=(3,3), operation=self)

            stresses = m3l.Variable(name=f"{beam_name}_stress", shape=(n, 5), operation=self)
            top_buckling = m3l.Variable(name=f"{beam_name}_top_bkl", shape=(n-1, ), operation=self)
            bot_buckling = m3l.Variable(name=f"{beam_name}_bot_bkl", shape=(n-1, ), operation=self)

            displacements_list.append(displacements)
            rotations_list.append(rotations)
            mass_list.append(mass)
            cg_list.append(cg)
            inertia_tensor_list.append(inertia_tensor)
            stresses_list.append(stresses)
            top_buckling_list.append(top_buckling)
            bot_buckling_list.append(bot_buckling)
        # top_buckling = m3l.Variable(name=f"{beam_name}_top_bkl", shape=(n-1, ), operation=self)
        # bot_buckling = m3l.Variable(name=f"{beam_name}_bot_bkl", shape=(n-1, ), operation=self)
        # stresses = m3l.Variable(name=f"{beam_name}_stress", shape=(n, 5), operation=self)
        struct_mass = m3l.Variable(name='mass', shape=(1,), operation=self)
        cg = m3l.Variable(name='cg_vector', shape=(3,), operation=self)
        inertia_tensor = m3l.Variable(name='inertia_tensor', shape=(3,3), operation=self)
        # mesh = beam_mesh.beam_nodes
        # beam_height = beam_mesh.height
        # beam_width = beam_mesh.width
        
        # # Gets information for naming/shapes
        # beam_name = list(self.parameters['beams'].keys())[0]   # this is only taking the first mesh added to the solver.
        # n = len(self.beams[beam_name]['nodes'])

        # self.arguments = {}
        # self.arguments[f'{beam_name}_width'] = beam_width
        # self.arguments[f'{beam_name}_height'] = beam_height
        # self.arguments[f'{beam_name}_ttop'] = t_top
        # self.arguments[f'{beam_name}_tbot'] = t_bot
        # self.arguments[f'{beam_name}_tweb'] = t_web
        # self.arguments[f'{beam_name}_mesh'] = mesh
        # if forces is not None:
        #     self.arguments[f'{beam_name}_forces'] = forces
        # if moments is not None:
        #     self.arguments[f'{beam_name}_moments'] = moments

        # # Create the M3L variables that are being output
        # displacements = m3l.Variable(name=f'{beam_name}_displacement', shape=mesh.shape, operation=self)
        # rotations = m3l.Variable(name=f'{beam_name}_rotation', shape=mesh.shape, operation=self)
        # mass = m3l.Variable(name='mass', shape=(1,), operation=self)
        # cg = m3l.Variable(name='cg_vector', shape=(3,), operation=self)
        # inertia_tensor = m3l.Variable(name='inertia_tensor', shape=(3,3), operation=self)

        # stresses = m3l.Variable(name=f"{beam_name}_stress", shape=(n, 5), operation=self)
        # top_buckling = m3l.Variable(name=f"{beam_name}_top_bkl", shape=(n-1, ), operation=self)
        # bot_buckling = m3l.Variable(name=f"{beam_name}_bot_bkl", shape=(n-1, ), operation=self)

        # outputs = EBBeamOutputs(
        #     displacements=displacements,
        #     rotations=rotations,
        #     mass=mass,
        #     cg_vector=cg,
        #     inertia_tensor=inertia_tensor,
        #     stresses=stresses,
        #     top_buckling=top_buckling,
        #     bot_buckling=bot_buckling,
        # )
            
        outputs = EBBeamOutputs(
            displacements=displacements_list,
            rotations=rotations_list,
            mass=mass_list,
            cg_vector=cg_list,
            inertia_tensor=inertia_tensor_list,
            stresses=stresses_list,
            top_buckling=top_buckling_list,
            bot_buckling=bot_buckling_list,
            struct_mass = struct_mass,
            cg_vector_not_list=cg,
            inertia_tensor_not_list=inertia_tensor,
            # stresses=stresses,
            # top_buckling=top_buckling,
            # bot_buckling=bot_buckling
        )
        
        return outputs


    

class EBBeamForces(m3l.ExplicitOperation):
    def initialize(self, kwargs):
        self.parameters.declare('name', types=str)
        self.parameters.declare('beams', default={})
        # self.parameters.declare('beam_mesh', types=LinearBeamMesh)
        self.parameters.declare('exclude_middle', default=False)

    def assign_attributes(self):
        self.beams = self.parameters['beams']
        # self.beam_mesh = self.parameters['beam_mesh']
        self.name = self.parameters['name']

    def compute(self) -> csdl.Model:
        beam_name = list(self.parameters['beams'].keys())[0]   # this is only taking the first mesh added to the solver.
        # beam_mesh = list(self.beam_mesh.parameters['meshes'].values())[0]   # this is only taking the first mesh added to the solver.
        nodal_forces = self.arguments['nodal_forces']
        exclude_middle = self.parameters['exclude_middle']

        csdl_model = csdl.Model()

        force_map = self.fmap(self.beam_mesh.value.reshape((-1,3)),
                              oml=self.nodal_forces_mesh.value.reshape((-1,3)),
                              exclude_middle=exclude_middle)

        flattened_nodal_forces_shape = (np.prod(nodal_forces.shape[:-1]), nodal_forces.shape[-1])
        nodal_forces_csdl = csdl_model.declare_variable(name='nodal_forces', shape=nodal_forces.shape)
        flattened_nodal_forces = csdl.reshape(nodal_forces_csdl, new_shape=flattened_nodal_forces_shape)
        force_map_csdl = csdl_model.create_input(f'nodal_to_{beam_name}_forces_map', val=force_map)
        flatenned_beam_mesh_forces = csdl.matmat(force_map_csdl, flattened_nodal_forces)
        output_shape = tuple(self.beam_mesh.shape[:-1]) + (nodal_forces.shape[-1],)
        beam_mesh_forces = csdl.reshape(flatenned_beam_mesh_forces, new_shape=output_shape)
        csdl_model.register_output(f'{beam_name}_forces', beam_mesh_forces)

        return csdl_model

    def evaluate(self, beam_mesh : m3l.Variable, nodal_forces:m3l.Variable, nodal_forces_mesh:am.MappedArray) -> m3l.Variable:
        '''
        Maps nodal forces from arbitrary locations to the mesh nodes.
        
        Parameters
        ----------
        nodal_forces : m3l.Variable
            The forces to be mapped to the mesh nodes.
        nodal_forces_mesh : m3l.Variable
            The mesh that the nodal forces are currently defined over.

        Returns
        -------
        mesh_forces : m3l.Variable
            The forces on the mesh.
        '''
        self.nodal_forces_mesh = nodal_forces_mesh
        self.beam_mesh = beam_mesh
        beam_name = list(self.parameters['beams'].keys())[0]   # this is only taking the first mesh added to the solver.
        # beam_mesh = list(self.beam_mesh.parameters['meshes'].values())[0]   # this is only taking the first mesh added to the solver.

        self.arguments = {'nodal_forces' : nodal_forces}
        output_shape = tuple(beam_mesh.shape[:-1]) + (nodal_forces.shape[-1],)
        beam_forces = m3l.Variable(name=f'{beam_name}_forces', shape=output_shape, operation=self)
        return beam_forces


    # def fmap(self, mesh, oml):
    #     # Fs = W*Fp
    #
    #     x, y = mesh.copy(), oml.copy()
    #     n, m = len(mesh), len(oml)
    #
    #     d = np.zeros((m,2))
    #     for i in range(m):
    #         dist = np.sum((x - y[i,:])**2, axis=1)
    #         d[i,:] = np.argsort(dist)[:2]
    #
    #     # create the weighting matrix:
    #     weights = np.zeros((n, m))
    #     for i in range(m):
    #         ia, ib = int(d[i,0]), int(d[i,1])
    #         a, b = x[ia,:], x[ib,:]
    #         p = y[i,:]
    #
    #         length = np.linalg.norm(b - a)
    #         norm = (b - a)/length
    #         t = np.dot(p - a, norm)
    #         # c is the closest point on the line segment (a,b) to point p:
    #         c =  a + t*norm
    #
    #         ac, bc = np.linalg.norm(c - a), np.linalg.norm(c - b)
    #         l = max(length, bc)
    #
    #         weights[ia, i] = (l - ac)/length
    #         weights[ib, i] = (l - bc)/length
    #
    #     return weights

    def fmap(self, mesh, oml, exclude_middle=False):
        from scipy.spatial.distance import cdist
        x, y = mesh.copy(), oml.copy()
        middle_index = int(x.shape[0] / 2)
        dist = cdist(y, x)
        weights = 1.0 / dist
        if exclude_middle:
            weights[:, middle_index] = np.zeros(y.shape[0])
        weights = weights.T
        weights /= weights.sum(axis=0)
        np.nan_to_num(weights, copy=False, nan=1.)
        return weights
    

class EBBeamMoments(m3l.ExplicitOperation):
    def initialize(self, kwargs):
        self.parameters.declare('mesh', default=None)
        self.parameters.declare('beams', default={})

    def compute(self, nodal_moments, nodal_moments_mesh):
        beam_name = list(self.parameters['beams'].keys())[0]   # this is only taking the first mesh added to the solver.
        mesh = list(self.parameters['mesh'].parameters['meshes'].values())[0]   # this is only taking the first mesh added to the solver.

        csdl_model = csdl.Model()

        force_map = self.fmap(mesh.value.reshape((-1,3)), oml=nodal_moments_mesh.value.reshape((-1,3)))

        nodal_moments = csdl_model.declare_variable(name='nodal_moments', shape=nodal_moments.shape)
        moment_map_csdl = csdl_model.create_input(f'nodal_to_{beam_name}_moments_map', val=force_map)
        beam_moments = csdl.matmat(moment_map_csdl, nodal_moments)
        csdl_model.register_output(f'{beam_name}_moments', beam_moments)

        return csdl_model


    def evaluate(self, nodal_moments:m3l.Variable, nodal_moments_mesh:am.MappedArray) -> m3l.Variable:
        '''
        Maps nodal moments from arbitrary locations to the mesh nodes.
        
        Parameters
        ----------
        nodal_moments : m3l.Variable
            The moments to be mapped to the mesh nodes.
        nodal_moments_mesh : m3l.Variable
            The mesh that the nodal moments are currently defined over.

        Returns
        -------
        mesh_moments : m3l.Variable
            The moments on the mesh.
        '''
        operation_csdl = self.compute(nodal_moments=nodal_moments, nodal_moments_mesh=nodal_moments_mesh)

        beam_name = list(self.parameters['beams'].keys())[0]   # this is only taking the first mesh added to the solver.
        mesh = list(self.parameters['mesh'].parameters['meshes'].values())[0]   # this is only taking the first mesh added to the solver.

        arguments = {'nodal_moments': nodal_moments}
        moment_map_operation = m3l.CSDLOperation(name='ebbeam_moment_map', arguments=arguments, operation_csdl=operation_csdl)
        beam_moments = m3l.Variable(name=f'{beam_name}_moments', shape=mesh.shape, operation=moment_map_operation)
        return beam_moments
    

    def fmap(self, mesh, oml):
        # Fs = W*Fp

        x, y = mesh.copy(), oml.copy()
        n, m = len(mesh), len(oml)

        d = np.zeros((m,2))
        for i in range(m):
            dist = np.sum((x - y[i,:])**2, axis=1)
            d[i,:] = np.argsort(dist)[:2]

        # create the weighting matrix:
        weights = np.zeros((n, m))
        for i in range(m):
            ia, ib = int(d[i,0]), int(d[i,1])
            a, b = x[ia,:], x[ib,:]
            p = y[i,:]

            length = np.linalg.norm(b - a)
            norm = (b - a)/length
            t = np.dot(p - a, norm)
            # c is the closest point on the line segment (a,b) to point p:
            c =  a + t*norm

            ac, bc = np.linalg.norm(c - a), np.linalg.norm(c - b)
            l = max(length, bc)
            
            weights[ia, i] = (l - ac)/length
            weights[ib, i] = (l - bc)/length

        return weights


class EBBeamNodalDisplacements(m3l.ExplicitOperation):
    def initialize(self, kwargs):
        self.parameters.declare('component')
        self.parameters.declare('beams', default={})
        self.parameters.declare('beam_mesh', types=LinearBeamMesh)

    def assign_attributes(self):
        self.component = self.parameters['component']
        self.beams = self.parameters['beams']
        self.beam_mesh = self.parameters['beam_mesh']

    def compute(self) -> csdl.Model:
        beam_name = list(self.parameters['beams'].keys())[0]   # this is only taking the first mesh added to the solver.
        mesh = list(self.beam_mesh.parameters['meshes'].values())[0]   # this is only taking the first mesh added to the solver.
        nodal_displacements_mesh = self.nodal_displacements_mesh
        beam_displacements = self.arguments[f'{beam_name}_displacement']

        csdl_model = csdl.Model()

        displacement_map = self.umap(mesh.value.reshape((-1,3)), oml=nodal_displacements_mesh.value.reshape((-1,3)))

        beam_displacements_csdl = csdl_model.declare_variable(name=f'{beam_name}_displacement', shape=beam_displacements.shape)
        displacement_map_csdl = csdl_model.create_input(f'{beam_name}_displacements_to_nodal_displacements', val=displacement_map)
        nodal_displacements = csdl.matmat(displacement_map_csdl, beam_displacements_csdl)
        csdl_model.register_output(f'{beam_name}_nodal_displacement', nodal_displacements)

        return csdl_model

    def evaluate(self, beam_displacements:m3l.Variable, nodal_displacements_mesh:am.MappedArray) -> m3l.Variable:
        '''
        Maps nodal forces and moments from arbitrary locations to the mesh nodes.
        
        Parameters
        ----------
        beam_displacements : m3l.Variable
            The displacements to be mapped from the beam mesh to the desired mesh.
        nodal_displacements_mesh : m3l.Variable
            The mesh to evaluate the displacements over.

        Returns
        -------
        nodal_displacements : m3l.Variable
            The displacements on the given nodal displacements mesh.
        '''

        beam_name = list(self.beams.keys())[0]   # this is only taking the first mesh added to the solver.

        self.name = f'{self.component.name}_eb_beam_displacement_map'
        self.arguments = {f'{beam_name}_displacement': beam_displacements}
        self.nodal_displacements_mesh = nodal_displacements_mesh

        nodal_displacements = m3l.Variable(name=f'{beam_name}_nodal_displacement', shape=nodal_displacements_mesh.shape, operation=self)
        return nodal_displacements


    def umap(self, mesh, oml):
        # Up = W*Us

        x, y = mesh.copy(), oml.copy()
        n, m = len(mesh), len(oml)

        d = np.zeros((m,2))
        for i in range(m):
            dist = np.sum((x - y[i,:])**2, axis=1)
            d[i,:] = np.argsort(dist)[:2]

        # create the weighting matrix:
        weights = np.zeros((m,n))
        for i in range(m):
            ia, ib = int(d[i,0]), int(d[i,1])
            a, b = x[ia,:], x[ib,:]
            p = y[i,:]

            length = np.linalg.norm(b - a)
            norm = (b - a)/length
            t = np.dot(p - a, norm)
            # c is the closest point on the line segment (a,b) to point p:
            c =  a + t*norm

            ac, bc = np.linalg.norm(c - a), np.linalg.norm(c - b)
            l = max(length, bc)
            
            weights[i, ia] = (l - ac)/length
            weights[i, ib] = (l - bc)/length

        return weights
    
# class EBBeamMass(m3l.ExplicitOperation):
#     def initialize(self, kwargs):
#         self.parameters.declare('elements')
#         self.parameters.declare('element_density_list')

class BeamM3LStrain(m3l.ExplicitOperation):
    def compute():
        pass

    def compute_derivatives(): # optional
        pass

class BeamM3LStress(m3l.ExplicitOperation):
    def compute():
        pass

    def compute_derivatives(): # optional
        pass








class LinearBeamMesh(m3l.ExplicitOperation):
    def initialize(self, kwargs):
        self.parameters.declare('name', types=str, default='linear_beam_mesh')
        self.parameters.declare('meshes', types=dict)
        self.parameters.declare('mesh_units', default='m')



class LinearBeamCSDL(csdl.Model):
    def initialize(self):
        self.parameters.declare('beams')
        self.parameters.declare('bounds')
        self.parameters.declare('joints')
        self.parameters.declare('mesh_units')
    
    def define(self):
        beams = self.parameters['beams']
        bounds = self.parameters['bounds']
        joints = self.parameters['joints']
        mesh_units = self.parameters['mesh_units']

        # solve the beam group:
        self.add(Aframe(beams=beams, bounds=bounds, joints=joints, mesh_units=mesh_units), name='Aframe')