import numpy as np
import scipy.sparse as sps
import array_mapper as am
import os
from pathlib import Path
import vedo
import pickle
from caddee.core.caddee_core.system_representation.system_primitive.system_primitive import SystemPrimitive
from caddee.core.caddee_core.system_representation.utils.io.step_io import read_openvsp_stp, write_step, read_gmsh_stp
from caddee.core.caddee_core.system_representation.utils.io.iges_io import read_iges, write_iges
from caddee import IMPORTS_FILES_FOLDER
from caddee import PROJECTIONS_FOLDER

class SpatialRepresentation:
    '''
    The description of the physical system.

    Parameters
    ----------


    '''
    def __init__(self, primitives:dict={}, primitive_indices:dict={}) -> None:
        self.primitives = primitives.copy()     # NOTE: This is one of those "I can't believe it" moments.
        self.primitive_indices = primitive_indices.copy()
        self.control_points = None  # Will be instantiated during assemble()
        self.file_name = ''
        self.inputs = {}
        self.outputs = {}

    
    def get_primitives(self, search_names=[]):
        '''
        Returns the primtiive objects that include the search name(s) in the primitive name.
        '''
        primitives = {}
        for primitive_name in self.primitives.keys():
            for search_name in search_names:
                if search_name in primitive_name:
                    primitives[primitive_name] = self.primitives[primitive_name]
                    break

        return primitives

    def get_geometry_primitives(self, search_names=[]):
        '''
        Returns the geometry primitive objects that include the search name(s) in the primitive name.
        '''
        primitives = self.get_primitives(search_names=search_names)
        geometry_primitives = {}
        for primitive in list(primitives.values()):
            geometry_primitive = primitive.geometry_primitive
            geometry_primitives[primitive.name] = geometry_primitive

        return geometry_primitives


    def project(self, points:np.ndarray, targets:list=None, direction:np.ndarray=None,
                grid_search_n:int=25, max_iterations=100, properties:list=['geometry'],
                offset:np.ndarray=None, plot:bool=False, comp_name:str='', force_reprojection:bool=False):
        '''
        Projects points onto the system.

        Parameters
        -----------
        points : {np.ndarray, am.MappedArray}
            The points to be projected onto the system.
        targets : list, optional
            The list of primitives to project onto.
        direction : {np.ndarray, am.MappedArray}, optional
            An axis for perfoming projection along an axis. The projection will return the closest point to the axis.
        grid_search_n : int, optional
            The resolution of the grid search prior to the Newton iteration for solving the optimization problem.
        max_iterations : int, optional
            The maximum number of iterations for the Newton iteration.
        properties : list
            The list of properties to be returned (in order) {geometry, parametric_coordinates, (material_name, array_of_properties),...}
        offset : np.ndarray
            An offset to apply after the parametric evaluation of the projection. TODO Fix offset!!
        plot : bool
            A boolean on whether or not to plot the projection result.
        '''
        #  TODO Consider parallelizing using Numba, or using the FFD method or in Cython.

        fn = os.path.basename(self.file_name)
        fn_wo_ext = f"{fn[:fn.rindex('.')]}_{comp_name}"

        if type(points) is am.MappedArray:
                points = points.value
        if type(direction) is am.MappedArray:
                direction = direction.value
        
        if direction is not None:
            projections = PROJECTIONS_FOLDER / f'{fn_wo_ext}_points_{str(round(np.linalg.norm(points), 4))}_direction_{str(round(np.linalg.norm(direction)))}_gridsearch_{grid_search_n}.pickle'
        else:
            projections = PROJECTIONS_FOLDER / f'{fn_wo_ext}_points_{str(round(np.linalg.norm(points), 4))}_gridsearch_{grid_search_n}.pickle'

        my_file = Path(projections) 
        
        if my_file.is_file():
            with open(projections, 'rb') as f:
                projections_dict = pickle.load(f)
            
            new_projections = False
            if np.array_equiv(points, projections_dict['function_input']['points']):
                pass
            else:
                new_projections = True
          
            if projections_dict['function_input']['targets'] == targets:
                pass
            else:
                new_projections = True
          
            if np.array_equiv(projections_dict['function_input']['direction'], direction):
                pass
            else:
                new_projections = True
          
            if projections_dict['function_input']['grid_search_n'] == grid_search_n:
                pass
            else:
                new_projections = True
          
            if projections_dict['function_input']['max_iterations'] == max_iterations:
                pass
            else:
                new_projections = True
            
            if projections_dict['function_input']['properties'] == properties:
                pass
            else:
                new_projections = True

            if projections_dict['function_input']['properties'] == properties:
                pass
            else:
                new_projections = True

            if projections_dict['function_input']['offset'] == offset:
                pass
            else:
                new_projections = True

            if new_projections or force_reprojection:
                print(f"Stored projections do not exist for component '{comp_name}' contained in file '{fn}'. Proceed with projection algorithm.")
                targets_list = []
                for target in targets:
                    targets_list.append(target)
                data_dict = dict()
                data_dict['function_input'] = {
                    'points' : points,
                    'targets' : targets_list,
                    'direction' : direction,
                    'grid_search_n' : grid_search_n,
                    'max_iterations' : max_iterations,
                    'properties' :  properties,
                    'offset' : offset,
                }

                # Proceed with code 
                if targets is None:
                    targets = list(self.primitives.values())
                elif type(targets) is dict:
                    pass    # get objects is list
                elif type(targets) is list:
                    for i, target in enumerate(targets):
                        if isinstance(target, str):
                            targets[i] = self.primitives[target]

                if type(points) is am.MappedArray:
                    points = points.value
                if type(direction) is am.MappedArray:
                    direction = direction.value

                if len(points.shape) == 1:
                    points = points.reshape((1, -1))    # Last axis is reserved for dimensionality of physical space
                
                num_targets = len(targets)
                projected_points_on_each_target = []
                target_names = []
                # Project all points onto each target
                for target in targets:   # TODO Parallelize this for loop
                    target_names.append(target.name)
                    target_projected_points = target.project(points=points, direction=direction, grid_search_n=grid_search_n,
                            max_iter=max_iterations, properties=['geometry', 'parametric_coordinates'])
                            # properties are not passed in here because we NEED geometry
                    projected_points_on_each_target.append(target_projected_points)

                projected_points_on_each_target_numpy = np.zeros(tuple((num_targets,)) + points.shape)
                for i in range(num_targets):
                        projected_points_on_each_target_numpy[i] = projected_points_on_each_target[i]['geometry'].value

                # Compare results across targets to keep best result
                distances = np.linalg.norm(projected_points_on_each_target_numpy - points, axis=-1)   # Computes norm across spatial axis
                closest_surfaces_indices = np.argmin(distances, axis=0) # Take argmin across surfaces
                if len(points.shape) == 1:
                    num_points = 1
                else:
                    num_points = np.cumprod(points.shape[:-1])[-1]
                flattened_surface_indices = closest_surfaces_indices.flatten()
                # num_control_points = np.cumprod(self.control_points.shape[:-1])[-1]
                # linear_map = sps.lil_array((num_points, num_control_points))
                projection_receiving_primitives = []
                projection_outputs = {}
                # for i in range(num_points): # for each point, assign the closest the projection result
                #     target_index = flattened_surface_indices[i]
                #     receiving_target = targets[target_index]
                #     if receiving_target not in projection_receiving_primitives:
                #         projection_receiving_primitives.append(receiving_target)
                #     # receiving_target_control_point_indices = self.primitive_indices[receiving_target.name]
                #     # point_map_on_receiving_target = projected_points_on_each_target[target_index].linear_map[i,:]
                #     # linear_map[i, receiving_target_control_point_indices] = point_map_on_receiving_target

                # for i in range(num_points):
                #     target_index = flattened_surface_indices[i]
                #     receiving_target = targets[target_index]
                #     if receiving_target not in projection_receiving_primitives:
                #         projection_receiving_primitives.append(receiving_target)

                for property in properties:
                    if property == 'parametric_coordinates':
                        nodes_parametric = []
                        for i in range(num_points):
                            target_index = flattened_surface_indices[i]
                            receiving_target_name = target_names[target_index]
                            receiving_target = targets[target_index]
                            u_coord = projected_points_on_each_target[target_index]['parametric_coordinates'][0][i]
                            v_coord = projected_points_on_each_target[target_index]['parametric_coordinates'][1][i]
                            node_parametric_coordinates = np.array([u_coord, v_coord]).reshape(1,2)
                            nodes_parametric.append((receiving_target_name, node_parametric_coordinates))
                        projection_outputs[property] = nodes_parametric
                    else:
                        num_control_points = np.cumprod(self.control_points[property].shape[:-1])[-1]
                        linear_map = sps.lil_array((num_points, num_control_points))
                        for i in range(num_points):
                            target_index = flattened_surface_indices[i]
                            receiving_target = targets[target_index]
                            receiving_target_control_point_indices = self.primitive_indices[receiving_target.name][property]
                            point_parametric_coordinates = projected_points_on_each_target[target_index]['parametric_coordinates']
                            if property == 'geometry':
                                point_map_on_receiving_target = receiving_target.geometry_primitive.compute_evaluation_map(u_vec=np.array([point_parametric_coordinates[0][i]]), 
                                                                                                    v_vec=np.array([point_parametric_coordinates[1][i]]))    
                            else:
                                point_map_on_receiving_target = receiving_target.material_primitives[property].compute_evaluation_map(u_vec=np.array([point_parametric_coordinates[0][i]]), 
                                                                                                    v_vec=np.array([point_parametric_coordinates[1][i]]))
                            linear_map[i, receiving_target_control_point_indices] = point_map_on_receiving_target

                        property_shape = points.shape[:-1] + (self.control_points[property].shape[-1],)
                        property_mapped_array = am.array(self.control_points[property], linear_map=linear_map.tocsc(), offset=offset, shape=property_shape)
                        projection_outputs[property] = property_mapped_array

                data_dict['projection_outputs'] = projection_outputs
                if direction is not None:
                    save_file = PROJECTIONS_FOLDER / f'{fn_wo_ext}_points_{str(round(np.linalg.norm(points), 4))}_direction_{str(round(np.linalg.norm(direction)))}_gridsearch_{grid_search_n}.pickle'
                else:
                    save_file = PROJECTIONS_FOLDER / f'{fn_wo_ext}_points_{str(round(np.linalg.norm(points), 4))}_gridsearch_{grid_search_n}.pickle'
                with open(save_file, 'wb+') as handle:
                    pickle.dump(data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
            else:
                print(f"Stored projections exist for component '{comp_name}' contained in file '{fn}'.")
                projection_outputs = projections_dict['projection_outputs']
                if len(points.shape) == 1:
                    num_points = 1
                else:
                    num_points = np.cumprod(points.shape[:-1])[-1]

        else:
            print(f"Stored projections do not exist for component '{comp_name}' contained in file '{fn}'. Proceed with projection algorithm.")
            targets_list = []
            for target in targets:
                targets_list.append(target)
            data_dict = dict()
            data_dict['function_input'] = {
                'points' : points,
                'targets' : targets_list,
                'direction' : direction,
                'grid_search_n' : grid_search_n,
                'max_iterations' : max_iterations,
                'properties' :  properties,
                'offset' : offset,
            }

            # Proceed with code 
            if targets is None:
                targets = list(self.primitives.values())
            elif type(targets) is dict:
                pass    # get objects is list
            elif type(targets) is list:
                for i, target in enumerate(targets):
                    if isinstance(target, str):
                        targets[i] = self.primitives[target]

            if type(points) is am.MappedArray:
                points = points.value
            if type(direction) is am.MappedArray:
                direction = direction.value

            if len(points.shape) == 1:
                points = points.reshape((1, -1))    # Last axis is reserved for dimensionality of physical space
            
            num_targets = len(targets)
            projected_points_on_each_target = []
            # Project all points onto each target
            target_names = []
            # Project all points onto each target
            for target in targets:   # TODO Parallelize this for loop
                target_names.append(target.name)
                target_projected_points = target.project(points=points, direction=direction, grid_search_n=grid_search_n,
                        max_iter=max_iterations, properties=['geometry', 'parametric_coordinates'])
                        # properties are not passed in here because we NEED geometry
                projected_points_on_each_target.append(target_projected_points)

            projected_points_on_each_target_numpy = np.zeros(tuple((num_targets,)) + points.shape)
            for i in range(num_targets):
                    projected_points_on_each_target_numpy[i] = projected_points_on_each_target[i]['geometry'].value

            # Compare results across targets to keep best result
            distances = np.linalg.norm(projected_points_on_each_target_numpy - points, axis=-1)   # Computes norm across spatial axis
            closest_surfaces_indices = np.argmin(distances, axis=0) # Take argmin across surfaces
            if len(points.shape) == 1:
                num_points = 1
            else:
                num_points = np.cumprod(points.shape[:-1])[-1]
            flattened_surface_indices = closest_surfaces_indices.flatten()
            # num_control_points = np.cumprod(self.control_points.shape[:-1])[-1]
            # linear_map = sps.lil_array((num_points, num_control_points))
            projection_receiving_primitives = []
            projection_outputs = {}
            # for i in range(num_points): # for each point, assign the closest the projection result
            #     target_index = flattened_surface_indices[i]
            #     receiving_target = targets[target_index]
            #     if receiving_target not in projection_receiving_primitives:
            #         projection_receiving_primitives.append(receiving_target)
            #     # receiving_target_control_point_indices = self.primitive_indices[receiving_target.name]
            #     # point_map_on_receiving_target = projected_points_on_each_target[target_index].linear_map[i,:]
            #     # linear_map[i, receiving_target_control_point_indices] = point_map_on_receiving_target

            # for i in range(num_points):
            #     target_index = flattened_surface_indices[i]
            #     receiving_target = targets[target_index]
            #     if receiving_target not in projection_receiving_primitives:
            #         projection_receiving_primitives.append(receiving_target)

            for property in properties:
                if property == 'parametric_coordinates':
                    nodes_parametric = []
                    for i in range(num_points):
                        target_index = flattened_surface_indices[i]
                        receiving_target_name = target_names[target_index]
                        receiving_target = targets[target_index]
                        u_coord = projected_points_on_each_target[target_index]['parametric_coordinates'][0][i]
                        v_coord = projected_points_on_each_target[target_index]['parametric_coordinates'][1][i]
                        node_parametric_coordinates = np.array([u_coord, v_coord]).reshape(1,2)
                        nodes_parametric.append((receiving_target_name, node_parametric_coordinates))
                    projection_outputs[property] = nodes_parametric
                else:
                    num_control_points = np.cumprod(self.control_points[property].shape[:-1])[-1]
                    linear_map = sps.lil_array((num_points, num_control_points))
                    for i in range(num_points):
                        target_index = flattened_surface_indices[i]
                        receiving_target = targets[target_index]
                        receiving_target_control_point_indices = self.primitive_indices[receiving_target.name][property]
                        point_parametric_coordinates = projected_points_on_each_target[target_index]['parametric_coordinates']
                        if property == 'geometry':
                            point_map_on_receiving_target = receiving_target.geometry_primitive.compute_evaluation_map(u_vec=np.array([point_parametric_coordinates[0][i]]), 
                                                                                                v_vec=np.array([point_parametric_coordinates[1][i]]))
                        else:
                            point_map_on_receiving_target = receiving_target.material_primitives[property].compute_evaluation_map(u_vec=np.array([point_parametric_coordinates[0][i]]), 
                                                                                                v_vec=np.array([point_parametric_coordinates[1][i]]))
                        linear_map[i, receiving_target_control_point_indices] = point_map_on_receiving_target

                    property_shape = points.shape[:-1] + (self.control_points[property].shape[-1],)
                    property_mapped_array = am.array(self.control_points[property], linear_map=linear_map.tocsc(), offset=offset, shape=property_shape)
                    projection_outputs[property] = property_mapped_array

            data_dict['projection_outputs'] = projection_outputs
            if direction is not None:
                save_file = PROJECTIONS_FOLDER / f'{fn_wo_ext}_points_{str(round(np.linalg.norm(points), 4))}_direction_{str(round(np.linalg.norm(direction)))}_gridsearch_{grid_search_n}.pickle'
            else:
                save_file = PROJECTIONS_FOLDER / f'{fn_wo_ext}_points_{str(round(np.linalg.norm(points), 4))}_gridsearch_{grid_search_n}.pickle'
            with open(save_file, 'wb+') as handle:
                pickle.dump(data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        projection_receiving_primitives = list(targets)

        # linear_map = linear_map.tocsc()
        # projected_points = am.array(self.control_points, linear_map=linear_map, offset=offset, shape=points.shape)

        if plot:
            # Plot the surfaces that are projected onto
            plotter = vedo.Plotter()
            primitive_meshes = self.plot(primitives=projection_receiving_primitives, opacity=0.25, show=False)
            # Plot 
            plotting_points = []
            # TODO This will break if geometry is not one of the properties. Fix this.
            flattened_projected_points = (projection_outputs['geometry'].value).reshape((num_points, -1)) # last axis usually has length 3 for x,y,z
            plotting_primitive_control_points = vedo.Points(flattened_projected_points, r=12, c='#00C6D7')  # TODO make this (1,3) instead of (3,)
            plotting_points.append(plotting_primitive_control_points)
            plotter.show(primitive_meshes, plotting_points, 'Projected Points', axes=1, viewup="z", interactive=True)

        if len(projection_outputs) == 1:
            return list(projection_outputs.values())[0]
        else:
            return projection_outputs


    def add_input(self, name, quantity, val=None):
        '''
        Adds an input to the mechanical structure. The design geometry optimization will manipulate
        the mechanical structure in order to achieve this input.
        '''
        self.inputs[name] = (quantity, val)
    
    def add_output(self, name, quantity):
        '''
        Adds an output to the system configuration. The system configuration will recalculate this output each iteration.
        '''
        self.outputs[name] = quantity


    def import_file(self, file_name : str, file_path : str = None):
        '''
        Imports geometry primitives from a file.

        Parameters
        ----------
        file_name : str
            The name of the file (with path) that containts the geometric information.
        
        '''
        self.file_name = file_name

        fn = os.path.basename(file_name)
        fn_wo_ext = fn[:fn.rindex('.')]
        if file_path is None:
            control_points = IMPORTS_FILES_FOLDER / f'{fn_wo_ext}_control_points.pickle'
            primitive_indices = IMPORTS_FILES_FOLDER / f'{fn_wo_ext}_primitive_indices.pickle'
            primitives = IMPORTS_FILES_FOLDER / f'{fn_wo_ext}_primitives.pickle'
        else:
            control_points = file_path + f'/{fn_wo_ext}_control_points.pickle'
            primitive_indices = file_path + f'/{fn_wo_ext}_primitive_indices.pickle'
            primitives = file_path + f'/{fn_wo_ext}_primitives.pickle'
        my_file = Path(control_points) 
        if my_file.is_file():
            with open(primitives, 'rb') as f:
                self.primitives = pickle.load(f)

            with open(primitive_indices, 'rb') as f:
                self.primitive_indices = pickle.load(f)

            with open(control_points, 'rb') as f:
                self.control_points = pickle.load(f)

        else:
            file_name = str(file_name)
            if (file_name[-4:].lower() == '.stp') or (file_name[-5:].lower() == '.step'):
                with open(file_name, 'r') as f:
                    if 'CASCADE' in f.read():  # Not sure, could be another string to identify
                        self.read_gmsh_stp(file_name)
                    else: 
                        self.read_openvsp_stp(file_name)
            elif (file_name[-4:].lower() == '.igs') or (file_name[-5:].lower() == '.iges'):
                raise NotImplementedError
                # self.read_iges(file_name) #TODO
            else:
                print("Please input an iges file or a stp file from openvsp.")

            self.assemble()
            save_file_name = os.path.basename(file_name)
            filename_without_ext = save_file_name[:save_file_name.rindex('.')]
            if file_path is None:
                with open(IMPORTS_FILES_FOLDER / f'{filename_without_ext}_control_points.pickle', 'wb+') as handle:
                    pickle.dump(self.control_points, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    # np.save(f, self.control_points)
                with open(IMPORTS_FILES_FOLDER / f'{filename_without_ext}_primitive_indices.pickle', 'wb+') as handle:
                    pickle.dump(self.primitive_indices, handle, protocol=pickle.HIGHEST_PROTOCOL)
                with open(IMPORTS_FILES_FOLDER / f'{filename_without_ext}_primitives.pickle', 'wb+') as handle:
                    pickle.dump(self.primitives, handle, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                with open(file_path + f'/{filename_without_ext}_control_points.pickle', 'wb+') as handle:
                    pickle.dump(self.control_points, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    # np.save(f, self.control_points)
                with open(file_path + f'/{filename_without_ext}_primitive_indices.pickle', 'wb+') as handle:
                    pickle.dump(self.primitive_indices, handle, protocol=pickle.HIGHEST_PROTOCOL)
                with open(file_path + f'/{filename_without_ext}_primitives.pickle', 'wb+') as handle:
                    pickle.dump(self.primitives, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def refit_geometry(self, num_control_points:int=25, fit_resolution:int=50, only_non_differentiable:bool=False, file_name=None, file_path=None):
        import caddee.core.primitives.bsplines.bspline_functions as mfd  # lsdo_manifolds

        if file_name is not None:
            fn = os.path.basename(file_name)
            fn_wo_ext = fn[:fn.rindex('.')]
            if file_path is None:
                control_points = IMPORTS_FILES_FOLDER / f'{fn_wo_ext}_control_points_{num_control_points}_{fit_resolution}.pickle'
                primitive_indices = IMPORTS_FILES_FOLDER / f'{fn_wo_ext}_primitive_indices_{num_control_points}_{fit_resolution}.pickle'
                primitives = IMPORTS_FILES_FOLDER / f'{fn_wo_ext}_primitives_{num_control_points}_{fit_resolution}.pickle'
            else:
                control_points = file_path + f'/{fn_wo_ext}_control_points_{num_control_points}_{fit_resolution}.pickle'
                primitive_indices = file_path + f'/{fn_wo_ext}_primitive_indices_{num_control_points}_{fit_resolution}.pickle'
                primitives = file_path + f'/{fn_wo_ext}_primitives_{num_control_points}_{fit_resolution}.pickle'
            my_file = Path(control_points) 
        if file_name is not None and my_file.is_file():
            with open(primitives, 'rb') as f:
                self.primitives = pickle.load(f)

            with open(primitive_indices, 'rb') as f:
                self.primitive_indices = pickle.load(f)

            with open(control_points, 'rb') as f:
                self.control_points = pickle.load(f)

        else:
            for primitive_name, primitive in self.primitives.items():
                i_should_refit = True
                if only_non_differentiable:
                    raise Warning("Refitting all surfaces for now regardless of differentiability.")
                    # if differentiable:
                    #     i_should_refit = False
                if i_should_refit:
                    self.primitives[primitive_name].geometry_primitive = \
                        mfd.refit_bspline(bspline=primitive.geometry_primitive, name=primitive_name, \
                                        num_control_points=(num_control_points,), fit_resolution=(fit_resolution,))
                    self.primitives[primitive_name].assemble()
            self.assemble()
            if file_name is not None:
                save_file_name = os.path.basename(file_name)
                filename_without_ext = save_file_name[:save_file_name.rindex('.')]
                if file_path is None:
                    with open(IMPORTS_FILES_FOLDER / f'{filename_without_ext}_control_points_{num_control_points}_{fit_resolution}.pickle', 'wb+') as handle:                    pickle.dump(self.control_points, handle, protocol=pickle.HIGHEST_PROTOCOL)
                        # np.save(f, self.control_points)
                    with open(IMPORTS_FILES_FOLDER / f'{filename_without_ext}_primitive_indices_{num_control_points}_{fit_resolution}.pickle', 'wb+') as handle:
                        pickle.dump(self.primitive_indices, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    with open(IMPORTS_FILES_FOLDER / f'{filename_without_ext}_primitives_{num_control_points}_{fit_resolution}.pickle', 'wb+') as handle:
                        pickle.dump(self.primitives, handle, protocol=pickle.HIGHEST_PROTOCOL)
                else:
                    with open(file_path + f'/{filename_without_ext}_control_points_{num_control_points}_{fit_resolution}.pickle', 'wb+') as handle:                    pickle.dump(self.control_points, handle, protocol=pickle.HIGHEST_PROTOCOL)
                        # np.save(f, self.control_points)
                    with open(file_path + f'/{filename_without_ext}_primitive_indices_{num_control_points}_{fit_resolution}.pickle', 'wb+') as handle:
                        pickle.dump(self.primitive_indices, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    with open(file_path + f'/{filename_without_ext}_primitives_{num_control_points}_{fit_resolution}.pickle', 'wb+') as handle:
                        pickle.dump(self.primitives, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def read_openvsp_stp(self, file_name):
        bsplines = read_openvsp_stp(file_name)
        imported_primitives = {}
        for bspline in list(bsplines.values()):
            primitive = SystemPrimitive(name=bspline.name, geometry_primitive=bspline)
            imported_primitives[primitive.name] = primitive
        self.primitives.update(imported_primitives)
        return imported_primitives

    def read_gmsh_stp(self, file_name):
        read_gmsh_stp(self, file_name)

    def read_iges(self,file_name):
        read_iges(self, file_name)

    def write_step(self, file_name, plot=False):
        write_step(self, file_name, plot)

    def write_iges(self, file_name, plot = False):
        write_iges(self, file_name, plot)

    '''
    Collects the primitives into a collectivized format.
    '''
    def assemble(self):
        self.control_points = {}
        starting_indices = {}

        for primitive in list(self.primitives.values()):
            # Adding indices for properties that don't already have starting indices to avoid KeyError
            for property_type in list(primitive.control_points.keys()):
                if property_type not in starting_indices:
                    starting_indices[property_type] = 0

            self.primitive_indices[primitive.name] = {}

            # Adding primitive control points to mechanical structure control points
            for property_type in list(primitive.control_points.keys()):
                primitive_property_control_points = primitive.control_points[property_type]
                primitive_num_control_points = np.cumprod(primitive_property_control_points.shape[:-1])[-1] 
                #NOTE: control points should always be (np,ndim)
                ending_index = starting_indices[property_type] + primitive_num_control_points
                self.primitive_indices[primitive.name][property_type] = np.arange(starting_indices[property_type], ending_index)
                starting_indices[property_type] = ending_index
                if property_type in self.control_points:
                    self.control_points[property_type] = np.vstack((self.control_points[property_type], 
                                                                    primitive_property_control_points.reshape((primitive_num_control_points,-1))))
                else:
                    self.control_points[property_type] = primitive_property_control_points.reshape((primitive_num_control_points, -1))

        # starting_index = 0
        # for primitive in list(self.primitives.values()):
        #     primitive_control_points = primitive.control_points
        #     primitive_num_control_points = np.cumprod(primitive_control_points.shape[:-1])[-1]    # control points should always be (np,ndim)
        #     ending_index = starting_index + primitive_num_control_points
        #     self.primitive_indices[primitive.name] = np.arange(starting_index, ending_index)
        #     starting_index = ending_index
        #     if self.control_points is None:
        #         self.control_points = primitive_control_points.reshape((primitive_num_control_points, -1))
        #     else:
        #         self.control_points = np.vstack((self.control_points, primitive_control_points.reshape((primitive_num_control_points, -1))))

    def update(self, updated_control_points:np.ndarray, primitive_names:list=['all']):
        '''
        Updates the control points of the mechanical structure or a portion of the mechanical structure.

        Parameters
        -----------
        updated_control_points : {np.ndarray, dict}
            The array or dictionary of new control points for the mechanical structure.
            An array updates geometric control points while a dictionary of the form
            {property_name:array_of_values} can be used to update any general set of properties.
        primitive_names : list=['all'], optional
            The list of primitives to be updated with the specified values.
        '''
        if primitive_names == ['all']:
            primitive_names = list(self.primitives.keys())
        
        if type(updated_control_points) is np.ndarray or \
            type(updated_control_points) is am.MappedArray:
            starting_index = 0
            for primitive_name in primitive_names:
                primitive = self.primitives[primitive_name]
                property_name = 'geometry'
                indices = self.primitive_indices[primitive_name][property_name]
                ending_index = starting_index + len(indices)
                self.control_points[property_name][indices] = updated_control_points[starting_index:ending_index]

                self.primitives[primitive_name].geometry_primitive.control_points = \
                    updated_control_points[starting_index:ending_index]

                starting_index = ending_index
        elif type(updated_control_points) is dict:
            starting_indices = {}
            for primitive_name in primitive_names:
                primitive = self.primitives[primitive_name]
                for property_name in list(updated_control_points.keys()):
                    indices = self.primitive_indices[primitive_name][property_name]
                    ending_index = starting_indices[property_name] + len(indices)
                    self.control_points[property_name][indices] = \
                        updated_control_points[property_name][starting_indices[property_name]:ending_index]

                    if property_name == 'geometry':
                        self.primitives[primitive_name].geometry_primitive.control_points = \
                            updated_control_points[property_name][starting_indices[property_name]:ending_index]
                    else:
                        self.primitives[primitive_name].material_primitives[property_name].control_points = \
                            updated_control_points[property_name][starting_indices[property_name]:ending_index]

                    starting_indices[property_name] = ending_index
        else:
            raise Exception("When updating, please pass in an array (for purely geometric update) \
                            or dictionary (for general property update)")


    '''
    Completes all setup steps. This may be deleted later.
    '''
    def setup(self):
        pass

    
    def evaluate(self):
        '''
        Evaluates the object. I had a reason for this but I need to remember what it will evaluate. (Actuations? Geo DV?).
        Actuate should probably be a method called "actuate" or something to that effect if it's a method here.
        After some thought, I think it would make sense for this class to not have an evaluate method since the Geometry
        is a true object that is a container for information and not really a model. Any methods should reflect what the object
        is doing.

        NOTE: I think (or at least I now think) that this evaluate is for meshes and outputs as a whole.
        '''
        raise Exception("Geometry.evaluate() is not implemented.")


    def plot_meshes(self, meshes:list, mesh_plot_types:list=['wireframe'], mesh_opacity:float=1., mesh_color:str='#F5F0E6',
                primitives:list=[], primitives_plot_types:list=['mesh'], primitives_opacity:float=0.25, primitives_color:str='#00629B',
                primitives_surface_texture:str="", additional_plotting_elements:list=[], camera:dict=None, show:bool=True):
        '''
        Plots a mesh over the geometry.
        '''
        plotting_elements = additional_plotting_elements.copy()
        if not isinstance(meshes, list) and not isinstance(meshes, tuple):
            meshes = [meshes]

        # Create plotting meshes for primitives
        plotting_elements = self.plot(primitives=primitives, plot_types=primitives_plot_types, opacity=primitives_opacity,
                                      color=primitives_color, surface_texture=primitives_surface_texture,
                                      additional_plotting_elements=plotting_elements,show=False)

        for mesh in meshes:
            if type(mesh) is am.MappedArray:
                points = mesh.value
            else:
                points = mesh

            if isinstance(mesh, tuple):
                # Is vector, so draw an arrow
                processed_points = ()
                for point in mesh:
                    if type(point) is am.MappedArray:
                        processed_points = processed_points + (point.value,)
                    else:
                        processed_points = processed_points + (point,)
                arrow = vedo.Arrow(tuple(processed_points[0].reshape((-1,))), tuple((processed_points[0] + processed_points[1]).reshape((-1,))), s=0.05)
                plotting_elements.append(arrow)
                continue

            if 'point_cloud' in mesh_plot_types:
                num_points = np.cumprod(points.shape[:-1])[-1]
                plotting_elements.append(vedo.Points(points.reshape((num_points,-1)), r=4).color('#00C6D7'))

            if points.shape[0] == 1:
                points = points.reshape((points.shape[1:]))

            if len(points.shape) == 2:  # If it's a curve
                from vedo import Line
                plotting_elements.append(Line(points).color(mesh_color).linewidth(3))
                
                if 'wireframe' in mesh_plot_types:
                    num_points = np.cumprod(points.shape[:-1])[-1]
                    plotting_elements.append(vedo.Points(points.reshape((num_points,-1)), r=12).color(mesh_color))
                continue

            if ('mesh' in mesh_plot_types or 'wireframe' in mesh_plot_types) and len(points.shape) == 3:
                num_points_u = points.shape[0]
                num_points_v = points.shape[1]
                num_points = num_points_u*num_points_v
                vertices = []
                faces = []
                for u_index in range(num_points_u):
                    for v_index in range(num_points_v):
                        vertex = tuple(points[u_index,v_index,:])
                        vertices.append(vertex)
                        if u_index != 0 and v_index != 0:
                            face = tuple((
                                (u_index-1)*num_points_v+(v_index-1),
                                (u_index-1)*num_points_v+(v_index),
                                (u_index)*num_points_v+(v_index),
                                (u_index)*num_points_v+(v_index-1),
                            ))
                            faces.append(face)

                plotting_mesh = vedo.Mesh([vertices, faces]).opacity(mesh_opacity).color('lightblue')
            if 'mesh' in mesh_plot_types:
                plotting_elements.append(plotting_mesh)
            if 'wireframe' in mesh_plot_types:
                # plotting_mesh = vedo.Mesh([vertices, faces]).opacity(mesh_opacity).color('blue')
                plotting_mesh = vedo.Mesh([vertices, faces]).opacity(mesh_opacity).color(mesh_color) # Default is UCSD Sand
                plotting_elements.append(plotting_mesh.wireframe().linewidth(3))
            
        if show:
            

            plotter = vedo.Plotter(size=(3200,2000))
            # plotter = vedo.Plotter()
            plotter.show(plotting_elements, 'Meshes', axes=1, viewup="z", interactive=True, camera=camera)
            return plotting_elements
        else:
            return plotting_elements
    
    def evaluate_parametric(self, parametric_nodes:list) -> am.MappedArray:
        num_control_points = np.cumprod(self.control_points['geometry'].shape[:-1])[-1]
        num_points = len(parametric_nodes)
        linear_map = sps.lil_array((num_points, num_control_points))
        i = 0
        for node in parametric_nodes:
            receiving_target = self.primitives[node[0]]
            point_map_on_receiving_target = receiving_target.geometry_primitive.compute_evaluation_map(u_vec=np.array([node[1][0,0]]), v_vec=np.array([node[1][0,1]]))
            receiving_target_control_point_indices = self.primitive_indices[receiving_target.name]['geometry']
            linear_map[i, receiving_target_control_point_indices] = point_map_on_receiving_target
            i += 1
        shape = (len(parametric_nodes),self.control_points['geometry'].shape[-1],)
        return am.array(self.control_points['geometry'], linear_map=linear_map.tocsc(), shape=shape)

    
    def plot(self, primitives:list=[], point_types=['evaluated_points'], plot_types:list=['mesh'],
             opacity:float=1., color:str='#00629B', surface_texture:str="",
             additional_plotting_elements:list=[], show:bool=True):
        '''
        Plots the geometry or a subset of the geometry.
        
        Parameters
        -----------
        primitives : list
            The list of primitives to be plotted. This can be the primitive names or the objects themselves.
        points_type : list
            The type of points to be plotted. {evaluated_points, control_points}
        plot_types : list
            The type of plot {mesh, wireframe, point_cloud}
        opactity : float
            The opacity of the plot. 0 is fully transparent and 1 is fully opaque.
        color : str
            The 6 digit color code for the plotting color of the primitives.
        surface_texture : str {"", "metallic", "glossy", "ambient",... see Vedo for more options}
            The surface texture for the primitive surfaces. (determines how light bounces off)
            More options: https://github.com/marcomusy/vedo/blob/master/examples/basic/lightings.py
        additional_plotting_elemets : list
            Vedo plotting elements that may have been returned from previous plotting functions that should be plotted with this plot.
        show : bool
            A boolean on whether to show the plot or not. If the plot is not shown, the Vedo plotting element is returned.
        '''

        plotting_elements = additional_plotting_elements.copy()
        if not primitives:  # If empty, plot geometry as a whole
            primitives = list(self.primitives.values())        
        if primitives[0] == 'all':
            primitives = list(self.primitives.values())
        if primitives[0] == 'none':
            return plotting_elements

        for primitive in primitives:
            if isinstance(primitive, str):
                primitive = self.primitives[primitive]

            plotting_elements = primitive.plot(point_types=point_types, plot_types=plot_types,
                                               opacity=opacity, color=color, surface_texture=surface_texture,
                                               additional_plotting_elements=plotting_elements, show=False)

        if show:
            plotter = vedo.Plotter()
            plotter.show(plotting_elements, 'Geometry', axes=1, viewup="z")
            return plotting_elements
        else:
            return plotting_elements


''' Leaning towards deleting this and creating a unified way to handle primitives. '''
# class BSplineGeometry:
#     '''
#     TODO: A B-spline geometry object.
#     The current Geometry object will likely be turned into a BSplineGeometry object
#     and a new general Geometry object will be made.

#     Parameters
#     ----------


#     '''
#     def __init__(self) -> None:
#         pass