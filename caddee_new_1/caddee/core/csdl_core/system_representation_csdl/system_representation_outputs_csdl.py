import csdl
# from csdl_om import Simulator
import numpy as np
import scipy.sparse as sps
import array_mapper as am

class SystemRepresentationOutputsCSDL(csdl.Model):
    '''
    Evaluates the system configuration outputs including meshes and user-specified quantities.
    '''

    def initialize(self):
        self.parameters.declare('system_representation')

    def define(self):
        system_representation = self.parameters['system_representation']
        
        design_outputs_model = ConfigurationOutputsModel(configuration_name='design', configuration=system_representation)
        self.add(design_outputs_model, name='design_outputs_model')

        for configuration_name, configuration in system_representation.configurations.items():
            if configuration.outputs:
                configuration_outputs_model = ConfigurationOutputsModel(configuration_name=configuration_name,
                                                                        configuration=configuration)
                self.add(configuration_outputs_model, name=f'{configuration_name}_outputs_model')
        


class ConfigurationOutputsModel(csdl.Model):
    '''
    Evaluates the outputs of a specific configuraiton.
    '''
    def initialize(self):
        self.parameters.declare('configuration_name')
        self.parameters.declare('configuration')

    def define(self):
        configuration_name = self.parameters['configuration_name']
        configuration = self.parameters['configuration']
        if configuration_name != 'design':
            num_nodes = configuration.num_nodes
        else:
            num_nodes = 1

        if configuration_name == 'design':
            system_representation = configuration
        else:
            system_representation = configuration.system_representation
        spatial_rep = system_representation.spatial_representation

        initial_geometry_control_points = spatial_rep.control_points['geometry']
        if num_nodes != 1:
            initial_geometry_control_points = np.broadcast_to(initial_geometry_control_points, 
                                                              shape=(num_nodes,)+initial_geometry_control_points.shape)

        # Count number of nonlinear outputs
        if configuration_name == 'design':
            outputs = spatial_rep.outputs
        else:
            outputs = configuration.outputs

        num_nonlinear_outputs = 0
        for output_name, output in outputs.items():
            if type(output) is am.NonlinearMappedArray:
                num_nonlinear_outputs += 1

        # Declare input
        configuration_geometry_tensor = self.declare_variable(f'{configuration_name}_geometry', val=initial_geometry_control_points)

        nonlinear_outputs_list = []
        for t in range(num_nodes):
            if num_nodes != 1:
                configuration_geometry_t = configuration_geometry_tensor[t,:,:]
                configuration_geometry = csdl.reshape(configuration_geometry_t, new_shape=configuration_geometry_tensor.shape[1:])
            else:
                configuration_geometry = configuration_geometry_tensor*1

            configuration_geometry_t = self.register_output(f'{configuration_name}_geometry_{t}', configuration_geometry)

            configuration_geometry_t_norm = csdl.pnorm(configuration_geometry_t)
            self.register_output(f'{configuration_name}_geometry_{t}_norm', configuration_geometry_t_norm)

            nonlinear_outputs = csdl.custom(configuration_geometry_t,
                                    op=NonlinearOutputsOperation(configuration_name=configuration_name, configuration=configuration, t=t))
            nonlinear_outputs_list.append(nonlinear_outputs)

        nonlinear_output_counter = 0
        for output_name, output in outputs.items():
            if type(output) is am.MappedArray:
                if num_nodes != 1:
                    flattened_shape = (num_nodes,np.prod(output.shape))
                    updated_mesh_csdl_flattened = self.create_output(f'flattened_{output_name}', shape=flattened_shape)

                for t in range(num_nodes):
                    if num_nodes != 1:
                        configuration_geometry_t = configuration_geometry_tensor[t,:,:]
                        configuration_geometry = csdl.reshape(configuration_geometry_t, new_shape=configuration_geometry_tensor.shape[1:])
                    else:
                        configuration_geometry = configuration_geometry_tensor

                    # mesh output
                    # updated_mesh_csdl = mesh.evaluate(system_representation_geometry)  # Option 1 where I implement this in array_mapper

                    # Option 2
                    mesh_linear_map = output.linear_map
                    mesh_offset_map = output.offset_map
                    updated_mesh_csdl_without_offset = csdl.sparsematmat(configuration_geometry, sparse_mat=mesh_linear_map)

                    if mesh_offset_map is not None:
                        mesh_offset_map_csdl = self.create_output(f'{configuration_name}_{output_name}_offset_map', mesh_offset_map)
                        updated_mesh_csdl = updated_mesh_csdl_without_offset + mesh_offset_map_csdl
                    else:
                        updated_mesh_csdl = updated_mesh_csdl_without_offset

                    updated_mesh_csdl_reshaped = csdl.reshape(updated_mesh_csdl, output.shape)

                    if num_nodes == 1:
                        self.register_output(output_name, updated_mesh_csdl_reshaped)
                    else:
                        updated_mesh_csdl_flattened[t,:] = csdl.reshape(updated_mesh_csdl_reshaped, 
                                                                        new_shape=(1,np.prod(output.shape)))

                if num_nodes != 1:
                    updated_mesh_csdl_reshaped = csdl.reshape(updated_mesh_csdl_flattened, (num_nodes,) + output.shape)
                    self.register_output(output_name, updated_mesh_csdl_reshaped)


            elif type(output) is am.NonlinearMappedArray:
                if num_nodes == 1:
                    if num_nonlinear_outputs == 1:
                        nonlinear_output = nonlinear_outputs
                    else:
                        nonlinear_output = nonlinear_outputs[nonlinear_output_counter]
                    self.register_output(f'{output_name}_{t}', nonlinear_output)
                    self.register_output(output_name, nonlinear_output*1)
                else:
                    flattened_output_shape = np.prod(output.shape)
                    flattened_nonlinear_output_across_time = self.create_output(f'flattened_{output_name}', shape=(num_nodes,flattened_output_shape))
                    for t in range(num_nodes):
                        output_t = nonlinear_outputs_list[t]
                        if num_nonlinear_outputs > 1:
                            output_t = output_t[nonlinear_output_counter]
                        flattened_nonlinear_output_across_time[t,:] = csdl.reshape(output_t, (1,)+output.shape)

                    nonlinear_output_across_time = csdl.reshape(flattened_nonlinear_output_across_time, (num_nodes,) + output.shape)
                    self.register_output(output_name, nonlinear_output_across_time)

                nonlinear_output_counter += 1



class NonlinearOutputsOperation(csdl.CustomExplicitOperation):
    def initialize(self):
        self.parameters.declare('configuration_name')
        self.parameters.declare('configuration')
        self.parameters.declare('t')


    def define(self):
        configuration_name = self.parameters['configuration_name']
        configuration = self.parameters['configuration']
        t = self.parameters['t']
        if configuration_name == 'design':
            system_representation = configuration
        else:
            system_representation = configuration.system_representation
        spatial_rep = system_representation.spatial_representation

        self.add_input(f'{configuration_name}_geometry_{t}', val=spatial_rep.control_points['geometry'].copy())

        if configuration_name == 'design':
            geo_outputs = spatial_rep.outputs
        else:
            geo_outputs = configuration.outputs

        for output_name, output in geo_outputs.items():
            if type(output) is am.NonlinearMappedArray:
                self.add_output(name=output_name+f'_{t}', val=output.value)

        # self.declare_derivatives('*','*')

        for output_name, output in geo_outputs.items():
            if type(output) is am.NonlinearMappedArray:
                self.declare_derivatives(output_name+f'_{t}',f'{configuration_name}_geometry_{t}')


    def compute(self, inputs, outputs):
        configuration_name = self.parameters['configuration_name']
        configuration = self.parameters['configuration']
        t = self.parameters['t']
        if configuration_name == 'design':
            system_representation = configuration
        else:
            system_representation = configuration.system_representation
        spatial_rep = system_representation.spatial_representation

        input = inputs[f'{configuration_name}_geometry_{t}']
        if configuration_name == 'design':
            geo_outputs = spatial_rep.outputs
        else:
            geo_outputs = configuration.outputs

        for output_name, output in geo_outputs.items():
            if type(output) is am.NonlinearMappedArray:
                outputs[output_name+f'_{t}'] = output.evaluate(input)
    


    def compute_derivatives(self, inputs, derivatives):
        configuration_name = self.parameters['configuration_name']
        configuration = self.parameters['configuration']
        t = self.parameters['t']
        if configuration_name == 'design':
            system_representation = configuration
        else:
            system_representation = configuration.system_representation
        spatial_rep = system_representation.spatial_representation

        input = inputs[f'{configuration_name}_geometry_{t}']
        if configuration_name == 'design':
            geo_outputs = spatial_rep.outputs
        else:
            geo_outputs = configuration.outputs

        for output_name, output in geo_outputs.items():

            if type(output) is am.NonlinearMappedArray:
                derivatives[output_name+f'_{t}', f'{configuration_name}_geometry_{t}'] = output.evaluate_derivative(input)
                # print(f'partials {output_name}', derivatives[output_name+f'_{t}', f'{configuration_name}_geometry'], np.linalg.norm( derivatives[output_name+f'_{t}', f'{configuration_name}_geometry']), np.max( derivatives[output_name+f'_{t}', f'{configuration_name}_geometry']), np.min( derivatives[output_name+f'_{t}', f'{configuration_name}_geometry']))


if __name__ == "__main__":
    import csdl
    # # from csdl_om import Simulator
    from python_csdl_backend import Simulator
    import numpy as np
    from vedo import Points, Plotter
    import array_mapper as am

    from caddee.core.caddee_core.system_representation.system_representation import SystemRepresentation
    system_representation = SystemRepresentation()
    spatial_rep = system_representation.spatial_representation
    from caddee.core.caddee_core.system_parameterization.system_parameterization import SystemParameterization
    system_parameterization = SystemParameterization(system_representation=system_representation)

    '''
    Single FFD Block
    '''
    file_path = 'models/stp/'
    spatial_rep.import_file(file_name=file_path+'rect_wing.stp')

    # Create Components
    from caddee.core.caddee_core.system_representation.component.component import LiftingSurface, Component
    wing_primitive_names = list(spatial_rep.get_primitives(search_names=['Wing']).keys())
    wing = LiftingSurface(name='wing', spatial_representation=spatial_rep, primitive_names=wing_primitive_names)  # TODO add material arguments
    system_representation.add_component(wing)

    # Mesh definition
    num_spanwise_vlm = 21
    num_chordwise_vlm = 5
    leading_edge = wing.project(np.linspace(np.array([0., -9., 0.]), np.array([0., 9., 0.]), num_spanwise_vlm), direction=np.array([0., 0., -1.]))  # returns MappedArray
    trailing_edge = wing.project(np.linspace(np.array([4., -9., 0.]), np.array([4., 9., 0.]), num_spanwise_vlm), direction=np.array([0., 0., -1.]))   # returns MappedArray
    chord_surface = am.linspace(leading_edge, trailing_edge, num_chordwise_vlm)
    wing_upper_surface_wireframe = wing.project(chord_surface.value + np.array([0., 0., 1.]), direction=np.array([0., 0., -1.]), grid_search_n=25)
    wing_lower_surface_wireframe = wing.project(chord_surface.value - np.array([0., 0., 1.]), direction=np.array([0., 0., 1.]), grid_search_n=25)
    wing_camber_surface = am.linspace(wing_upper_surface_wireframe, wing_lower_surface_wireframe, 1) # this linspace will return average when n=1

    system_representation.add_output('wing_camber_surface', wing_camber_surface)

    starboard_tip = wing.project(np.array([2., 9., 7.5]), direction=np.array([0., 0., -1.]))
    port_tip = wing.project(np.array([2., -9., 7.5]), direction=np.array([0., 0., -1.]))
    wingspan_vector = starboard_tip - port_tip
    wingspan = am.norm(wingspan_vector)     # NOTE: Nonlinear operations don't return MappedArrays. They return NonlinearMappedarrays
    spatial_rep.add_output(name='wingspan', quantity=wingspan)
    root_leading = wing.project(np.array([0., 0., 7.5]), direction=np.array([0., 0., -1.]))
    root_trailing = wing.project(np.array([4., 0., 7.5]), direction=np.array([0., 0., -1.]))
    root_chord_vector = root_leading - root_trailing
    root_chord = am.norm(root_chord_vector)     # NOTE: Nonlinear operations don't return MappedArrays. They return NonlinearMappedarrays
    # spatial_rep.add_output(name='wing_root_chord', quantity=root_chord)

    # # Parameterization
    from caddee.core.caddee_core.system_parameterization.free_form_deformation.ffd_functions import create_cartesian_enclosure_volume
    from caddee.core.caddee_core.system_parameterization.free_form_deformation.ffd_block import SRBGFFDBlock

    wing_geometry_primitives = wing.get_geometry_primitives()
    wing_ffd_bspline_volume = create_cartesian_enclosure_volume(wing_geometry_primitives, num_control_points=(11, 2, 2), order=(4,2,2), xyz_to_uvw_indices=(1,0,2))
    wing_ffd_block = SRBGFFDBlock(name='wing_ffd_block', primitive=wing_ffd_bspline_volume, embedded_entities=wing_geometry_primitives)

    wing_ffd_block.add_rotation_u(name='twist_distribution', order=4, num_dof=10, value=-1/2*np.array([0., 0.11, 0.22, 0.33, 0.44, 0.44, 0.33, 0.22, 0.11, 0.]))
    wing_ffd_block.add_rotation_v(name='wingtip_twist', order=4, num_dof=10, value=-np.array([np.pi/2, 0., 0., 0., 0., 0., 0., 0., 0., -np.pi/2]))
    wing_ffd_block.add_translation_w(name='wingtip_translation', order=4, num_dof=10, value=np.array([2., 0., 0., 0., 0., 0., 0., 0., 0., 2.]))

    from caddee.core.caddee_core.system_parameterization.free_form_deformation.ffd_set import SRBGFFDSet
    ffd_set = SRBGFFDSet(name='ffd_set', ffd_blocks={wing_ffd_block.name : wing_ffd_block})

    ffd_set.setup(project_embedded_entities=True)
    affine_section_properties = ffd_set.evaluate_affine_section_properties()
    affine_deformed_ffd_control_points = ffd_set.evaluate_affine_block_deformations()
    rotational_section_properties = ffd_set.evaluate_rotational_section_properties()
    rotated_ffd_control_points = ffd_set.evaluate_rotational_block_deformations()
    ffd_control_points = ffd_set.evaluate_control_points()
    embedded_entities = ffd_set.evaluate_embedded_entities()
    spatial_rep.update(embedded_entities)

    updated_wing_camber_surface = wing_camber_surface.evaluate()

    sim = Simulator(SystemRepresentationOutputsCSDL(system_representation=system_representation))
    sim.run()

    print('wingspan', sim['wingspan'])
    print("Python and CSDL difference: wingspan", np.linalg.norm(wingspan.value - sim['wingspan']))
    # print('wing root chord', sim['wing_root_chord'])
    # print("Python and CSDL difference", np.linalg.norm(root_chord.value - sim['wing_root_chord']))

    wing_camber_surface_csdl = sim['wing_camber_surface'].reshape(wing_camber_surface.shape)
    print("Python and CSDL difference: wing camber surface", np.linalg.norm(wing_camber_surface_csdl - updated_wing_camber_surface))

    spatial_rep.plot_meshes([wing_camber_surface_csdl], mesh_plot_types=['wireframe'], mesh_opacity=1.)

    '''
    Multiple FFD blocks
    '''
    system_representation = SystemRepresentation()
    spatial_rep = system_representation.spatial_representation
    file_path = 'models/stp/'
    spatial_rep.import_file(file_name=file_path+'lift_plus_cruise_final_3.stp')

    # Create Components
    from caddee.core.caddee_core.system_representation.component.component import LiftingSurface
    wing_primitive_names = list(spatial_rep.get_primitives(search_names=['Wing']).keys())
    wing = LiftingSurface(name='wing', spatial_representation=spatial_rep, primitive_names=wing_primitive_names)  # TODO add material arguments
    tail_primitive_names = list(spatial_rep.get_primitives(search_names=['Tail_1']).keys())
    horizontal_stabilizer = LiftingSurface(name='tail', spatial_representation=spatial_rep, primitive_names=tail_primitive_names)  # TODO add material arguments
    system_representation.add_component(wing)
    system_representation.add_component(horizontal_stabilizer)

    # Meshes definitions
    num_spanwise_vlm = 21
    num_chordwise_vlm = 5
    leading_edge = wing.project(np.linspace(np.array([8., -26., 7.5]), np.array([8., 26., 7.5]), num_spanwise_vlm), direction=np.array([0., 0., -1.]))  # returns MappedArray
    trailing_edge = wing.project(np.linspace(np.array([15., -26., 7.5]), np.array([15., 26., 7.5]), num_spanwise_vlm), direction=np.array([0., 0., -1.]))   # returns MappedArray
    chord_surface = am.linspace(leading_edge, trailing_edge, num_chordwise_vlm)
    wing_upper_surface_wireframe = wing.project(chord_surface.value + np.array([0., 0., 1.]), direction=np.array([0., 0., -1.]), grid_search_n=15)
    wing_lower_surface_wireframe = wing.project(chord_surface.value - np.array([0., 0., 1.]), direction=np.array([0., 0., 1.]), grid_search_n=15)
    wing_camber_surface = am.linspace(wing_upper_surface_wireframe, wing_lower_surface_wireframe, 1) # this linspace will return average when n=1

    num_spanwise_vlm = 11
    num_chordwise_vlm = 3
    leading_edge = horizontal_stabilizer.project(np.linspace(np.array([27., -6.5, 6.]), np.array([27., 6.75, 6.]), num_spanwise_vlm), direction=np.array([0., 0., -1.]), grid_search_n=15)  # returns MappedArray
    trailing_edge = horizontal_stabilizer.project(np.linspace(np.array([31.5, -6.5, 6.]), np.array([31.5, 6.75, 6.]), num_spanwise_vlm), direction=np.array([0., 0., -1.]), grid_search_n=15)   # returns MappedArray
    chord_surface = am.linspace(leading_edge, trailing_edge, num_chordwise_vlm)
    horizontal_stabilizer_upper_surface_wireframe = horizontal_stabilizer.project(chord_surface.value + np.array([0., 0., 1.]), direction=np.array([0., 0., -1.]), grid_search_n=15)
    horizontal_stabilizer_lower_surface_wireframe = horizontal_stabilizer.project(chord_surface.value - np.array([0., 0., 1.]), direction=np.array([0., 0., 1.]), grid_search_n=15)
    horizontal_stabilizer_camber_surface = am.linspace(horizontal_stabilizer_upper_surface_wireframe, horizontal_stabilizer_lower_surface_wireframe, 1) # this linspace will return average when n=1

    system_representation.add_output('wing_camber_surface', wing_camber_surface)
    system_representation.add_output('horizontal_stabilizer_camber_surface', horizontal_stabilizer_camber_surface)
    starboard_trailing_tip = wing.project(np.array([15., 26., 7.5]), direction=np.array([0., 0., -1.]))
    port_trailing_tip = wing.project(np.array([15., -26., 7.5]), direction=np.array([0., 0., -1.]))
    wingspan_vector = starboard_trailing_tip - port_trailing_tip
    wingspan = am.norm(wingspan_vector)     # NOTE: Nonlinear operations don't return MappedArrays. They return NonlinearMappedarrays
    spatial_rep.add_output(name='wingspan', quantity=wingspan)
    root_leading = wing.project(np.array([9., 0., 7.5]), direction=np.array([0., 0., -1.]))
    root_trailing = wing.project(np.array([15., 0., 7.5]), direction=np.array([0., 0., -1.]))
    root_chord_vector = root_leading - root_trailing
    root_chord = am.norm(root_chord_vector)     # NOTE: Nonlinear operations don't return MappedArrays. They return NonlinearMappedarrays
    spatial_rep.add_output(name='wing_root_chord', quantity=root_chord)

    # # Parameterization
    from caddee.core.caddee_core.system_parameterization.free_form_deformation.ffd_functions import create_cartesian_enclosure_volume
    from caddee.core.caddee_core.system_parameterization.free_form_deformation.ffd_block import SRBGFFDBlock

    wing_geometry_primitives = wing.get_geometry_primitives()
    wing_ffd_bspline_volume = create_cartesian_enclosure_volume(wing_geometry_primitives, num_control_points=(11, 2, 2), order=(4,2,2), xyz_to_uvw_indices=(1,0,2))
    wing_ffd_block = SRBGFFDBlock(name='wing_ffd_block', primitive=wing_ffd_bspline_volume, embedded_entities=wing_geometry_primitives)
    wing_ffd_block.add_scale_v(name='linear_taper', order=2, num_dof=3, value=np.array([0., 1., 0.]), cost_factor=1.)
    wing_ffd_block.add_scale_w(name='constant_thickness_scaling', order=1, num_dof=1, value=np.array([0.5]))
    wing_ffd_block.add_rotation_u(name='twist_distribution', order=4, num_dof=10, value=-1/4*np.array([0., 0.11, 0.22, 0.33, 0.44, 0.44, 0.33, 0.22, 0.11, 0.]))

    horizontal_stabilizer_geometry_primitives = horizontal_stabilizer.get_geometry_primitives()
    horizontal_stabilizer_ffd_bspline_volume = create_cartesian_enclosure_volume(horizontal_stabilizer_geometry_primitives, num_control_points=(11, 2, 2), order=(4,2,2), xyz_to_uvw_indices=(1,0,2))
    horizontal_stabilizer_ffd_block = SRBGFFDBlock(name='horizontal_stabilizer_ffd_block', primitive=horizontal_stabilizer_ffd_bspline_volume, embedded_entities=horizontal_stabilizer_geometry_primitives)
    horizontal_stabilizer_ffd_block.add_scale_v(name='horizontal_stabilizer_linear_taper', order=2, num_dof=3, value=np.array([0.5, 0.5, 0.5]), cost_factor=1.)
    horizontal_stabilizer_ffd_block.add_rotation_u(name='horizontal_stabilizer_twist_distribution', order=1, num_dof=1, value=np.array([np.pi/10]))

    from caddee.core.caddee_core.system_parameterization.free_form_deformation.ffd_set import SRBGFFDSet
    ffd_set = SRBGFFDSet(name='ffd_set', ffd_blocks={wing_ffd_block.name : wing_ffd_block, horizontal_stabilizer_ffd_block.name : horizontal_stabilizer_ffd_block})

    ffd_set.setup(project_embedded_entities=True)
    affine_section_properties = ffd_set.evaluate_affine_section_properties()
    rotational_section_properties = ffd_set.evaluate_rotational_section_properties()
    affine_deformed_ffd_control_points = ffd_set.evaluate_affine_block_deformations()
    rotated_ffd_control_points = ffd_set.evaluate_rotational_block_deformations()
    ffd_control_points = ffd_set.evaluate_control_points()
    ffd_embedded_entities = ffd_set.evaluate_embedded_entities()

    updated_primitives_names = wing.primitive_names.copy()
    updated_primitives_names.extend(horizontal_stabilizer.primitive_names.copy())
    spatial_rep.update(ffd_embedded_entities, updated_primitives_names)

    wing_camber_surface.evaluate(spatial_rep.control_points)
    horizontal_stabilizer_camber_surface.evaluate(spatial_rep.control_points)

    sim = Simulator(SystemRepresentationOutputsCSDL(system_representation=system_representation))
    sim.run()

    print('wingspan', sim['wingspan'])
    print("Python and CSDL difference: wingspan", np.linalg.norm(wingspan.value - sim['wingspan']))
    print('wing root chord', sim['wing_root_chord'])
    print("Python and CSDL difference: wing root chord", np.linalg.norm(root_chord.value - sim['wing_root_chord']))

    wing_camber_surface_csdl = sim['wing_camber_surface'].reshape(wing_camber_surface.shape)
    horizontal_stabilizer_camber_surface_csdl = sim['horizontal_stabilizer_camber_surface'].reshape(horizontal_stabilizer_camber_surface.shape)
    print("Python and CSDL difference: wing camber surface", np.linalg.norm(wing_camber_surface_csdl - wing_camber_surface.value))
    print("Python and CSDL difference: horizontal stabilizer camber surface", np.linalg.norm(horizontal_stabilizer_camber_surface_csdl - horizontal_stabilizer_camber_surface.value))

    spatial_rep.plot_meshes([wing_camber_surface_csdl, horizontal_stabilizer_camber_surface_csdl], mesh_plot_types=['wireframe'], mesh_opacity=1.)