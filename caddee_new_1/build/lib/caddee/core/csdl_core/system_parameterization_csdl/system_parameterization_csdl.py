from caddee.utils.base_model_csdl import BaseModelCSDL
import csdl
# # from csdl_om import Simulator
import numpy as np
import scipy.sparse as sps

from caddee.core.csdl_core.system_parameterization_csdl.system_representation_assembly_csdl import SystemRepresentationAssemblyCSDL
from caddee.core.csdl_core.system_parameterization_csdl.geometry_parameterization_solver_csdl import GeometryParameterizationSolverCSDL


class SystemParameterizationCSDL(BaseModelCSDL):
    '''
    Evaluates the parameterization of the system representation.
    '''

    def initialize(self):
        self.parameters.declare('system_parameterization')

    def define(self):
        system_parameterization = self.parameters['system_parameterization']

        # Call setup on all parameterizations
        num_free_dof = 0
        for geometry_parameterization_name, geometry_parameterization in system_parameterization.geometry_parameterizations.items():
            geometry_parameterization.setup()
            num_free_dof += geometry_parameterization.num_affine_free_dof

        if num_free_dof != 0 and system_parameterization.inputs:
            geometry_parameterization_solver = GeometryParameterizationSolverCSDL(system_parameterization=system_parameterization)
            self.add(submodel=geometry_parameterization_solver, name='geometry_parameterization_solver_model')

        for geometry_parameterization_name in system_parameterization.geometry_parameterizations:
            parameterization_model = system_parameterization.geometry_parameterizations[geometry_parameterization_name].assemble_csdl()
            self.add(submodel=parameterization_model, name=geometry_parameterization_name)

        system_representation_assebmly = SystemRepresentationAssemblyCSDL(system_parameterization=system_parameterization)
        self.add(submodel=system_representation_assebmly, name='system_representation_assembly_model')



if __name__ == "__main__":
    import csdl
    # # from csdl_om import Simulator
    from python_csdl_backend import Simulator
    import numpy as np

    from caddee.core.caddee_core.system_representation.system_representation import SystemRepresentation
    from caddee.core.caddee_core.system_parameterization.system_parameterization import SystemParameterization
    system_representation = SystemRepresentation()
    spatial_rep = system_representation.spatial_representation
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

    # # Parameterization
    from caddee.core.caddee_core.system_parameterization.free_form_deformation.ffd_functions import create_cartesian_enclosure_volume
    from caddee.core.caddee_core.system_parameterization.free_form_deformation.ffd_block import SRBGFFDBlock

    wing_ffd_set_primitives = wing.get_geometry_primitives()
    wing_ffd_bspline_volume = create_cartesian_enclosure_volume(wing_ffd_set_primitives, num_control_points=(11, 2, 2), order=(4,2,2), xyz_to_uvw_indices=(1,0,2))
    wing_ffd_block = SRBGFFDBlock(name='wing_ffd_block', primitive=wing_ffd_bspline_volume, embedded_entities=wing_ffd_set_primitives)

    wing_ffd_block.add_rotation_u(name='twist_distribution', order=4, num_dof=10, value=-1/2*np.array([0., 0.11, 0.22, 0.33, 0.44, 0.44, 0.33, 0.22, 0.11, 0.]))
    wing_ffd_block.add_rotation_v(name='wingtip_twist', order=4, num_dof=10, value=-np.array([np.pi/2, 0., 0., 0., 0., 0., 0., 0., 0., -np.pi/2]))
    wing_ffd_block.add_translation_w(name='wingtip_translation', order=4, num_dof=10, value=np.array([2., 0., 0., 0., 0., 0., 0., 0., 0., 2.]))

    from caddee.core.caddee_core.system_parameterization.free_form_deformation.ffd_set import SRBGFFDSet
    ffd_set = SRBGFFDSet(name='ffd_set', ffd_blocks={wing_ffd_block.name : wing_ffd_block})
    # ffd_set.setup(project_embedded_entities=True)
    system_parameterization.add_geometry_parameterization(ffd_set)
    system_parameterization.setup()

    # affine_section_properties = ffd_set.evaluate_affine_section_properties()
    # affine_deformed_ffd_control_points = ffd_set.evaluate_affine_block_deformations()
    # rotational_section_properties = ffd_set.evaluate_rotational_section_properties()
    # rotated_ffd_control_points = ffd_set.evaluate_rotational_block_deformations()
    # ffd_control_points = ffd_set.evaluate_control_points()
    # embedded_entities = ffd_set.evaluate_embedded_entities()
    # print('Python evaluation: embedded entities: \n', embedded_entities)

    sim = Simulator(SystemParameterizationCSDL(system_parameterization=system_parameterization))
    sim.run()
    # sim.visualize_implementation()        # Only csdl_om can do this

    # print('CSDL evaluation: ffd embedded entities: \n', sim['ffd_embedded_entities'])
    # print("Python and CSDL difference", np.linalg.norm(sim['ffd_embedded_entities'] - embedded_entities))

    spatial_rep.update(sim['system_representation_geometry'])
    plotting_elements = wing_ffd_block.plot_sections(control_points=sim['ffd_control_points'].reshape(wing_ffd_bspline_volume.shape), plot_embedded_entities=False, opacity=0.75, show=False)
    spatial_rep.plot(additional_plotting_elements=plotting_elements)

    '''
    Multiple FFD blocks
    '''
    system_representation = SystemRepresentation()
    spatial_rep = system_representation.spatial_representation
    system_parameterization = SystemParameterization(system_representation=system_representation)

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

    # # Parameterization
    from caddee.core.caddee_core.system_parameterization.free_form_deformation.ffd_functions import create_cartesian_enclosure_volume
    from caddee.core.caddee_core.system_parameterization.free_form_deformation.ffd_block import SRBGFFDBlock

    wing_ffd_set_primitives = wing.get_geometry_primitives()
    wing_ffd_bspline_volume = create_cartesian_enclosure_volume(wing_ffd_set_primitives, num_control_points=(11, 2, 2), order=(4,2,2), xyz_to_uvw_indices=(1,0,2))
    wing_ffd_block = SRBGFFDBlock(name='wing_ffd_block', primitive=wing_ffd_bspline_volume, embedded_entities=wing_ffd_set_primitives)
    wing_ffd_block.add_rotation_u(name='twist_distribution', order=4, num_dof=10, value=-1/2*np.array([0., 0.11, 0.22, 0.33, 0.44, 0.44, 0.33, 0.22, 0.11, 0.]))
    wing_ffd_block.add_scale_v(name="chord_distribution_scaling", order=2, num_dof=3, value=np.array([0.5, 1.5, 0.5]))

    horizontal_stabilizer_ffd_set_primitives = horizontal_stabilizer.get_geometry_primitives()
    horizontal_stabilizer_ffd_bspline_volume = create_cartesian_enclosure_volume(horizontal_stabilizer_ffd_set_primitives, num_control_points=(11, 2, 2), order=(4,2,2), xyz_to_uvw_indices=(1,0,2))
    horizontal_stabilizer_ffd_block = SRBGFFDBlock(name='horizontal_stabilizer_ffd_block', primitive=horizontal_stabilizer_ffd_bspline_volume, embedded_entities=horizontal_stabilizer_ffd_set_primitives)
    horizontal_stabilizer_ffd_block.add_rotation_u(name='horizontal_stabilizer_twist_distribution', order=1, num_dof=1, value=np.array([np.pi/10]))
    # horizontal_stabilizer_ffd_block.add_scale_v(name="chord_distribution_scaling", order=2, num_dof=3, value=np.array([-0.5, 0.5, -0.5]))

    from caddee.core.caddee_core.system_parameterization.free_form_deformation.ffd_set import SRBGFFDSet
    ffd_set = SRBGFFDSet(name='ffd_set', ffd_blocks={wing_ffd_block.name : wing_ffd_block, horizontal_stabilizer_ffd_block.name : horizontal_stabilizer_ffd_block})
    # ffd_set.setup(project_embedded_entities=True)
    system_parameterization.add_geometry_parameterization(ffd_set)
    system_parameterization.setup()

    # affine_section_properties = ffd_set.evaluate_affine_section_properties()
    # rotational_section_properties = ffd_set.evaluate_rotational_section_properties()
    # affine_deformed_ffd_control_points = ffd_set.evaluate_affine_block_deformations()
    # rotated_ffd_control_points = ffd_set.evaluate_rotational_block_deformations()
    # ffd_control_points = ffd_set.evaluate_control_points()
    # ffd_embedded_entities = ffd_set.evaluate_embedded_entities()
    # print('Python evaluation: FFD embedded entities: \n', rotated_ffd_control_points)

    sim = Simulator(SystemParameterizationCSDL(system_parameterization=system_parameterization))
    sim.run()
    # sim.visualize_implementation()    $ Only usable with csdl_om

    # print('CSDL evaluation: FFD embedded entities: \n', sim['ffd_embedded_entities'])
    # print("Python and CSDL difference", np.linalg.norm(sim['ffd_embedded_entities'] - ffd_embedded_entities))

    updated_primitives_names = wing.primitive_names.copy()
    updated_primitives_names.extend(horizontal_stabilizer.primitive_names.copy())
    # spatial_rep.update(sim['ffd_embedded_entities'], updated_primitives_names)
    spatial_rep.update(sim['system_representation_geometry'])
    
    plotting_elements = wing_ffd_block.plot_sections(
        control_points=(sim['ffd_control_points'][:wing_ffd_block.num_control_points,:]).reshape(wing_ffd_bspline_volume.shape),
        plot_embedded_entities=False, opacity=0.75, show=False)
    plotting_elements = horizontal_stabilizer_ffd_block.plot_sections(
        control_points=(sim['ffd_control_points'][wing_ffd_block.num_control_points:,:]).reshape(wing_ffd_bspline_volume.shape), 
        plot_embedded_entities=False, opacity=0.75, additional_plotting_elements=plotting_elements, show=False)
    spatial_rep.plot(additional_plotting_elements=plotting_elements)
