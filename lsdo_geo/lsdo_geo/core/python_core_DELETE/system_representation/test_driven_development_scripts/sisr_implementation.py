import caddee as cd 
import numpy as np
import array_mapper as am

# evtol = cd.CADDEE()
# from lsdo_geo.caddee_core.caddee import CADDEE
# evtol = CADDEE()
# evtol.set_units('SI')

from lsdo_geo.caddee_core.system_representation.system_representation import SystemRepresentation
system_representation = SystemRepresentation()
from lsdo_geo.caddee_core.system_parameterization.system_parameterization import SystemParameterization
system_parameterization = SystemParameterization(system_representation=system_representation)

file_path = 'models/stp/'
spatial_rep = system_representation.spatial_representation
spatial_rep.import_file(file_name=file_path+'lift_plus_cruise_final_3.stp')

spatial_rep.plot(plot_types=['mesh'])

# Create Components
from lsdo_geo.caddee_core.system_representation.component.component import LiftingSurface
wing_primitive_names = list(spatial_rep.get_primitives(search_names=['Wing']).keys())
wing = LiftingSurface(name='wing', spatial_representation=spatial_rep, primitive_names=wing_primitive_names)  # TODO add material arguments
tail_primitive_names = list(spatial_rep.get_primitives(search_names=['Tail_1']).keys())
horizontal_stabilizer = LiftingSurface(name='tail', spatial_representation=spatial_rep, primitive_names=tail_primitive_names)  # TODO add material arguments
system_representation.add_component(wing)
system_representation.add_component(horizontal_stabilizer)

''' TODO: Skip Joints/Actuators for now '''
# TODO Redo actuations to use the kinematic optimization.
# # Actuator
# # # Example: tilt-wing
# rotation_axis = starboard_tip_quarter_chord - port_tip_quarter_chord
# rotation_origin = wing.project_points([1., 0., 2.])
# tilt_wing_actuator = cd.Actuator(actuating_components=[wing, rotor1], rotation_origin=rotation_origin, rotation_axis=rotation_axis)
# tilt_wing_actuator = cd.Actuator(actuating_components=['wing', 'rotor1'], rotation_origin=rotation_origin, rotation_axis=rotation_axis)
# system_parameterization.add_actuator(tilt_wing_actuator)

# TODO Go back and implement Joints after the kinematics optimization (make MBD optimization (energy minimization probably))
# Joints
# Example: rotor mounted to boom
# rotor_to_boom_connection_point = np.array([6., 18., 2.])
# rotor_to_boom_connection_point_on_rotor = rotor1.project(rotor_to_boom_connection_point)
# rotor_to_boom_connection_point_on_boom = boom.project(rotor_to_boom_connection_point)
# rotor_boom_joint = cd.Joint()


# Note: Powertrain and material definitions have been skipped for the sake of time in this iteration.

# # Parameterization
from lsdo_geo.caddee_core.system_parameterization.free_form_deformation.ffd_functions import create_cartesian_enclosure_volume
from lsdo_geo.caddee_core.system_parameterization.free_form_deformation.ffd_block import SRBGFFDBlock

wing_geometry_primitives = wing.get_geometry_primitives()
wing_ffd_b_spline_volume = create_cartesian_enclosure_volume(wing_geometry_primitives, num_coefficients=(11, 2, 2), order=(4,2,2), xyz_to_uvw_indices=(1,0,2))
wing_ffd_block = SRBGFFDBlock(name='wing_ffd_block', primitive=wing_ffd_b_spline_volume, embedded_entities=wing_geometry_primitives)
wing_ffd_block.add_scale_v(name='linear_taper', order=2, num_dof=3, value=np.array([0., 1., 0.]), cost_factor=1.)
wing_ffd_block.add_rotation_u(name='twist_distribution', order=4, num_dof=10, value=-1/2*np.array([0., 0.11, 0.22, 0.33, 0.44, 0.44, 0.33, 0.22, 0.11, 0.]))

horizontal_stabilizer_geometry_primitives = horizontal_stabilizer.get_geometry_primitives()
horizontal_stabilizer_ffd_b_spline_volume = create_cartesian_enclosure_volume(horizontal_stabilizer_geometry_primitives, num_coefficients=(11, 2, 2), order=(4,2,2), xyz_to_uvw_indices=(1,0,2))
horizontal_stabilizer_ffd_block = SRBGFFDBlock(name='horizontal_stabilizer_ffd_block', primitive=horizontal_stabilizer_ffd_b_spline_volume, embedded_entities=horizontal_stabilizer_geometry_primitives)
horizontal_stabilizer_ffd_block.add_scale_v(name='horizontal_stabilizer_linear_taper', order=2, num_dof=3, value=np.array([0.5, 0.5, 0.5]), cost_factor=1.)
horizontal_stabilizer_ffd_block.add_rotation_u(name='horizontal_stabilizer_twist_distribution', order=1, num_dof=1, value=np.array([np.pi/10]))

plotting_elements = wing_ffd_block.plot(plot_embedded_entities=False, show=False)
plotting_elements = horizontal_stabilizer_ffd_block.plot(plot_embedded_entities=False, show=False, additional_plotting_elements=plotting_elements)
spatial_rep.plot(additional_plotting_elements=plotting_elements)

from lsdo_geo.caddee_core.system_parameterization.free_form_deformation.ffd_set import SRBGFFDSet
ffd_set = SRBGFFDSet(name='ffd_set', ffd_blocks={wing_ffd_block.name : wing_ffd_block, horizontal_stabilizer_ffd_block.name : horizontal_stabilizer_ffd_block})

''' TODO Finish addressing code starting from below. '''

# Geometric inputs, outputs, and constraints
''' Note: Where to add geometric inputs, outputs, and constraints? '''
''' How do we want to add parameters? We can try to use the convention from the spreadsheet, but there are too many possibilities? '''
''' Fundamental issue: In most solvers, you know exactly what the inputs need to be. For geometry, we don't. '''
# wing_to_tail_vector =  horizontal_stabilizer.project(np.array([0., 0., 0.])) - wing_root_trailing_edge

# spatial_representation.add_variable('wing_root_chord', computed_upstream=False, dv=True, quantity=MagnitudeCalculation(wing_root_chord_vector))   # geometric design variable
# spatial_representation.add_variable('wing_starboard_tip_chord', computed_upstream=True, connection_name='wing_tip_chord', quantity=MagnitudeCalculation(wing_starboard_tip_chord_vector))   # geometric input
# spatial_representation.add_variable('wing_port_tip_chord', computed_upstream=True, connection_name='wing_tip_chord', quantity=MagnitudeCalculation(wing_port_tip_chord_vector))   # geometric input
# # Note: This will throw an error because CSDL does not allow for a single variable to conenct to multiple (wing_tip_chord --> wing_starboard_tip_chord and wing_tip_chord --> wing_port_tip_chord)

# spatial_representation.add_variable('wing_to_tail_distance', computed_upstream=False, val=10, quantity=MagnitudeCalculation(wing_port_tip_chord_vector))    # Geometric constraint (very similar to geometric input)

# spatial_representation.add_variable('wing_span', output_name='wingspan', quantity=MagnitudeCalculation(wing_span_vector))   # geometric output


# Mesh definitions
num_spanwise_vlm = 21
num_chordwise_vlm = 5
leading_edge = wing.project(np.linspace(np.array([8., -26., 7.5]), np.array([8., 26., 7.5]), num_spanwise_vlm), direction=np.array([0., 0., -1.]), plot=True)  # returns MappedArray
trailing_edge = wing.project(np.linspace(np.array([15., -26., 7.5]), np.array([15., 26., 7.5]), num_spanwise_vlm), direction=np.array([0., 0., -1.]), plot=True)   # returns MappedArray
chord_surface = am.linspace(leading_edge, trailing_edge, num_chordwise_vlm)
spatial_rep.plot_meshes([chord_surface])
wing_upper_surface_wireframe = wing.project(chord_surface.value + np.array([0., 0., 1.]), direction=np.array([0., 0., -1.]), grid_search_density=15, plot=True)
wing_lower_surface_wireframe = wing.project(chord_surface.value - np.array([0., 0., 1.]), direction=np.array([0., 0., 1.]), grid_search_density=15, plot=True)
wing_camber_surface = am.linspace(wing_upper_surface_wireframe, wing_lower_surface_wireframe, 1) # this linspace will return average when n=1
spatial_rep.plot_meshes([wing_camber_surface])


num_spanwise_vlm = 11
num_chordwise_vlm = 3
leading_edge = horizontal_stabilizer.project(np.linspace(np.array([27., -6.5, 6.]), np.array([27., 6.75, 6.]), num_spanwise_vlm), direction=np.array([0., 0., -1.]), grid_search_density=15)  # returns MappedArray
trailing_edge = horizontal_stabilizer.project(np.linspace(np.array([31.5, -6.5, 6.]), np.array([31.5, 6.75, 6.]), num_spanwise_vlm), direction=np.array([0., 0., -1.]), grid_search_density=15)   # returns MappedArray
chord_surface = am.linspace(leading_edge, trailing_edge, num_chordwise_vlm)
horizontal_stabilizer_upper_surface_wireframe = horizontal_stabilizer.project(chord_surface.value + np.array([0., 0., 1.]), direction=np.array([0., 0., -1.]), grid_search_density=15)
horizontal_stabilizer_lower_surface_wireframe = horizontal_stabilizer.project(chord_surface.value - np.array([0., 0., 1.]), direction=np.array([0., 0., 1.]), grid_search_density=15)
horizontal_stabilizer_camber_surface = am.linspace(horizontal_stabilizer_upper_surface_wireframe, horizontal_stabilizer_lower_surface_wireframe, 1) # this linspace will return average when n=1

plotting_meshes = [wing_camber_surface, horizontal_stabilizer_camber_surface]
spatial_rep.plot_meshes(plotting_meshes, mesh_plot_types=['wireframe'], mesh_opacity=1.)

ffd_set.setup()

affine_section_properties = ffd_set.evaluate_affine_section_properties()
rotational_section_properties = ffd_set.evaluate_rotational_section_properties()
affine_ffd_coefficients_local_frame = ffd_set.evaluate_affine_block_deformations(plot=True)
ffd_coefficients_local_frame = ffd_set.evaluate_rotational_block_deformations(plot=True)
ffd_coefficients = ffd_set.evaluate_coefficients(plot=True)
updated_geometry = ffd_set.evaluate_embedded_entities(plot=True)
updated_primitives_names = wing.primitive_names.copy()
updated_primitives_names.extend(horizontal_stabilizer.primitive_names.copy())


print('Sample evaluation: affine section properties: \n', affine_section_properties)
print('Sample evaluation: rotational section properties: \n', rotational_section_properties)

# Performing assembly of geometry from different parameterizations which usually happens in SystemParameterization
spatial_rep.update(updated_geometry, updated_primitives_names)

wing_camber_surface.evaluate(spatial_rep.coefficients)
horizontal_stabilizer_camber_surface.evaluate(spatial_rep.coefficients)
spatial_rep.plot_meshes([wing_camber_surface, horizontal_stabilizer_camber_surface], mesh_plot_types=['wireframe'], mesh_opacity=1.)

wing_vlm_mesh = VLMMesh(meshes=[wing_camber_surface, horizontal_stabilizer_camber_surface])


# Define solvers: After defining solver meshes we define solvers 
# bem = BEM(meshes=[rotor1_bem_mesh])
vlm = VLM(meshes=[wing_vlm_mesh])
# imga = IMGA(imga_mesh)
# motor_model = MotorSolverTC1(some_motor_mesh) # this should still take in a notion of a mesh in order to be associated with a node in the powertrain

# TODO Will come back to SISR after January. Note: May need some basic data transfer for just forward passes though.
# # Define coupled-analyses
# ''' Note: Need more time to iron out API and deisgn of this. A particularly challenging case for API is 3-way coupling (bem-wake_solver-vlm coupling) '''
# a_s_transfer_mesh = cd.TransferMesh(component=wing, mesh_attributes='place_holder')
# a_s_coupling = CoupledGroup(transfer_mesh=a_s_transfer_mesh)
# a_s_coupling.add_models(solvers=[vlm, imga])
# p = a_s_coupling.vlm.get_output()    # pressures
# u = a_s_coupling.imga.get_output()   # displacements
# a_s_coupling.vlm.set_input(u)
# a_s_coupling.imga.set_input(p)
# a_s_coupling.nonlinear_solver = NLSolvers.Newton(
#     max_iter=50,
#     tol=1e-5,
# )

# SISR TODO TODO TODO:
# 1. Develop API for allowing the user to specify the information that is needed
# 2. Develop python classes for taking in user information and performing necessary operations
# -- Take in user information on:
# -- -- 1. Which solvers are in the problem (the blocks in the N2)
# -- -- 2. Which solvers are coupled/which states are transferred between each solver (the connections in the N2)
# -- -- 3. How the coupling is solved (Fixed-point vs. Newton, etc.) (The solver in the N2)
# -- -- 4. SISR approach specific: The system mesh for each solver (if using a user-prescribed mesh approach)
# -- Perform operations:
# -- -- 1. Precompute transfer maps (using constraints such as energy conservation and using mesh if that is the approach)
# -- -- 2. Assemble CSDL models
# -- -- 3. Apply appropriate CSDL implicit model solver
# -- -- 4. This is a broader CADDEE thing that is out of the scope of this task, but assemble the csdl model structure with this implicit model

# Connections notes
# 1. Make the connections for models to be the same as the connections for system configuration power_systems_architecture?
# -- a. Could potentially be a default, but this should not be THE rule because describing how the system will be solved is independent from the configuration

# Mesh eval meeting notes:
# 1. Mesh evaluation CSDL is in design scenario (for the mechanics models that are being vectorized). Sizing models also have their own mesh eval model.
# 2. Actuation CSDL is also in design scenario (just before mesh evaluation)
# -- They specify the actuation modules on the design scenario cruise.actuations is a dictionary with {actuator.name, actuation_value}
# 3. We have a separate rigid body rotation model for attitude variables (roll,pitch,yaw)
# 3. Geometric outputs are computed in SystemRepresentationCSDL (for now)
# 4. SystemParameterization and SystemConfig are also modules so we use set_module_input/set_module_output
# 5. For specifying inputs like radius to a solver
# -- If we don't have geometry, use bem.set_module_input, otherwise, system_representation.set_module_input('radius', val=, dv=) 
# --   and connection is made using promotions.

# TODO List:
# 1. 
# 2. 
# 3. 
# 4.
# 5.
