'''Example TC2 geometry setup: Description of example 2'''
import numpy as np
import caddee.api as cd 
import lsdo_geo as lg
import m3l
from caddee import GEOMETRY_FILES_FOLDER
from caddee.utils.helper_functions.geometry_helpers import make_rotor_mesh, make_vlm_camber_mesh, make_1d_box_beam_mesh, compute_component_surface_area, BladeParameters
from caddee.utils.aircraft_models.drag_models.drag_build_up import DragComponent
from lsdo_geo.core.parameterization.volume_sectional_parameterization import VolumeSectionalParameterization
import lsdo_geo.splines.b_splines as bsp
import gc
gc.enable()
# Instantiate system model
system_model = m3l.Model()

# Importing and refitting the geometry
geometry = lg.import_geometry(GEOMETRY_FILES_FOLDER / 'LPC_final_custom_blades.stp', parallelize=True)
geometry.refit(parallelize=True, order=(4, 4))
# geometry.plot()

plot_meshes = False
geometry_dv = False
FFD = False

# region Declaring all components
# Wing, tails, fuselage
wing = geometry.declare_component(component_name='wing', b_spline_search_names=['Wing'])
h_tail = geometry.declare_component(component_name='h_tail', b_spline_search_names=['Tail_1'])
vtail = geometry.declare_component(component_name='vtail', b_spline_search_names=['Tail_2'])
fuselage = geometry.declare_component(component_name='fuselage', b_spline_search_names=['Fuselage_***.main'])

# Nose hub
nose_hub = geometry.declare_component(component_name='weird_nose_hub', b_spline_search_names=['EngineGroup_10'])

# Pusher prop
pp_disk = geometry.declare_component(component_name='pp_disk', b_spline_search_names=['Rotor-9-disk'])
pp_blade_1 = geometry.declare_component(component_name='pp_blade_1', b_spline_search_names=['Rotor_9_blades, 0'])
pp_blade_2 = geometry.declare_component(component_name='pp_blade_2', b_spline_search_names=['Rotor_9_blades, 1'])
pp_blade_3 = geometry.declare_component(component_name='pp_blade_3', b_spline_search_names=['Rotor_9_blades, 2'])
pp_blade_4 = geometry.declare_component(component_name='pp_blade_4', b_spline_search_names=['Rotor_9_blades, 3'])
pp_hub = geometry.declare_component(component_name='pp_hub', b_spline_search_names=['Rotor_9_Hub'])

# Rotor: rear left outer
rlo_disk = geometry.declare_component(component_name='rlo_disk', b_spline_search_names=['Rotor_2_disk'])
rlo_blade_1 = geometry.declare_component(component_name='rlo_blade_1', b_spline_search_names=['Rotor_2_blades, 0'])
rlo_blade_2 = geometry.declare_component(component_name='rlo_blade_2', b_spline_search_names=['Rotor_2_blades, 1'])
rlo_hub = geometry.declare_component(component_name='rlo_hub', b_spline_search_names=['Rotor_2_Hub'])
rlo_boom = geometry.declare_component(component_name='rlo_boom', b_spline_search_names=['Rotor_2_Support'])

# Rotor: rear left inner
rli_disk = geometry.declare_component(component_name='rli_disk', b_spline_search_names=['Rotor_4_disk'])
rli_blade_1 = geometry.declare_component(component_name='rli_blade_1', b_spline_search_names=['Rotor_4_blades, 0'])
rli_blade_2 = geometry.declare_component(component_name='rli_blade_2', b_spline_search_names=['Rotor_4_blades, 1'])
rli_hub = geometry.declare_component(component_name='rli_hub', b_spline_search_names=['Rotor_4_Hub'])
rli_boom = geometry.declare_component(component_name='rli_boom', b_spline_search_names=['Rotor_4_Support'])

# Rotor: rear right inner
rri_disk = geometry.declare_component(component_name='rri_disk', b_spline_search_names=['Rotor_6_disk'])
rri_blade_1 = geometry.declare_component(component_name='rri_blade_1', b_spline_search_names=['Rotor_6_blades, 0'])
rri_blade_2 = geometry.declare_component(component_name='rri_blade_2', b_spline_search_names=['Rotor_6_blades, 1'])
rri_hub = geometry.declare_component(component_name='rri_hub', b_spline_search_names=['Rotor_6_Hub'])
rri_boom = geometry.declare_component(component_name='rri_boom', b_spline_search_names=['Rotor_6_Support'])

# Rotor: rear right outer
rro_disk = geometry.declare_component(component_name='rro_disk', b_spline_search_names=['Rotor_8_disk'])
rro_blade_1 = geometry.declare_component(component_name='rro_blade_1', b_spline_search_names=['Rotor_8_blades, 0'])
rro_blade_2 = geometry.declare_component(component_name='rro_blade_2', b_spline_search_names=['Rotor_8_blades, 1'])
rro_hub = geometry.declare_component(component_name='rro_hub', b_spline_search_names=['Rotor_8_Hub'])
rro_boom = geometry.declare_component(component_name='rro_boom', b_spline_search_names=['Rotor_8_Support'])

# Rotor: front left outer
flo_disk = geometry.declare_component(component_name='flo_disk', b_spline_search_names=['Rotor_1_disk'])
flo_blade_1 = geometry.declare_component(component_name='flo_blade_1', b_spline_search_names=['Rotor_1_blades, 0'])
flo_blade_2 = geometry.declare_component(component_name='flo_blade_2', b_spline_search_names=['Rotor_1_blades, 1'])
flo_hub = geometry.declare_component(component_name='flo_hub', b_spline_search_names=['Rotor_1_Hub'])
flo_boom = geometry.declare_component(component_name='flo_boom', b_spline_search_names=['Rotor_1_Support'])

# Rotor: front left inner
fli_disk = geometry.declare_component(component_name='fli_disk', b_spline_search_names=['Rotor_3_disk'])
fli_blade_1 = geometry.declare_component(component_name='fli_blade_1', b_spline_search_names=['Rotor_3_blades, 0'])
fli_blade_2 = geometry.declare_component(component_name='fli_blade_2', b_spline_search_names=['Rotor_3_blades, 1'])
fli_hub = geometry.declare_component(component_name='fli_hub', b_spline_search_names=['Rotor_3_Hub'])
fli_boom = geometry.declare_component(component_name='fli_boom', b_spline_search_names=['Rotor_3_Support'])

# Rotor: front right inner
fri_disk = geometry.declare_component(component_name='fri_disk', b_spline_search_names=['Rotor_5_disk'])
fri_blade_1 = geometry.declare_component(component_name='fri_blade_1', b_spline_search_names=['Rotor_5_blades, 0'])
fri_blade_2 = geometry.declare_component(component_name='fri_blade_2', b_spline_search_names=['Rotor_5_blades, 1'])
fri_hub = geometry.declare_component(component_name='fri_hub', b_spline_search_names=['Rotor_5_Hub'])
fri_boom = geometry.declare_component(component_name='fri_boom', b_spline_search_names=['Rotor_5_Support'])

# Rotor: front right outer
fro_disk = geometry.declare_component(component_name='fro_disk', b_spline_search_names=['Rotor_7_disk'])
fro_blade_1 = geometry.declare_component(component_name='fro_blade_1', b_spline_search_names=['Rotor_7_blades, 0'])
fro_blade_2 = geometry.declare_component(component_name='fro_blade_2', b_spline_search_names=['Rotor_7_blades, 1'])
fro_hub = geometry.declare_component(component_name='fro_hub', b_spline_search_names=['Rotor_7_Hub'])
fro_boom = geometry.declare_component(component_name='fro_boom', b_spline_search_names=['Rotor_7_Support'])
# endregion

# Region geometric desing variables
# Wing
wing_area_input = system_model.create_input('wing_area_input', val=210., dv_flag=geometry_dv, lower=150, upper=250, scaler=8e-3)
wing_aspect_ratio_input = system_model.create_input('wing_aspect_ratio_input', val=12.12, dv_flag=geometry_dv, lower=8, upper=16, scaler=1e-1)
wing_taper_ratio_input = system_model.create_input('wing_taper_ratio_input', val=0.5)

wing_span_input = (wing_aspect_ratio_input * wing_area_input)**0.5
wing_root_chord_input = 2 * wing_area_input/((1 + wing_taper_ratio_input) * wing_span_input)
wing_tip_chord_left_input = wing_root_chord_input * wing_taper_ratio_input / 2.7
wing_tip_chord_right_input = wing_tip_chord_left_input * 1

# Tail
tail_area_input = system_model.create_input('tail_area_input', val=39.5, dv_flag=geometry_dv, lower=25, upper=55, scaler=1e-2)
tail_aspect_ratio_input = system_model.create_input('tail_aspect_ratio_input', val=4.3, dv_flag=geometry_dv, lower=3, upper=8, scaler=1e-1)
tail_taper_ratio_input = system_model.create_input('tail_taper_ratio_input', val=0.6)

tail_span_input = (tail_aspect_ratio_input * tail_area_input)**0.5
tail_root_chord_input = 2 * tail_area_input/((1 + tail_taper_ratio_input) * tail_span_input)
tail_tip_chord_left_input = tail_root_chord_input * tail_taper_ratio_input
tail_tip_chord_right_input = tail_tip_chord_left_input * 1

# Tail moment arm
tail_moment_arm_input = system_model.create_input(name='tail_moment_arm_input', val=17., dv_flag=False, lower=12., upper=22., scaler=1e-1)

# Radii
flo_radius = fro_radius = front_outer_radius = system_model.create_input(name='front_outer_radius', val=10/2, dv_flag=geometry_dv, lower=5/2, upper=15/2, scaler=1e-1)
fli_radius = fri_radius = front_inner_radius = system_model.create_input(name='front_inner_radius', val=10/2, dv_flag=geometry_dv, lower=5/2, upper=15/2, scaler=1e-1)
rlo_radius = rro_radius = rear_outer_radius = system_model.create_input(name='rear_outer_radius', val=10/2, dv_flag=geometry_dv, lower=5/2, upper=15/2, scaler=1e-1)
rli_radius = rri_radius = rear_inner_radius = system_model.create_input(name='rear_inner_radius', val=10/2, dv_flag=geometry_dv, lower=5/2, upper=15/2, scaler=1e-1)
dv_radius_list = [rlo_radius, rli_radius, rri_radius, rro_radius, flo_radius, fli_radius, fri_radius, fro_radius]

pusher_prop_radius = system_model.create_input(name='pusher_prop_radius', val=9/2, dv_flag=geometry_dv, lower=7/2, upper=11/2, scaler=1e-1)



wing_te_right = np.array([13.4, 25.250, 7.5])
wing_te_left = np.array([13.4, -25.250, 7.5])
wing_te_center = np.array([14.332, 0., 8.439])
wing_le_left = np.array([12.356, -25.25, 7.618])
wing_le_right = np.array([12.356, 25.25, 7.618])
wing_le_center = np.array([8.892, 0., 8.633])
wing_qc = np.array([10.25, 0., 8.5])

tail_te_right = np.array([31.5, 6.75, 6.])
tail_te_left = np.array([31.5, -6.75, 6.])
tail_le_right = np.array([26.5, 6.75, 6.])
tail_le_left = np.array([26.5, -6.75, 6.])
tail_te_center = np.array([31.187, 0., 8.009])
tail_le_center = np.array([27.428, 0., 8.009])
tail_qc = np.array([24.15, 0., 8.])


# region Making meshes
# Wing 
num_spanwise_vlm = 25
num_chordwise_vlm = 8


# wing_actuation_angle = system_model.create_input('wing_act_angle', val=0, dv_flag=geometry_dv, lower=-15, upper=15, scaler=1e-1)

wing_meshes = make_vlm_camber_mesh(
    geometry=geometry,
    wing_component=wing,
    num_spanwise=num_spanwise_vlm,
    num_chordwise=num_chordwise_vlm,
    te_right=wing_te_right,
    te_left=wing_te_left,
    te_center=wing_te_center,
    le_left=wing_le_left,
    le_right=wing_le_right,
    le_center=wing_le_center,
    grid_search_density_parameter=50,
    # actuation_axis=[
    #     0.75 * wing_te_right + 0.25 * wing_le_right,
    #     0.75 * wing_te_left + 0.25 * wing_le_left
    # ],
    # actuation_angle=wing_actuation_angle,
    off_set_x=0.2,
    bunching_cos=True,
    plot=False,
    mirror=True,
)


# tail mesh
num_spanwise_vlm_htail = 8
num_chordwise_vlm_htail = 4

actuation_axis=[
    0.5 * tail_te_right + 0.5 * tail_le_right,
    0.5 * tail_te_left + 0.5 * tail_le_left
]

axis_origin = geometry.evaluate(h_tail.project(actuation_axis[0]))
axis_vector = geometry.evaluate(h_tail.project(actuation_axis[1])) - axis_origin

tail_meshes = make_vlm_camber_mesh(
    geometry=geometry,
    wing_component=h_tail, 
    num_spanwise=num_spanwise_vlm_htail,
    num_chordwise=num_chordwise_vlm_htail,
    te_right=tail_te_right,
    te_left=tail_te_left,
    le_right=tail_le_right,
    le_left=tail_le_left,
    plot=False,
    mirror=True,
)

# v tail mesh
num_spanwise_vlm_vtail = 8
num_chordwise_vlm_vtail = 6
vtail_meshes = make_vlm_camber_mesh(
    geometry=geometry,
    wing_component=vtail,
    num_spanwise=num_spanwise_vlm_vtail,
    num_chordwise=num_chordwise_vlm_vtail,
    le_left=np.array([20.843, 0., 8.231,]),
    le_right=np.array([29.434, 0., 13.911,]),
    te_left=np.array([30.543, 0., 8.231]),
    te_right=np.array([32.065, 0., 13.911]),
    plot=False,
    orientation='vertical',
    zero_y=True,
)

# wing beam mesh
num_wing_beam = 21

box_beam_mesh = make_1d_box_beam_mesh(
    geometry=geometry,
    wing_component=wing,
    num_beam_nodes=num_wing_beam,
    te_right=wing_te_right,
    te_left=wing_te_left,
    te_center=wing_te_center,
    le_left=wing_le_left,
    le_right=wing_le_right,
    le_center=wing_le_center,
    beam_width=0.5,
    node_center=0.5,
    plot=False,
)

# Fuselage mesh
num_fuselage_len = 10
num_fuselage_height = 4
nose = np.array([2.464, 0., 5.113])
rear = np.array([31.889, 0., 7.798])
nose_points_parametric = fuselage.project(nose, grid_search_density_parameter=20)
rear_points_parametric = fuselage.project(rear)

nose_points_m3l = geometry.evaluate(nose_points_parametric)
rear_points_m3l = geometry.evaluate(rear_points_parametric)

fuselage_linspace = m3l.linspace(nose_points_m3l, rear_points_m3l, num_fuselage_len)

fueslage_top_points_parametric = fuselage.project(fuselage_linspace.value + np.array([0., 0., 3]), direction=np.array([0.,0.,-1.]), plot=False, grid_search_density_parameter=20)
fueslage_bottom_points_parametric = fuselage.project(fuselage_linspace.value - np.array([0., 0., 3]), direction=np.array([0.,0.,1.]), plot=False, grid_search_density_parameter=20)

fueslage_top_points_m3l = geometry.evaluate(fueslage_top_points_parametric)
fueslage_bottom_points_m3l = geometry.evaluate(fueslage_bottom_points_parametric)

fuesleage_mesh = m3l.linspace(fueslage_top_points_m3l.reshape((-1, 3)), fueslage_bottom_points_m3l.reshape((-1, 3)),  int(num_fuselage_height + 1))
fuesleage_mesh.description = 'zero_y'

# nose2 = np.array([3.439, 1.148, 6.589])
# rear2 = np.array([22.030, 1.148, 6.589])
# nose2_points = geometry.evaluate(fuselage.project(nose2, plot=False, grid_search_density_parameter=20))
# rear2_points = geometry.evaluate(fuselage.project(rear2, plot=False, grid_search_density_parameter=20))
# fuselage_linspace2 = m3l.linspace(nose2_points, rear2_points, num_fuselage_len)
# fuselage_left_points = geometry.evaluate(fuselage.project(fuselage_linspace2.value, direction=np.array([0., 1., 0.]), plot=False)).reshape((-1, 3))
# multiplier = np.ones(fuselage_left_points.shape)
# multiplier[:, 1] *= -1
# fuselage_right_points = fuselage_left_points * multiplier
# # fuselage_right_points = geometry.evaluate(fuselage.project(fuselage_linspace2.value - np.array([0., 5., 0.,]), direction=np.array([0., -1., 0.]), plot=False)).reshape((-1, 3))
# fuselage_mesh_2 = m3l.linspace(fuselage_left_points, fuselage_right_points, num_fuselage_height)# .reshape((num_fuselage_len, num_fuselage_height, 3))
# fuselage_mesh_2.description = 'average_z'

# region Rotor meshes
num_radial = 30
num_spanwise_vlm_rotor = 8
num_chord_vlm_rotor = 2

# Pusher prop
blade_1_params = BladeParameters(
    blade_component=pp_blade_1,
    point_on_leading_edge=np.array([31.649, 2.209, 7.411]),
    num_spanwise_vlm=num_spanwise_vlm_rotor,
    num_chordwise_vlm=num_chord_vlm_rotor,
)

blade_2_params = BladeParameters(
    blade_component=pp_blade_2,
    point_on_leading_edge=np.array([31.704, 0.421, 10.654]),
    num_spanwise_vlm=num_spanwise_vlm_rotor,
    num_chordwise_vlm=num_chord_vlm_rotor,
)

blade_3_params = BladeParameters(
    blade_component=pp_blade_3,
    point_on_leading_edge=np.array([31.853, -2.536, 8.270]),
    num_spanwise_vlm=num_spanwise_vlm_rotor,
    num_chordwise_vlm=num_chord_vlm_rotor,
)

blade_4_params = BladeParameters(
    blade_component=pp_blade_4,
    point_on_leading_edge=np.array([31.672, -0.408, 5.254]),
    num_spanwise_vlm=num_spanwise_vlm_rotor,
    num_chordwise_vlm=num_chord_vlm_rotor,
)

pp_mesh = make_rotor_mesh(
    geometry=geometry,
    num_radial=num_radial,
    disk_component=pp_disk,
    origin=np.array([32.625, 0., 7.79]),
    y1=np.array([31.94, 0.00, 3.29]),
    y2=np.array([31.94, 0.00, 12.29]),
    z1=np.array([31.94, -4.50, 7.78]),
    z2=np.array([31.94, 4.45, 7.77]),
    # blade_geometry_parameters=[blade_1_params, blade_2_params, blade_3_params, blade_4_params],
    create_disk_mesh=False,
    plot=False,
    # radius=pusher_prop_radius,
)


# Rear left outer
rlo_blade_1_params = BladeParameters(
    blade_component=rlo_blade_1,
    point_on_leading_edge=np.array([22.018, -19.243, 9.236]),
    num_spanwise_vlm=num_spanwise_vlm_rotor,
    num_chordwise_vlm=num_chord_vlm_rotor,
)

rlo_blade_2_params = BladeParameters(
    blade_component=rlo_blade_2,
    point_on_leading_edge=np.array([16.382, -18.257, 9.236]),
    num_spanwise_vlm=num_spanwise_vlm_rotor,
    num_chordwise_vlm=num_chord_vlm_rotor,
)

rlo_mesh = make_rotor_mesh(
    geometry=geometry,
    num_radial=num_radial,
    disk_component=rlo_disk,
    origin=np.array([19.2, -18.75, 9.01]),
    y1=np.array([19.2, -13.75, 9.01]),
    y2=np.array([19.2, -23.75, 9.01]),
    z1=np.array([14.2, -18.75, 9.01]),
    z2=np.array([24.2, -18.75, 9.01]),
    # blade_geometry_parameters=[rlo_blade_1_params, rlo_blade_2_params],
    create_disk_mesh=False,
    plot=False,
    # boom_is_thrust_origin=rlo_boom,
    # radius=rlo_radius,
)


# Rear right outer 
rro_blade_1_params = BladeParameters(
    blade_component=rro_blade_1,
    point_on_leading_edge=np.array([16.382, 18.257, 9.236]),
    num_spanwise_vlm=num_spanwise_vlm_rotor,
    num_chordwise_vlm=num_chord_vlm_rotor,
)

rro_blade_2_params = BladeParameters(
    blade_component=rro_blade_2,
    point_on_leading_edge=np.array([22.018, 19.195, 9.248]),
    num_spanwise_vlm=num_spanwise_vlm_rotor,
    num_chordwise_vlm=num_chord_vlm_rotor,
)

rro_mesh = make_rotor_mesh(
    geometry=geometry,
    num_radial=num_radial,
    disk_component=rro_disk,
    origin=np.array([19.2, 18.75, 9.01]),
    y1=np.array([19.2, 23.75, 9.01]),
    y2=np.array([19.2, 13.75, 9.01]),
    z1=np.array([14.2, 18.75, 9.01]),
    z2=np.array([24.2, 18.75, 9.01]),
    # blade_geometry_parameters=[rro_blade_1_params, rro_blade_2_params],
    create_disk_mesh=False,
    plot=False,
    # boom_is_thrust_origin=rro_boom,
    # radius=rro_radius,
)


# Front left outer 
flo_blade_1_params = BladeParameters(
    blade_component=flo_blade_1,
    point_on_leading_edge=np.array([7.888, -19.243, 6.956]),
    num_spanwise_vlm=num_spanwise_vlm_rotor,
    num_chordwise_vlm=num_chord_vlm_rotor,
)

flo_blade_2_params = BladeParameters(
    blade_component=flo_blade_2,
    point_on_leading_edge=np.array([2.252, -18.257, 6.956]),
    num_spanwise_vlm=num_spanwise_vlm_rotor,
    num_chordwise_vlm=num_chord_vlm_rotor,
)

flo_mesh = make_rotor_mesh(
    geometry=geometry,
    num_radial=num_radial,
    disk_component=flo_disk,
    origin=np.array([5.07, -18.75, 6.73]),
    y1=np.array([5.070, -13.750, 6.730]),
    y2=np.array([5.070, -23.750, 6.730]),
    z1=np.array([0.070, -18.750, 6.730]),
    z2=np.array([10.070, -18.750, 6.730]),
    # blade_geometry_parameters=[flo_blade_1_params, flo_blade_2_params],
    create_disk_mesh=False,
    plot=False,
    # boom_is_thrust_origin=flo_boom,
    # radius=flo_radius,
)


# Front right outer 
fro_blade_1_params = BladeParameters(
    blade_component=fro_blade_1,
    point_on_leading_edge=np.array([2.252, 18.257, 6.956]),
    num_spanwise_vlm=num_spanwise_vlm_rotor,
    num_chordwise_vlm=num_chord_vlm_rotor,
)

fro_blade_2_params = BladeParameters(
    blade_component=fro_blade_2,
    point_on_leading_edge=np.array([7.888, 19.243, 6.956]),
    num_spanwise_vlm=num_spanwise_vlm_rotor,
    num_chordwise_vlm=num_chord_vlm_rotor,
)

fro_mesh = make_rotor_mesh(
    geometry=geometry,
    num_radial=num_radial,
    disk_component=fro_disk,
    origin=np.array([5.07, 18.75, 6.73]),
    y1=np.array([5.070, 23.750, 6.730]),
    y2=np.array([5.070, 13.750, 6.730]),
    z1=np.array([0.070, 18.750, 6.730]),
    z2=np.array([10.070, 18.750, 6.730]),
    # blade_geometry_parameters=[fro_blade_1_params, fro_blade_2_params],
    create_disk_mesh=False,
    plot=False,
    # boom_is_thrust_origin=fro_boom,
    # radius=fro_radius,
)

# Rear left inner
rli_blade_1_params = BladeParameters(
    blade_component=rli_blade_1,
    point_on_leading_edge=np.array([15.578, -8.969, 9.437]),
    num_spanwise_vlm=num_spanwise_vlm_rotor,
    num_chordwise_vlm=num_chord_vlm_rotor,
)

rli_blade_2_params = BladeParameters(
    blade_component=rli_blade_2,
    point_on_leading_edge=np.array([21.578, -7.993, 9.593]),
    num_spanwise_vlm=num_spanwise_vlm_rotor,
    num_chordwise_vlm=num_chord_vlm_rotor,
)

rli_mesh = make_rotor_mesh(
    geometry=geometry,
    num_radial=num_radial,
    disk_component=rli_disk,
    origin=np.array([18.760, -8.537, 9.919]),
    y1=np.array([18.760, -3.499, 9.996]),
    y2=np.array([18.760, -13.401, 8.604]),
    z1=np.array([13.760, -8.450, 9.300]),
    z2=np.array([23.760, -8.450, 9.300]),
    # blade_geometry_parameters=[rli_blade_1_params, rli_blade_2_params],
    create_disk_mesh=False,
    plot=False,
    # boom_is_thrust_origin=rli_boom,
    # radius=rli_radius,
)


# Rear right inner
rri_blade_1_params = BladeParameters(
    blade_component=rri_blade_1,
    point_on_leading_edge=np.array([15.578, 8.969, 9.437]),
    num_spanwise_vlm=num_spanwise_vlm_rotor,
    num_chordwise_vlm=num_chord_vlm_rotor,
)

rri_blade_2_params = BladeParameters(
    blade_component=rri_blade_2,
    point_on_leading_edge=np.array([21.942, 7.989, 9.575]),
    num_spanwise_vlm=num_spanwise_vlm_rotor,
    num_chordwise_vlm=num_chord_vlm_rotor,
)

rri_mesh = make_rotor_mesh(
    geometry=geometry,
    num_radial=num_radial,
    disk_component=rri_disk,
    origin=np.array([18.760, 8.537, 9.919]),
    y1=np.array([18.760, 13.401, 8.604]),
    y2=np.array([18.760, 3.499, 9.996]),
    z1=np.array([13.760, 8.450, 9.300]),
    z2=np.array([23.760, 8.450, 9.300]),
    # blade_geometry_parameters=[rri_blade_1_params, rri_blade_2_params],
    create_disk_mesh=False,
    plot=False,
    # boom_is_thrust_origin=rri_boom,
    # radius=rri_radius,
)

# Front left inner
fli_blade_1_params = BladeParameters(
    blade_component=fli_blade_1,
    point_on_leading_edge=np.array([2.175, -8.634, 7.208]),
    num_spanwise_vlm=num_spanwise_vlm_rotor,
    num_chordwise_vlm=num_chord_vlm_rotor,
)

fli_blade_2_params = BladeParameters(
    blade_component=fli_blade_2,
    point_on_leading_edge=np.array([7.085, -7.692, 7.341]),
    num_spanwise_vlm=num_spanwise_vlm_rotor,
    num_chordwise_vlm=num_chord_vlm_rotor,
)

fli_mesh = make_rotor_mesh(
    geometry=geometry,
    num_radial=num_radial,
    disk_component=fli_disk,
    origin=np.array([4.630, -8.217, 7.659]),
    y1=np.array([4.630, -3.179, 7.736]),
    y2=np.array([4.630, -13.081, 6.344]),
    z1=np.array([-0.370, -8.130, 7.040]),
    z2=np.array([9.630, -8.130, 7.040]),
    # blade_geometry_parameters=[fli_blade_1_params, fli_blade_2_params],
    create_disk_mesh=False,
    plot=False,
    # boom_is_thrust_origin=fli_boom,
    # radius=fli_radius,
)

# Front right inner
fri_blade_1_params = BladeParameters(
    blade_component=fri_blade_1,
    point_on_leading_edge=np.array([7.448, 7.673, 7.333]),
    num_spanwise_vlm=num_spanwise_vlm_rotor,
    num_chordwise_vlm=num_chord_vlm_rotor,
)

fri_blade_2_params = BladeParameters(
    blade_component=fri_blade_2,
    point_on_leading_edge=np.array([1.085, 8.626, 7.155]),
    num_spanwise_vlm=num_spanwise_vlm_rotor,
    num_chordwise_vlm=num_chord_vlm_rotor,
)

fri_mesh = make_rotor_mesh(
    geometry=geometry,
    num_radial=num_radial,
    disk_component=fri_disk,
    origin=np.array([4.630, 8.217, 7.659]), 
    y1=np.array([4.630, 13.081, 6.344]),
    y2=np.array([4.630, 3.179, 7.736]),
    z1=np.array([-0.370, 8.130, 7.040]),
    z2=np.array([9.630, 8.130, 7.040]),
    # blade_geometry_parameters=[fri_blade_1_params, fri_blade_2_params],
    create_disk_mesh=False,
    plot=False,
    # boom_is_thrust_origin=fri_boom,
    # radius=fri_radius,
)

radius_1_list = [flo_mesh.radius, fli_mesh.radius, fri_mesh.radius, fro_mesh.radius,
                   rlo_mesh.radius, rli_mesh.radius, rri_mesh.radius, rro_mesh.radius]

radius_2_list = [flo_mesh._radius_2, fli_mesh._radius_2, fri_mesh._radius_2, fro_mesh._radius_2,
                   rlo_mesh._radius_2, rli_mesh._radius_2, rri_mesh._radius_2, rro_mesh._radius_2]




# endregion
# endregion


# region projections for drag components
# Fuselage 
fuselage_l1_parametric = fuselage.project(np.array([1.889, 0., 4.249]))
fuselage_l2_parametric = fuselage.project(np.array([31.889, 0., 7.798]))

fuselage_d1_parametric = fuselage.project(np.array([10.916, -2.945, 5.736]))
fuselage_d2_parametric = fuselage.project(np.array([10.916, 2.945, 5.736]))

# wing
wing_mid_le_parametric = wing.project(np.array([8.892, 0., 8.633]))
wing_mid_te_parametric = wing.project(np.array([14.332, 0, 8.439]))

wing_le_right_parametric = wing.project(wing_le_right)
wing_le_left__parametric = wing.project(wing_le_left)

# htail
h_tail_mid_le_parametric = h_tail.project(np.array([27.806, -6.520, 8.008]))
h_tail_mid_te_parametric = h_tail.project(np.array([30.050, -6.520, 8.008]))

# vtail
vtail_mid_le_parametric = vtail.project(np.array([26.971, 0.0, 11.038]))
vtail_mid_te_parametric = vtail.project(np.array([31.302, 0.0, 11.038]))

# boom
boom_l1_parametric = rlo_boom.project(np.array([20.000, -18.750, 7.613]))
boom_l2_parametric = rlo_boom.project(np.array([12.000, -18.750, 7.613]))

boom_d1_parametric = rlo_boom.project(np.array([15.600, -19.250, 7.613]))
boom_d2_parametric = rlo_boom.project(np.array([15.600, -18.250, 7.613]))

# lift hubs
hub_l1_parametric = rlo_hub.project(np.array([18.075, -18.750,9.525]))
hub_l2_parametric = rlo_hub.project(np.array([20.325, -18.750,9.525]))

# blade 
blade_tip_parametric = rlo_blade_2.project(np.array([14.200, -18.626, 9.040]))
blade_hub_parametric = rlo_blade_2.project(np.array([18.200, -18.512, 9.197]))
# endregion

if FFD:
    # region FFD
    # region Wing
    import time 
    constant_b_spline_curve_1_dof_space =  bsp.BSplineSpace(name='constant_b_spline_curve_1_dof_space', order=1, parametric_coefficients_shape=(1,))
    linear_b_spline_curve_2_dof_space = bsp.BSplineSpace(name='linear_b_spline_curve_2_dof_space', order=2, parametric_coefficients_shape=(2,))
    linear_b_spline_curve_3_dof_space = bsp.BSplineSpace(name='linear_b_spline_curve_3_dof_space', order=2, parametric_coefficients_shape=(3,))
    cubic_b_spline_curve_5_dof_space = bsp.BSplineSpace(name='cubic_b_spline_curve_5_dof_space', order=4, parametric_coefficients_shape=(5,))
    from lsdo_geo.core.parameterization.parameterization_solver import ParameterizationSolver
    lpc_param_solver = ParameterizationSolver()

    t1 = time.time()
    wing_ffd_block = lg.construct_ffd_block_around_entities(name='wing_ffd_block', entities=wing, num_coefficients=(2, 11, 2), order=(2, 4, 2))
    t2 = time.time()
    print(t2-t1)
    wing_ffd_block_sect_param = VolumeSectionalParameterization(name='wing_ffd_sect_param', principal_parametric_dimension=1, 
                                                                parameterized_points=wing_ffd_block.coefficients,
                                                                parameterized_points_shape=wing_ffd_block.coefficients_shape)
    t3 = time.time()
    print(t3-t2)


    wing_ffd_block_sect_param.add_sectional_translation(name='wing_span_stretch', axis=1)
    wing_ffd_block_sect_param.add_sectional_stretch(name='wing_chord_stretch', axis=0)


    wing_span_strech_coefficients = system_model.create_input('wing_span_stretch_coefficients', shape=(2, ), val=np.array([0., 0.]))
    wing_span_strech_b_spline = bsp.BSpline(name='wing_span_b_spline', space=linear_b_spline_curve_2_dof_space, 
                                            coefficients=wing_span_strech_coefficients, num_physical_dimensions=1)
        
    wing_chord_stretch_coefficients = system_model.create_input('wing_chord_stretch_coefficients', shape=(3, ), val=np.array([0., 0., 0.]))
    wing_chord_stretch_b_spline = bsp.BSpline(name='wing_chord_b_spline', space=linear_b_spline_curve_3_dof_space, coefficients=wing_chord_stretch_coefficients, 
                                            num_physical_dimensions=1)


    section_parametric_coordinates = np.linspace(0., 1., wing_ffd_block_sect_param.num_sections).reshape((-1,1))
    wing_wingspan_stretch = wing_span_strech_b_spline.evaluate(section_parametric_coordinates)
    wing_chord_stretch = wing_chord_stretch_b_spline.evaluate(section_parametric_coordinates)

    sectional_parameters = {
            'wing_span_stretch': wing_wingspan_stretch,
            'wing_chord_stretch': wing_chord_stretch,
    }


    wing_ffd_block_coefficients = wing_ffd_block_sect_param.evaluate(sectional_parameters, plot=False)
    wing_coefficients = wing_ffd_block.evaluate(wing_ffd_block_coefficients, plot=False)
    # endregion

    # region h-tail
    htail_ffd_block = lg.construct_ffd_block_around_entities(name='htail_ffd_block', entities=h_tail, num_coefficients=(2, 3, 2))
    htail_ffd_block_sect_param = VolumeSectionalParameterization(name='htail_ffd_sect_param', principal_parametric_dimension=1, 
                                                                parameterized_points=htail_ffd_block.coefficients,
                                                                parameterized_points_shape=htail_ffd_block.coefficients_shape)

    htail_ffd_block_sect_param.add_sectional_translation(name='htail_span_stretch', axis=1)
    htail_ffd_block_sect_param.add_sectional_stretch(name='htail_chord_stretch', axis=0)
    htail_ffd_block_sect_param.add_sectional_translation(name='htail_translation_x', axis=0)

    htail_span_stretch_coefficients = system_model.create_input('htail_span_stretch_coefficients', shape=(2, ), val=np.array([0., 0.]))
    htail_span_strech_b_spline = bsp.BSpline(name='htail_span_b_spline', space=linear_b_spline_curve_2_dof_space, 
                                            coefficients=htail_span_stretch_coefficients, num_physical_dimensions=1)
        
    htail_chord_stretch_coefficients = system_model.create_input('htail_chord_stretch_coefficients', shape=(3, ), val=np.array([0., 0., 0.]))
    htail_chord_stretch_b_spline = bsp.BSpline(name='htail_chord_b_spline', space=linear_b_spline_curve_3_dof_space, coefficients=htail_chord_stretch_coefficients, 
                                            num_physical_dimensions=1)

    htail_chord_transl_x_coefficients = system_model.create_input('htail_translation_x_coefficients', shape=(1, ), val=np.array([0.]))
    htail_chord_transl_x_b_spline = bsp.BSpline(name='htail_translation_x_b_spline', space=constant_b_spline_curve_1_dof_space, coefficients=htail_chord_transl_x_coefficients, 
                                            num_physical_dimensions=1)


    section_parametric_coordinates = np.linspace(0., 1., htail_ffd_block_sect_param.num_sections).reshape((-1,1))
    htail_span_stretch = htail_span_strech_b_spline.evaluate(section_parametric_coordinates)
    htail_chord_stretch = htail_chord_stretch_b_spline.evaluate(section_parametric_coordinates)
    htail_translation_x = htail_chord_transl_x_b_spline.evaluate(section_parametric_coordinates)

    sectional_parameters = {
            'htail_span_stretch': htail_span_stretch,
            'htail_chord_stretch': htail_chord_stretch,
            'htail_translation_x': htail_translation_x,
    }

    htail_ffd_block_coefficients = htail_ffd_block_sect_param.evaluate(sectional_parameters, plot=False)
    htail_coefficients = htail_ffd_block.evaluate(htail_ffd_block_coefficients, plot=False)
    # endregion
    
    # region fuselage
    fuselage_ffd_block = lg.construct_ffd_block_around_entities(name='fuselage_ffd_block', entities=fuselage, num_coefficients=(2, 2, 2))
    fuselage_ffd_block_sect_param = VolumeSectionalParameterization(name='fuselage_ffd_sect_param', principal_parametric_dimension=0,
                                                                    parameterized_points=fuselage_ffd_block.coefficients,
                                                                    parameterized_points_shape=fuselage_ffd_block.coefficients_shape)

    fuselage_ffd_block_sect_param.add_sectional_translation(name='fuselage_stretch', axis=0)
    fuselage_stretch_coeffs = system_model.create_input('fuselage_stretch_coefficients', shape=(2, ), val=np.array([0., 0.]))
    fuselage_stretch_b_spline = bsp.BSpline(name='fuselage_b_spline', space=linear_b_spline_curve_2_dof_space,
                                            coefficients=fuselage_stretch_coeffs, num_physical_dimensions=1)

    section_parametric_coordinates = np.linspace(0., 1., fuselage_ffd_block_sect_param.num_sections).reshape((-1, 1))
    fuselage_stretch = fuselage_stretch_b_spline.evaluate(section_parametric_coordinates)

    sectional_parameters = {
        'fuselage_stretch' : fuselage_stretch
    }

    fuselage_ffd_block_coefficients = fuselage_ffd_block_sect_param.evaluate(sectional_parameters, plot=False)
    fuselage_coefficients = fuselage_ffd_block.evaluate(fuselage_ffd_block_coefficients, plot=False)
    # endregion

    # region v-tail
    vtail_ffd_block = lg.construct_ffd_block_around_entities(name='vtail_ffd_block', entities=vtail, num_coefficients=(2, 2, 2))
    vtail_ffd_block_sect_param = VolumeSectionalParameterization(name='vtail_ffd_sect_param', principal_parametric_dimension=0,
                                                                parameterized_points=vtail_ffd_block.coefficients, parameterized_points_shape=vtail_ffd_block.coefficients_shape)

    vtail_ffd_block_sect_param.add_sectional_translation(name='vtail_translation_x', axis=0)
    vtail_transl_x_coeffs = system_model.create_input(name='vtail_transl_x_coefficients', shape=(1, ), val=np.array([0.]))
    vtail_transl_x_b_spline = bsp.BSpline(name='vtail_b_spline', space=constant_b_spline_curve_1_dof_space, coefficients=vtail_transl_x_coeffs, num_physical_dimensions=1)

    section_parametric_coordinates = np.linspace(0., 1., vtail_ffd_block_sect_param.num_sections).reshape((-1, 1))
    vtail_transl_x = vtail_transl_x_b_spline.evaluate(section_parametric_coordinates)

    sectional_parameters = {
        'vtail_translation_x' : vtail_transl_x
    }

    vtail_ffd_block_coefficients = vtail_ffd_block_sect_param.evaluate(sectional_parameters, plot=False)
    vtail_coefficients = vtail_ffd_block.evaluate(vtail_ffd_block_coefficients, plot=False)
    # endregion

    # region weird nose hub
    nose_hub_ffd_block = lg.construct_ffd_block_around_entities(name='weird_nose_hub_ffd_block', entities=nose_hub, num_coefficients=(2, 2, 2))
    nose_hub_ffd_block_sect_param = VolumeSectionalParameterization(name='weird_nose_hub_ffd_sect_param', principal_parametric_dimension=0,
                                                                    parameterized_points=nose_hub_ffd_block.coefficients, parameterized_points_shape=nose_hub_ffd_block.coefficients_shape)

    nose_hub_ffd_block_sect_param.add_sectional_translation(name='nose_hub_translation_x', axis=0)
    nose_hub_transl_x_coeffs = system_model.create_input(name='nose_hub_transl_x_coefficients', shape=(1, ), val=np.array([0.]))
    nose_hub_transl_x_b_spline = bsp.BSpline(name='nose_hub_b_spline', space=constant_b_spline_curve_1_dof_space, coefficients=nose_hub_transl_x_coeffs, num_physical_dimensions=1)

    section_parametric_coordinates = np.linspace(0., 1., nose_hub_ffd_block_sect_param.num_sections).reshape((-1, 1))
    nose_hub_transl_x = nose_hub_transl_x_b_spline.evaluate(section_parametric_coordinates)

    sectional_parameters = {
        'nose_hub_translation_x' : nose_hub_transl_x
    }

    nose_hub_ffd_block_coefficients = nose_hub_ffd_block_sect_param.evaluate(sectional_parameters, plot=False)
    nose_hub_coefficients = nose_hub_ffd_block.evaluate(nose_hub_ffd_block_coefficients, plot=False)
    # endregion

    # region Boom + hub + disk
    rlo_disk_pt = rlo_hub_pt = np.array([19.200, -18.750, 9.635])
    rro_disk_pt = rro_hub_pt = np.array([19.200, 18.750, 9.635])
    rlo_boom_pt = np.array([12.000, -18.750, 7.613])
    rro_boom_pt = np.array([12.000, 18.750, 7.613])

    flo_disk_pt = flo_hub_pt = np.array([5.070, -18.750, 7.355])
    fro_disk_pt = fro_hub_pt = np.array([5.070, 18.750, 7.355])
    flo_boom_pt = np.array([12.200, -18.750, 7.615])
    fro_boom_pt = np.array([12.200, 18.750, 7.615])

    rli_disk_pt = rli_hub_pt = np.array([18.760, -8.537, 9.919])
    rri_disk_pt = rri_hub_pt = np.array([18.760, 8.537, 9.919])
    rli_boom_pt = np.array([11.500, -8.250, 7.898])
    rri_boom_pt = np.array([11.500, 8.250, 7.898])

    fli_disk_pt = fli_hub_pt = np.array([4.630, -8.217, 7.659])
    fri_disk_pt = fri_hub_pt = np.array([4.630, 8.217, 7.659])
    fli_boom_pt = np.array([11.741, -8.250, 7.900])
    fri_boom_pt = np.array([11.741, 8.250, 7.900])

    hub_disk_components = [
        (rlo_disk, rlo_hub, rlo_blade_1, rlo_blade_2), (rli_disk, rli_hub, rli_blade_1, rli_blade_2), (rri_disk, rri_hub, rri_blade_1, rri_blade_2), (rro_disk, rro_hub, rro_blade_1, rro_blade_2), 
        (flo_disk, flo_hub, flo_blade_1, flo_blade_2), (fli_disk, fli_hub, fli_blade_1, fli_blade_2), (fri_disk, fri_hub, fri_blade_1, fri_blade_2), (fro_disk, fro_hub, fro_blade_1, fro_blade_2)
    ]

    points_on_disk = [rlo_disk_pt, rli_disk_pt, rri_disk_pt, rro_disk_pt, flo_disk_pt, fli_disk_pt, fri_disk_pt, fro_disk_pt]

    components_to_be_connected = [rlo_boom, rli_boom, rri_boom, rro_boom, flo_boom, fli_boom, fri_boom, fro_boom]
    # components_to_be_connected = [rlo_boom, rro_boom]
    # components_to_be_connected = [rro_boom]
    points_on_components = [rlo_boom_pt, rli_boom_pt, rri_boom_pt, rro_boom_pt, flo_boom_pt, fli_boom_pt, fri_boom_pt, fro_boom_pt]
    # points_on_components = [rlo_boom_pt, rro_boom_pt]
    # points_on_components = [rro_boom_pt]
    components_to_connect_to = wing
    prinicpal_parametric_dimension = 0
    prinicpal_parametric_dimension_booms = 0
    space = constant_b_spline_curve_1_dof_space
    plot = False

    boom_distances = []
    ffd_block_translation_x_sect_coeffs_list = []
    ffd_block_translation_y_sect_coeffs_list = []
    ffd_block_translation_z_sect_coeffs_list = []
    ffd_block_sect_param_list = []
    ffd_block_list = []
    booms_coefficients_list = []

    # Booms
    for i, comp in enumerate(components_to_be_connected):
        block_name = f'{comp.name}_ffd_block'
        ffd_block = lg.construct_ffd_block_around_entities(name=block_name, entities=comp, num_coefficients=(2, 2, 2))
        ffd_block_sect_param = VolumeSectionalParameterization(name=f'{block_name}_sect_param', principal_parametric_dimension=prinicpal_parametric_dimension_booms, 
                                                            parameterized_points=ffd_block.coefficients,
                                                            parameterized_points_shape=ffd_block.coefficients_shape)
        
        ffd_block_sect_param.add_sectional_translation(name=f'{block_name}_translation_x', axis=0)
        ffd_block_sect_param.add_sectional_translation(name=f'{block_name}_translation_y', axis=1)
        ffd_block_sect_param.add_sectional_translation(name=f'{block_name}_translation_z', axis=2)


        ffd_block_translation_x_sect_coeffs = system_model.create_input(f'{block_name}_translation_x_coefficients', shape=(1, ), val=np.array([0]))
        ffd_block_translation_y_sect_coeffs = system_model.create_input(f'{block_name}_translation_y_coefficients', shape=(1, ), val=np.array([0]))
        ffd_block_translation_z_sect_coeffs = system_model.create_input(f'{block_name}_translation_z_coefficients', shape=(1, ), val=np.array([0]))

        ffd_block_translation_x_sect_coeffs_list.append(ffd_block_translation_x_sect_coeffs)
        ffd_block_translation_y_sect_coeffs_list.append(ffd_block_translation_y_sect_coeffs)
        ffd_block_translation_z_sect_coeffs_list.append(ffd_block_translation_z_sect_coeffs)

        ffd_block_translation_x_b_spline = bsp.BSpline(name=f'{block_name}_translation_x_bspline', space=space, coefficients=ffd_block_translation_x_sect_coeffs, 
                                                        num_physical_dimensions=1)
        
        ffd_block_translation_y_b_spline = bsp.BSpline(name=f'{block_name}_translation_y_bspline', space=space, coefficients=ffd_block_translation_y_sect_coeffs, 
                                                        num_physical_dimensions=1)
        
        ffd_block_translation_z_b_spline = bsp.BSpline(name=f'{block_name}_translation_z_bspline', space=space, coefficients=ffd_block_translation_z_sect_coeffs, 
                                                        num_physical_dimensions=1)


        section_parametric_coordinates = np.linspace(0., 1., ffd_block_sect_param.num_sections).reshape((-1,1))
        comp_transl_x = ffd_block_translation_x_b_spline.evaluate(section_parametric_coordinates)
        comp_transl_y = ffd_block_translation_y_b_spline.evaluate(section_parametric_coordinates)
        comp_transl_z = ffd_block_translation_z_b_spline.evaluate(section_parametric_coordinates)

        sectional_parameters = {
                f'{block_name}_translation_x': comp_transl_x,
                f'{block_name}_translation_y': comp_transl_y,
                f'{block_name}_translation_z': comp_transl_z,
        }

        comp_ffd_block_coefficients = ffd_block_sect_param.evaluate(sectional_parameters, plot=plot)
        comp_coefficients = ffd_block.evaluate(comp_ffd_block_coefficients, plot=plot)
        booms_coefficients_list.append(comp_coefficients)
        ffd_block_sect_param_list.append(ffd_block_sect_param)
        ffd_block_list.append(ffd_block)

    # Lift rotors disk + Hub + blades
    hub_disk_distances = []
    hub_wing_distances = []
    ffd_block_translation_x_sect_coeffs_list_2 = []
    ffd_block_translation_y_sect_coeffs_list_2 = []
    ffd_block_translation_z_sect_coeffs_list_2 = []
    ffd_block_sect_param_list_2 = []
    ffd_block_list_2 = []
    ffd_block_stretch_x_sect_coeffs_list = []
    ffd_block_stretch_y_sect_coeffs_list = []
    rotor_coefficients_list = []

    for i, comp in enumerate(hub_disk_components):
        block_name = f'{comp[0].name}_{comp[1].name}'
        disk = comp[0]
        hub = comp[1]
        blade_1 = comp[2]
        blade_2 = comp[3]
        boom = components_to_be_connected[i]
        ffd_block = lg.construct_ffd_block_around_entities(name=block_name, entities=list(comp), num_coefficients=(2, 2, 2))
        ffd_block_sect_param = VolumeSectionalParameterization(name=f'{block_name}_sect_param', principal_parametric_dimension=prinicpal_parametric_dimension, 
                                                            parameterized_points=ffd_block.coefficients,
                                                            parameterized_points_shape=ffd_block.coefficients_shape)
        
        # Allow the lift rotors to move/translate
        ffd_block_sect_param.add_sectional_translation(name=f'{block_name}_translation_x', axis=0)
        ffd_block_sect_param.add_sectional_translation(name=f'{block_name}_translation_y', axis=1)
        ffd_block_sect_param.add_sectional_translation(name=f'{block_name}_translation_z', axis=2)

        # Allow the lift rotors to change radius
        # ffd_block_sect_param.add_sectional_stretch(name=f'{block_name}_stretch_x', axis=0)
        ffd_block_sect_param.add_sectional_stretch(name=f'{block_name}_stretch_y', axis=1)
        # ffd_block_sect_param.add_sectional_stretch(name=f'{block_name}_stretch_z', axis=2)

        ffd_block_translation_x_sect_coeffs = system_model.create_input(f'{block_name}_translation_x_coefficients', shape=(2, ), val=np.array([0, 0]))
        ffd_block_translation_y_sect_coeffs = system_model.create_input(f'{block_name}_translation_y_coefficients', shape=(1, ), val=np.array([0]))
        ffd_block_translation_z_sect_coeffs = system_model.create_input(f'{block_name}_translation_z_coefficients', shape=(1, ), val=np.array([0]))
        # ffd_block_stretch_x_sect_coeffs = system_model.create_input(f'{block_name}_stretch_x_coefficients', shape=(1, ), val=np.array([0.]))
        ffd_block_stretch_y_sect_coeffs = system_model.create_input(f'{block_name}_stretch_y_coefficients', shape=(1, ), val=np.array([0]))
        # ffd_block_stretch_z_sect_coeffs = system_model.create_input(f'{block_name}_stretch_z_coefficients', shape=(1, ), val=np.array([10.]))

        ffd_block_translation_x_sect_coeffs_list_2.append(ffd_block_translation_x_sect_coeffs)
        ffd_block_translation_y_sect_coeffs_list_2.append(ffd_block_translation_y_sect_coeffs)
        ffd_block_translation_z_sect_coeffs_list_2.append(ffd_block_translation_z_sect_coeffs)
        ffd_block_stretch_y_sect_coeffs_list.append(ffd_block_stretch_y_sect_coeffs)

        time_b_spline_1 = time.time()
        ffd_block_translation_x_b_spline = bsp.BSpline(name=f'{block_name}_translation_x_bspline', space=linear_b_spline_curve_2_dof_space, coefficients=ffd_block_translation_x_sect_coeffs, 
                                                        num_physical_dimensions=1)
        
        ffd_block_translation_y_b_spline = bsp.BSpline(name=f'{block_name}_translation_y_bspline', space=space, coefficients=ffd_block_translation_y_sect_coeffs, 
                                                        num_physical_dimensions=1)
        
        ffd_block_translation_z_b_spline = bsp.BSpline(name=f'{block_name}_translation_z_bspline', space=space, coefficients=ffd_block_translation_z_sect_coeffs, 
                                                        num_physical_dimensions=1)
        
        # ffd_block_stretch_x_b_spline = bsp.BSpline(name=f'{block_name}_stretch_x_bspline', space=linear_b_spline_curve_2_dof_space, coefficients=ffd_block_stretch_x_sect_coeffs, 
        #                                                 num_physical_dimensions=1)
        
        ffd_block_stretch_y_b_spline = bsp.BSpline(name=f'{block_name}_stretch_y_bspline', space=space, coefficients=ffd_block_stretch_y_sect_coeffs, 
                                                        num_physical_dimensions=1)
        
        # ffd_block_stretch_z_b_spline = bsp.BSpline(name=f'{block_name}_stretch_z_bspline', space=space, coefficients=ffd_block_stretch_z_sect_coeffs, 
        #                                                 num_physical_dimensions=1)

        section_parametric_coordinates = np.linspace(0., 1., ffd_block_sect_param.num_sections).reshape((-1,1))
        comp_transl_x = ffd_block_translation_x_b_spline.evaluate(section_parametric_coordinates)
        comp_transl_y = ffd_block_translation_y_b_spline.evaluate(section_parametric_coordinates)
        comp_transl_z = ffd_block_translation_z_b_spline.evaluate(section_parametric_coordinates)
        # comp_stretch_x = ffd_block_stretch_x_b_spline.evaluate(section_parametric_coordinates)
        comp_stretch_y = ffd_block_stretch_y_b_spline.evaluate(section_parametric_coordinates)
        # comp_stretch_z = ffd_block_stretch_y_b_spline.evaluate(section_parametric_coordinates)

        sectional_parameters = {
                f'{block_name}_translation_x': comp_transl_x,
                f'{block_name}_translation_y': comp_transl_y,
                f'{block_name}_translation_z': comp_transl_z,
                # f'{block_name}_stretch_x': comp_stretch_x,
                f'{block_name}_stretch_y': comp_stretch_y,
                # f'{block_name}_stretch_z': comp_stretch_z,
        }

        # ffd_block.plot()

        comp_ffd_block_coefficients = ffd_block_sect_param.evaluate(sectional_parameters, plot=plot)
        comp_coefficients = ffd_block.evaluate(comp_ffd_block_coefficients, plot=plot)
        rotor_coefficients_list.append(comp_coefficients)
        ffd_block_sect_param_list_2.append(ffd_block_sect_param)
        ffd_block_list_2.append(ffd_block)

    # Pusher prop disk + hub + blades
    pusher_prop_ffd_block = lg.construct_ffd_block_around_entities(name='pusher_prop_ffd_block', entities=[pp_disk, pp_hub, pp_blade_1, pp_blade_2, pp_blade_3, pp_blade_4], num_coefficients=(2, 2, 2))
    pusher_prop_ffd_block_sect_param = VolumeSectionalParameterization(name='pusher_prop_ffd_block_sect_param', principal_parametric_dimension=0, 
                                                            parameterized_points=pusher_prop_ffd_block.coefficients,
                                                            parameterized_points_shape=pusher_prop_ffd_block.coefficients_shape)

    pusher_prop_ffd_block_sect_param.add_sectional_translation(name='pusher_prop_ffd_block_translation_x', axis=0)
    pusher_prop_ffd_block_translation_x_sect_coeffs = system_model.create_input('pusher_prop_ffd_block_translation_x_coefficients', shape=(1, ), val=np.array([0]))
    pusher_prop_ffd_block_translation_x_b_spline = bsp.BSpline(name='pusher_prop_ffd_block_translation_x_bspline', space=space, coefficients=pusher_prop_ffd_block_translation_x_sect_coeffs, 
                                                        num_physical_dimensions=1)

    pusher_prop_ffd_block_sect_param.add_sectional_stretch(name='pusher_prop_ffd_block_sectional_stretch_y', axis=1)
    pusher_prop_ffd_block_stretch_y_coefficients = system_model.create_input('pusher_prop_ffd_block_stretch_y_coefficients', shape=(1, ), val=np.array([0]))
    pusher_prop_ffd_block_stretch_y_b_spline = bsp.BSpline(name='pusher_prop_ffd_block_stretch_y_bspline', space=space, coefficients=pusher_prop_ffd_block_stretch_y_coefficients, 
                                                        num_physical_dimensions=1)

    pusher_prop_ffd_block_sect_param.add_sectional_stretch(name='pusher_prop_ffd_block_sectional_stretch_z', axis=2)
    pusher_prop_ffd_block_stretch_z_coefficients = system_model.create_input('pusher_prop_ffd_block_stretch_z_coefficients', shape=(1, ), val=np.array([0]))
    pusher_prop_ffd_block_stretch_z_b_spline = bsp.BSpline(name='pusher_prop_ffd_block_stretch_z_bspline', space=space, coefficients=pusher_prop_ffd_block_stretch_z_coefficients, 
                                                        num_physical_dimensions=1)

    section_parametric_coordinates = np.linspace(0., 1., pusher_prop_ffd_block_sect_param.num_sections).reshape((-1,1))
    pusher_prop_transl_x = pusher_prop_ffd_block_translation_x_b_spline.evaluate(section_parametric_coordinates)
    pusher_prop_stretch_y = pusher_prop_ffd_block_stretch_y_b_spline.evaluate(section_parametric_coordinates)
    pusher_prop_stretch_z = pusher_prop_ffd_block_stretch_z_b_spline.evaluate(section_parametric_coordinates)

    sectional_parameters = {
        'pusher_prop_ffd_block_translation_x': pusher_prop_transl_x,
        'pusher_prop_ffd_block_sectional_stretch_y': pusher_prop_stretch_y,
        'pusher_prop_ffd_block_sectional_stretch_z': pusher_prop_stretch_z,
    }

    pusher_prop_ffd_block_coefficients = pusher_prop_ffd_block_sect_param.evaluate(sectional_parameters, plot=plot)
    pusher_prop_coefficients = pusher_prop_ffd_block.evaluate(pusher_prop_ffd_block_coefficients, plot=plot)


    # Assigning coefficients + Setting up inner optimization inputs/constraints
    ################################################################
    # endregion
    
    # Assign coefficients
    coefficients_list = []
    coefficients_list.append(wing_coefficients)
    coefficients_list.append(fuselage_coefficients)
    coefficients_list.append(htail_coefficients)
    coefficients_list.append(vtail_coefficients)
    coefficients_list.append(nose_hub_coefficients)

    b_spline_names_list = []
    b_spline_names_list.append(wing.b_spline_names)
    b_spline_names_list.append(fuselage.b_spline_names)
    b_spline_names_list.append(h_tail.b_spline_names)
    b_spline_names_list.append(vtail.b_spline_names)
    b_spline_names_list.append(nose_hub.b_spline_names)

    geometry.assign_coefficients(coefficients=wing_coefficients, b_spline_names=wing.b_spline_names)
    geometry.assign_coefficients(coefficients=fuselage_coefficients, b_spline_names=fuselage.b_spline_names)
    geometry.assign_coefficients(coefficients=htail_coefficients, b_spline_names=h_tail.b_spline_names)
    geometry.assign_coefficients(coefficients=vtail_coefficients, b_spline_names=vtail.b_spline_names)
    geometry.assign_coefficients(coefficients=nose_hub_coefficients, b_spline_names=nose_hub.b_spline_names)

    for i, comp in enumerate(components_to_be_connected):
        # geometry.assign_coefficients(coefficients=comp_coefficients, b_spline_names=comp.b_spline_names)
        coefficients_list.append(booms_coefficients_list[i])
        b_spline_names_list.append(comp.b_spline_names)

    for i, comp in enumerate(hub_disk_components):
        disk = comp[0]
        hub = comp[1]
        blade_1 = comp[2]
        blade_2 = comp[3]
        boom = components_to_be_connected[i]

        coefficients_list.append(rotor_coefficients_list[i][disk.name + '_coefficients'])
        coefficients_list.append(rotor_coefficients_list[i][hub.name + '_coefficients'])
        coefficients_list.append(rotor_coefficients_list[i][blade_1.name + '_coefficients'])
        coefficients_list.append(rotor_coefficients_list[i][blade_2.name + '_coefficients'])
        
        b_spline_names_list.append(disk.b_spline_names)
        b_spline_names_list.append(hub.b_spline_names)
        b_spline_names_list.append(blade_1.b_spline_names)
        b_spline_names_list.append(blade_2.b_spline_names)

        # geometry.assign_coefficients(coefficients=comp_coefficients[disk.name + '_coefficients'], b_spline_names=disk.b_spline_names)
        # geometry.assign_coefficients(coefficients=comp_coefficients[hub.name + '_coefficients'], b_spline_names=hub.b_spline_names)
        # geometry.assign_coefficients(coefficients=comp_coefficients[blade_1.name + '_coefficients'], b_spline_names=blade_1.b_spline_names)
        # geometry.assign_coefficients(coefficients=comp_coefficients[blade_2.name + '_coefficients'], b_spline_names=blade_2.b_spline_names)

    geometry.assign_coefficients(coefficients=pusher_prop_coefficients[pp_disk.name + '_coefficients'], b_spline_names=pp_disk.b_spline_names)
    geometry.assign_coefficients(coefficients=pusher_prop_coefficients[pp_hub.name + '_coefficients'], b_spline_names=pp_hub.b_spline_names)
    geometry.assign_coefficients(coefficients=pusher_prop_coefficients[pp_blade_1.name + '_coefficients'], b_spline_names=pp_blade_1.b_spline_names)
    geometry.assign_coefficients(coefficients=pusher_prop_coefficients[pp_blade_2.name + '_coefficients'], b_spline_names=pp_blade_2.b_spline_names)
    geometry.assign_coefficients(coefficients=pusher_prop_coefficients[pp_blade_3.name + '_coefficients'], b_spline_names=pp_blade_3.b_spline_names)
    geometry.assign_coefficients(coefficients=pusher_prop_coefficients[pp_blade_4.name + '_coefficients'], b_spline_names=pp_blade_4.b_spline_names)

    coefficients_list.append(pusher_prop_coefficients[pp_disk.name + '_coefficients'])
    coefficients_list.append(pusher_prop_coefficients[pp_hub.name + '_coefficients'])
    coefficients_list.append(pusher_prop_coefficients[pp_blade_1.name + '_coefficients'])
    coefficients_list.append(pusher_prop_coefficients[pp_blade_2.name + '_coefficients'])
    coefficients_list.append(pusher_prop_coefficients[pp_blade_3.name + '_coefficients'])
    coefficients_list.append(pusher_prop_coefficients[pp_blade_4.name + '_coefficients'])

    b_spline_names_list.append(pp_disk.b_spline_names)
    b_spline_names_list.append(pp_hub.b_spline_names)
    b_spline_names_list.append(pp_blade_1.b_spline_names)
    b_spline_names_list.append(pp_blade_2.b_spline_names)
    b_spline_names_list.append(pp_blade_3.b_spline_names)
    b_spline_names_list.append(pp_blade_4.b_spline_names)

    geometry.assign_coefficients(coefficients=coefficients_list, b_spline_names=b_spline_names_list)

    # geometry.plot()

    # Evaluations 
    wing_te_right_m3l = geometry.evaluate(wing.project(wing_te_right))
    wing_te_left_m3l = geometry.evaluate(wing.project(wing_te_left))
    wing_te_center_m3l = geometry.evaluate(wing.project(wing_te_center))
    wing_le_left_m3l = geometry.evaluate(wing.project(wing_le_left))
    wing_le_right_m3l = geometry.evaluate(wing.project(wing_le_right))
    wing_le_center_m3l = geometry.evaluate(wing.project(wing_le_center))
    wing_span = m3l.norm(wing_te_right_m3l - wing_te_left_m3l)
    root_chord = m3l.norm(wing_te_center_m3l-wing_le_center_m3l)
    tip_chord_left = m3l.norm(wing_te_left_m3l-wing_le_left_m3l)
    tip_chord_right = m3l.norm(wing_te_right_m3l-wing_le_right_m3l)


    fuselage_front = geometry.evaluate(fuselage.project(points=np.array([1.889, 0., -0.175])))
    fuselage_rear = geometry.evaluate(fuselage.project(points=np.array([31.889, 0., 7.798])))
    fuselage_length = m3l.norm(fuselage_rear-fuselage_front)


    tail_te_right_m3l = geometry.evaluate(h_tail.project(tail_te_right))
    tail_te_left_m3l = geometry.evaluate(h_tail.project(tail_te_left))
    tail_le_right_m3l = geometry.evaluate(h_tail.project(tail_le_right))
    tail_le_left_m3l = geometry.evaluate(h_tail.project(tail_le_left))
    tail_te_center_m3l = geometry.evaluate(h_tail.project(tail_te_center))
    tail_le_center_m3l = geometry.evaluate(h_tail.project(tail_le_center))
    htail_span = m3l.norm(tail_te_right_m3l - tail_te_left_m3l)
    htail_tip_chord_left = m3l.norm(tail_te_left_m3l - tail_le_left_m3l)
    htail_tip_chord_right = m3l.norm(tail_te_right_m3l - tail_le_right_m3l)
    htail_root_chord = m3l.norm(tail_te_center_m3l - tail_le_center_m3l)

    htail_qc_m3l = geometry.evaluate(h_tail.project(tail_te_center))
    wing_qc_m3l = geometry.evaluate(wing.project(tail_te_center))

    tail_moment_arm = m3l.norm(htail_qc_m3l - wing_qc_m3l)
    # tail_moment_arm = htail_qc_m3l - wing_qc_m3l


    vtail_point_m3l = geometry.evaluate(vtail.project(np.array([30.543, 0., 8.231])))
    vtail_minus_fuselage = m3l.norm(vtail_point_m3l - fuselage_rear)

    wing_point_on_fuselage_m3l = geometry.evaluate(fuselage.project(wing_le_center))
    wing_point_on_wing_m3l = geometry.evaluate(wing.project(wing_le_center))
    wing_fuselage_connection = m3l.norm(wing_point_on_fuselage_m3l-wing_point_on_wing_m3l)
    # wing_fuselage_connection = (wing_point_on_fuselage_m3l-wing_point_on_wing_m3l)

    h_tail_point_on_fuselage_m3l = geometry.evaluate(fuselage.project(tail_te_center))
    h_tail_point_on_htail_m3l = geometry.evaluate(h_tail.project(tail_te_center))
    h_tail_fuselage_connection = m3l.norm(h_tail_point_on_fuselage_m3l-h_tail_point_on_htail_m3l)

    nose_hub_point_m3l = geometry.evaluate(nose_hub.project(np.array([2.250, 0., 4.150])))
    fuselage_minus_nose_hub = m3l.norm(fuselage_front-nose_hub_point_m3l)

    for i, comp in enumerate(components_to_be_connected):
        point_on_component = points_on_components[i]
        point_on_component_m3l = geometry.evaluate(comp.project(point_on_component, plot=False))
        point_on_component_to_be_connecte_to_m3l = geometry.evaluate(components_to_connect_to.project(point_on_component, plot=False))

        # euclidean_distance = m3l.norm(point_on_component_m3l-point_on_component_to_be_connecte_to_m3l)
        euclidean_distance = point_on_component_m3l-point_on_component_to_be_connecte_to_m3l
        boom_distances.append(euclidean_distance)

        # euclidean_distance = point_on_component_m3l-point_on_component_to_be_connecte_to_m3l
        # boom_distances.append(euclidean_distance)
        


    for i, comp in enumerate(hub_disk_components):
    #     disk = comp[0]
        hub = comp[1]
        blade_1 = comp[2]
        blade_2 = comp[3]
        boom = components_to_be_connected[i]

        point_on_disk = points_on_disk[i]
        point_on_hub_m3l = geometry.evaluate(hub.project(point_on_disk))
        # point_on_wing_m3l = geometry.evaluate(wing.project(point_on_disk))
        point_on_boom_m3l = geometry.evaluate(boom.project(point_on_disk))
        hub_minus_boom = point_on_hub_m3l - point_on_boom_m3l # or hub to boom

        hub_minus_boom_dist = m3l.norm(hub_minus_boom)

        # hub_disk_distances.append(hub_minus_disk_dist)
        # hub_wing_distances.append(hub_minus_boom_dist)
        
        hub_wing_distances.append(hub_minus_boom)
    
        
        # ffd_block_stretch_x_sect_coeffs_list.append(ffd_block_stretch_x_sect_coeffs)
        


    pusher_prop_point_m3l = geometry.evaluate(pp_disk.project(np.array([32.625, 0., 7.79])))
    pusher_prop_minus_fuselage = m3l.norm(pusher_prop_point_m3l - fuselage_rear)

    pp_mesh = pp_mesh.update(geometry=geometry)
    rlo_mesh = rlo_mesh.update(geometry=geometry)
    rro_mesh = rro_mesh.update(geometry=geometry)
    flo_mesh = flo_mesh.update(geometry=geometry)
    fro_mesh = fro_mesh.update(geometry=geometry)
    rli_mesh = rli_mesh.update(geometry=geometry)
    rri_mesh = rri_mesh.update(geometry=geometry)
    fli_mesh = fli_mesh.update(geometry=geometry)
    fri_mesh = fri_mesh.update(geometry=geometry)

    radius_1_list = [flo_mesh.radius, fli_mesh.radius, fri_mesh.radius, fro_mesh.radius,
                    rlo_mesh.radius, rli_mesh.radius, rri_mesh.radius, rro_mesh.radius]

    radius_2_list = [flo_mesh._radius_2, fli_mesh._radius_2, fri_mesh._radius_2, fro_mesh._radius_2,
                    rlo_mesh._radius_2, rli_mesh._radius_2, rri_mesh._radius_2, rro_mesh._radius_2]

    ################################################################

    # Declaring states and input to the inner optimization
    lpc_param_solver.declare_state('wing_span_stretch_coefficients', wing_span_strech_coefficients, penalty_factor=1000)
    lpc_param_solver.declare_state('wing_chord_stretch_coefficients', wing_chord_stretch_coefficients)
    lpc_param_solver.declare_state('htail_span_stretch_coefficients', htail_span_stretch_coefficients, penalty_factor=1000)
    lpc_param_solver.declare_state('htail_chord_stretch_coefficients', htail_chord_stretch_coefficients)
    lpc_param_solver.declare_state('fuselage_stretch_coefficients', fuselage_stretch_coeffs)
    lpc_param_solver.declare_state('htail_translation_x_coefficients', htail_chord_transl_x_coefficients, penalty_factor=1e-5)
    lpc_param_solver.declare_state('nose_hub_transl_x_coefficients', nose_hub_transl_x_coeffs)
    lpc_param_solver.declare_state('vtail_transl_x_coefficients', vtail_transl_x_coeffs, penalty_factor=1)
    lpc_param_solver.declare_state('pusher_prop_ffd_block_translation_x_coefficients', pusher_prop_ffd_block_translation_x_sect_coeffs)
    lpc_param_solver.declare_state('pusher_prop_ffd_block_stretch_y_coefficients', pusher_prop_ffd_block_stretch_y_coefficients)
    lpc_param_solver.declare_state('pusher_prop_ffd_block_stretch_z_coefficients', pusher_prop_ffd_block_stretch_z_coefficients)


    lpc_param_solver.declare_input(name=wing_span_input.name, input=wing_span)
    lpc_param_solver.declare_input(name=wing_tip_chord_left_input.name, input=tip_chord_left)
    lpc_param_solver.declare_input(name=wing_tip_chord_right_input.name, input=tip_chord_right)
    lpc_param_solver.declare_input(name=wing_root_chord_input.name, input=root_chord)
    lpc_param_solver.declare_input(name=tail_tip_chord_left_input.name, input=htail_tip_chord_left)
    lpc_param_solver.declare_input(name=tail_tip_chord_right_input.name, input=htail_tip_chord_right)
    lpc_param_solver.declare_input(name=tail_root_chord_input.name, input=htail_root_chord)
    lpc_param_solver.declare_input(name=tail_span_input.name, input=htail_span)
    # lpc_param_solver.declare_input(name=tail_moment_arm_input.name, input=tail_moment_arm)
    lpc_param_solver.declare_input(name='nose_hub_to_fuselage_connection', input=fuselage_minus_nose_hub)
    lpc_param_solver.declare_input(name='vtail_to_fuselage_connection', input=vtail_minus_fuselage)
    lpc_param_solver.declare_input(name='pusher_prop_to_fuselage_connection', input=pusher_prop_minus_fuselage)
    lpc_param_solver.declare_input(name='pusher_prop_r1', input=pp_mesh.radius)
    lpc_param_solver.declare_input(name='pusher_prop_r2', input=pp_mesh._radius_2)
    lpc_param_solver.declare_input(name='wing_fuselage_connection', input=wing_fuselage_connection)
    lpc_param_solver.declare_input(name='h_tail_fuselage_connection', input=h_tail_fuselage_connection)




    nose_hub_to_fuselage_connection_input = m3l.Variable(name='nose_hub_to_fuselage_connection', shape=(1, ), value=fuselage_minus_nose_hub.value)
    vtail_to_fuselage_connection_input = m3l.Variable(name='vtail_to_fuselage_connection', shape=(1, ), value=vtail_minus_fuselage.value)
    pusher_prop_to_fuselage_connection_input = m3l.Variable(name='pusher_prop_to_fuselage_connection', shape=(1, ), value=pusher_prop_minus_fuselage.value)
    pusher_prop_r1_input = m3l.Variable(name='pusher_prop_r1', shape=(1, ), value=pusher_prop_radius.value)
    pusher_prop_r2_input = m3l.Variable(name='pusher_prop_r2', shape=(1, ), value=pusher_prop_radius.value)
    wing_fuselage_connection_input = m3l.Variable(name='wing_fuselage_connection', shape=(1, ), value=wing_fuselage_connection.value)
    h_tail_fuselage_connection_input = m3l.Variable(name='h_tail_fuselage_connection', shape=(1, ), value=h_tail_fuselage_connection.value)


    parameterization_inputs = {
        wing_span_input.name : wing_span_input,
        wing_tip_chord_left_input.name : wing_tip_chord_left_input,
        wing_tip_chord_right_input.name : wing_tip_chord_right_input,
        wing_root_chord_input.name : wing_root_chord_input,
        tail_tip_chord_left_input.name : tail_tip_chord_left_input,
        tail_tip_chord_right_input.name : tail_tip_chord_right_input,
        tail_root_chord_input.name : tail_root_chord_input,
        tail_span_input.name : tail_span_input,
        # tail_moment_arm_input.name : tail_moment_arm_input,
        'nose_hub_to_fuselage_connection' : nose_hub_to_fuselage_connection_input,
        'vtail_to_fuselage_connection' : vtail_to_fuselage_connection_input,
        'pusher_prop_to_fuselage_connection' : pusher_prop_to_fuselage_connection_input,
        'pusher_prop_r1' : pusher_prop_r1_input,
        'pusher_prop_r2' : pusher_prop_r2_input,
        'wing_fuselage_connection' : wing_fuselage_connection_input,
        'h_tail_fuselage_connection' : h_tail_fuselage_connection_input,
    }

    # Booms
    for i, comp in enumerate(components_to_be_connected):
        block_name = f'{comp.name}_ffd_block'
        lpc_param_solver.declare_state(name=f'{block_name}_translation_x_coefficients', state=ffd_block_translation_x_sect_coeffs_list[i], penalty_factor=1)
        lpc_param_solver.declare_state(name=f'{block_name}_translation_y_coefficients', state=ffd_block_translation_y_sect_coeffs_list[i], penalty_factor=1)
        lpc_param_solver.declare_state(name=f'{block_name}_translation_z_coefficients', state=ffd_block_translation_z_sect_coeffs_list[i], penalty_factor=1)

        lpc_param_solver.declare_input(name=f'{block_name}_distance_to_be_enforced', input=boom_distances[i])
        component_distance_input = m3l.Variable(name=f'{block_name}_distance_to_be_enforced', shape=(3, ), value=boom_distances[i].value)
        # component_distance_input = m3l.Variable(name=f'{block_name}_distance_to_be_enforced', shape=(3, ), value=boom_distances[i].value)
        # system_model.create_input(name=f'{block_name}_distance_to_be_enforced', shape=(1, ), val=boom_distances[i].value)

        parameterization_inputs[f'{block_name}_distance_to_be_enforced'] = component_distance_input

    # Hub + Disk + blades
    for i, comp in enumerate(hub_disk_components):
        block_name = f'{comp[0].name}_{comp[1].name}'
        lpc_param_solver.declare_state(name=f'{block_name}_translation_x_coefficients', state=ffd_block_translation_x_sect_coeffs_list_2[i], penalty_factor=1)
        lpc_param_solver.declare_state(name=f'{block_name}_translation_y_coefficients', state=ffd_block_translation_y_sect_coeffs_list_2[i], penalty_factor=1)
        lpc_param_solver.declare_state(name=f'{block_name}_translation_z_coefficients', state=ffd_block_translation_z_sect_coeffs_list_2[i], penalty_factor=1)
        # lpc_param_solver.declare_state(name=f'{block_name}_stretch_x_coefficients', state=ffd_block_stretch_x_sect_coeffs_list[i], penalty_factor=1)
        lpc_param_solver.declare_state(name=f'{block_name}_stretch_y_coefficients', state=ffd_block_stretch_y_sect_coeffs_list[i], penalty_factor=1)

        lpc_param_solver.declare_input(name=f'{block_name}_hub_wing_distance_to_be_enforced', input=hub_wing_distances[i])
        lpc_param_solver.declare_input(name=f'{block_name}_r1', input=radius_1_list[i])
        lpc_param_solver.declare_input(name=f'{block_name}_r2', input=radius_2_list[i])
        
        component_distance_input_2 = m3l.Variable(name=f'{block_name}_hub_wing_distance_to_be_enforced', shape=(3, ), value=hub_wing_distances[i].value)
        r1_input = m3l.Variable(name=f'{block_name}_r1', shape=(1, ), value=dv_radius_list[i].value)
        r2_input = m3l.Variable(name=f'{block_name}_r2', shape=(1, ), value=dv_radius_list[i].value)

        parameterization_inputs[f'{block_name}_hub_wing_distance_to_be_enforced'] = component_distance_input_2
        parameterization_inputs[f'{block_name}_r1'] = r1_input
        parameterization_inputs[f'{block_name}_r2'] = r2_input




    # Evaluate parameterization solver
    outputs_dict = lpc_param_solver.evaluate(inputs=parameterization_inputs, plot=False)

    # RE-ASSIGNING COEFFICIENTS AFTER INNER OPTIMIZATION
 
    coefficients_list = []
    # b_spline_names_list = []
    # region Wing
    wing_span_stretch_coefficients = outputs_dict['wing_span_stretch_coefficients']
    left_wing_wingspan_stretch_b_spline = bsp.BSpline(name='wingspan_stretch_b_spline', space=linear_b_spline_curve_2_dof_space,
                                                        coefficients=wing_span_stretch_coefficients, num_physical_dimensions=1)

    wing_chord_stretch_coefficients = outputs_dict['wing_chord_stretch_coefficients']
    wing_chord_stretch_b_spline = bsp.BSpline(name='wing_chord_stretch_b_spline', space=linear_b_spline_curve_3_dof_space,
                                                    coefficients=wing_chord_stretch_coefficients, num_physical_dimensions=1)
    wing_twist_coefficients = m3l.Variable(name='wing_twist_coefficients', shape=(5,),
                                                value=np.array([0., 0., 0., 0., 0.]))
    wing_twist_b_spline = bsp.BSpline(name='wing_twist_b_spline', space=cubic_b_spline_curve_5_dof_space,
                                                coefficients=wing_twist_coefficients, num_physical_dimensions=1)
    # wing_ffd_block_sect_param.plot()

    section_parametric_coordinates = np.linspace(0., 1., wing_ffd_block_sect_param.num_sections).reshape((-1,1))
    wing_wingspan_stretch = left_wing_wingspan_stretch_b_spline.evaluate(section_parametric_coordinates)
    wing_sectional_chord_stretch = wing_chord_stretch_b_spline.evaluate(section_parametric_coordinates)
    wing_sectional_twist = wing_twist_b_spline.evaluate(section_parametric_coordinates)

    sectional_parameters = {
            'wing_span_stretch': wing_wingspan_stretch,
            'wing_chord_stretch': wing_sectional_chord_stretch,
            'wing_twist': wing_sectional_twist,
                                }

    wing_ffd_block_coefficients = wing_ffd_block_sect_param.evaluate(sectional_parameters, plot=False)
    wing_coefficients = wing_ffd_block.evaluate(wing_ffd_block_coefficients, plot=False)
    b_spline_names_list.append(wing.b_spline_names)
    # endregion

    # region h-tail
    htail_span_stretch_coefficients = outputs_dict['htail_span_stretch_coefficients']
    htail_span_stretch_b_spline = bsp.BSpline(name='htail_span_b_spline', space=linear_b_spline_curve_2_dof_space,
                                                        coefficients=htail_span_stretch_coefficients, num_physical_dimensions=1)

    htail_chord_stretch_coefficients = outputs_dict['htail_chord_stretch_coefficients']
    htail_chord_stretch_b_spline = bsp.BSpline(name='htail_chord_stretch_b_spline', space=linear_b_spline_curve_3_dof_space,
                                                    coefficients=htail_chord_stretch_coefficients, num_physical_dimensions=1)

    htail_ffd_block_translation_x_sect_coeffs = outputs_dict['htail_translation_x_coefficients']
    htail_ffd_block_translation_x_b_spline = bsp.BSpline(name='htail_translation_x_b_spline', space=constant_b_spline_curve_1_dof_space, 
                                                coefficients=htail_ffd_block_translation_x_sect_coeffs, num_physical_dimensions=1)

    section_parametric_coordinates = np.linspace(0., 1., htail_ffd_block_sect_param.num_sections).reshape((-1,1))
    htail_span_stretch = htail_span_stretch_b_spline.evaluate(section_parametric_coordinates)
    htail_sectional_chord_stretch = htail_chord_stretch_b_spline.evaluate(section_parametric_coordinates)
    htail_translation_x = htail_ffd_block_translation_x_b_spline.evaluate(section_parametric_coordinates)


    sectional_parameters = {
        'htail_span_stretch' : htail_span_stretch,
        'htail_chord_stretch' : htail_sectional_chord_stretch,
        'htail_translation_x' : htail_translation_x,
    }

    htail_ffd_block_coefficients = htail_ffd_block_sect_param.evaluate(sectional_parameters)
    htail_coefficients = htail_ffd_block.evaluate(htail_ffd_block_coefficients)

    # endregion

    # region Fuselage
    fuselage_stretch_coefficients = outputs_dict['fuselage_stretch_coefficients']
    fuselage_stretch_b_spline = bsp.BSpline(name='fuselage_stretch_b_spline', space=linear_b_spline_curve_2_dof_space,
                                                        coefficients=fuselage_stretch_coefficients, num_physical_dimensions=1)

    section_parametric_coordinates = np.linspace(0., 1., fuselage_ffd_block_sect_param.num_sections).reshape((-1, 1))
    fuselage_stretch = fuselage_stretch_b_spline.evaluate(section_parametric_coordinates)

    sectional_parameters = {
        'fuselage_stretch' : fuselage_stretch
    }

    fuselage_ffd_block_coefficients = fuselage_ffd_block_sect_param.evaluate(sectional_parameters)
    fuselage_coefficients = fuselage_ffd_block.evaluate(fuselage_ffd_block_coefficients)
    # endregion

    # region Weird nose hub
    nose_hub_transl_x_coefficients = outputs_dict['nose_hub_transl_x_coefficients']
    nose_hub_transl_x_b_spline = bsp.BSpline(name='nose_hub_transl_x_b_spline', space=constant_b_spline_curve_1_dof_space,
                                                        coefficients=nose_hub_transl_x_coefficients, num_physical_dimensions=1)

    section_parametric_coordinates = np.linspace(0., 1., nose_hub_ffd_block_sect_param.num_sections).reshape((-1, 1))
    nose_hub_translation_x = nose_hub_transl_x_b_spline.evaluate(section_parametric_coordinates)

    sectional_parameters = {
        'nose_hub_translation_x' : nose_hub_translation_x
    }

    nose_hub_ffd_block_coefficients = nose_hub_ffd_block_sect_param.evaluate(sectional_parameters)
    nose_hub_coefficients = nose_hub_ffd_block.evaluate(nose_hub_ffd_block_coefficients)
    # endregion

    # region V-tail
    vtail_transl_x_coefficients = outputs_dict['vtail_transl_x_coefficients']
    vtail_transl_x_b_spline = bsp.BSpline(name='vtail_transl_x_b_spline', space=constant_b_spline_curve_1_dof_space,
                                                        coefficients=vtail_transl_x_coefficients, num_physical_dimensions=1)

    section_parametric_coordinates = np.linspace(0., 1., vtail_ffd_block_sect_param.num_sections).reshape((-1, 1))
    vtail_translation_x = vtail_transl_x_b_spline.evaluate(section_parametric_coordinates)

    sectional_parameters = {
        'vtail_translation_x' : vtail_translation_x
    }

    vtail_ffd_block_coefficients = vtail_ffd_block_sect_param.evaluate(sectional_parameters)
    vtail_coefficients = vtail_ffd_block.evaluate(vtail_ffd_block_coefficients)
    # endregion

    # region pusher prop
    pusher_prop_ffd_block_translation_x_sect_coeffs = outputs_dict['pusher_prop_ffd_block_translation_x_coefficients']
    pusher_prop_ffd_block_translation_x_b_spline = bsp.BSpline(name='pusher_prop_ffd_block_translation_x_bspline', space=space, coefficients=pusher_prop_ffd_block_translation_x_sect_coeffs, 
                                                        num_physical_dimensions=1)

    pusher_prop_ffd_block_stretch_y_coefficients = outputs_dict['pusher_prop_ffd_block_stretch_y_coefficients']
    pusher_prop_ffd_block_stretch_y_b_spline = bsp.BSpline(name='pusher_prop_ffd_block_stretch_y_bspline', space=space, coefficients=pusher_prop_ffd_block_stretch_y_coefficients, 
                                                        num_physical_dimensions=1)

    pusher_prop_ffd_block_stretch_z_coefficients = outputs_dict['pusher_prop_ffd_block_stretch_z_coefficients']
    pusher_prop_ffd_block_stretch_z_b_spline = bsp.BSpline(name='pusher_prop_ffd_block_stretch_z_bspline', space=space, coefficients=pusher_prop_ffd_block_stretch_z_coefficients, 
                                                        num_physical_dimensions=1)


    section_parametric_coordinates = np.linspace(0., 1., pusher_prop_ffd_block_sect_param.num_sections).reshape((-1,1))
    pusher_prop_transl_x = pusher_prop_ffd_block_translation_x_b_spline.evaluate(section_parametric_coordinates)
    pusher_prop_stretch_y = pusher_prop_ffd_block_stretch_y_b_spline.evaluate(section_parametric_coordinates)
    pusher_prop_stretch_z = pusher_prop_ffd_block_stretch_z_b_spline.evaluate(section_parametric_coordinates)

    sectional_parameters = {
        'pusher_prop_ffd_block_translation_x': pusher_prop_transl_x,
        'pusher_prop_ffd_block_sectional_stretch_y': pusher_prop_stretch_y,
        'pusher_prop_ffd_block_sectional_stretch_z': pusher_prop_stretch_z,
    }

    pusher_prop_ffd_block_coefficients = pusher_prop_ffd_block_sect_param.evaluate(sectional_parameters, plot=plot)
    pusher_prop_coefficients = pusher_prop_ffd_block.evaluate(pusher_prop_ffd_block_coefficients, plot=plot)
    # endregion

    coefficients_list.append(wing_coefficients)
    coefficients_list.append(fuselage_coefficients)
    coefficients_list.append(htail_coefficients)
    coefficients_list.append(vtail_coefficients)
    coefficients_list.append(nose_hub_coefficients)
    

    for i, comp in enumerate(components_to_be_connected):
        block_name = f'{comp.name}_ffd_block'
        ffd_block_translation_x_sect_coeffs = outputs_dict[f'{block_name}_translation_x_coefficients']
        ffd_block_translation_x_b_spline = bsp.BSpline(name=f'{block_name}_translation_x_bspline', space=constant_b_spline_curve_1_dof_space, 
                                                coefficients=ffd_block_translation_x_sect_coeffs, num_physical_dimensions=1)
        ffd_block_translation_y_sect_coeffs = outputs_dict[f'{block_name}_translation_y_coefficients']
        ffd_block_translation_y_b_spline = bsp.BSpline(name=f'{block_name}_translation_y_bspline', space=constant_b_spline_curve_1_dof_space, 
                                                coefficients=ffd_block_translation_y_sect_coeffs, num_physical_dimensions=1)
        ffd_block_translation_z_sect_coeffs = outputs_dict[f'{block_name}_translation_z_coefficients']
        ffd_block_translation_z_b_spline = bsp.BSpline(name=f'{block_name}_translation_z_bspline', space=constant_b_spline_curve_1_dof_space, 
                                                coefficients=ffd_block_translation_z_sect_coeffs, num_physical_dimensions=1)

        section_parametric_coordinates = np.linspace(0., 1., ffd_block_sect_param_list[i].num_sections).reshape((-1,1))
        comp_transl_x = ffd_block_translation_x_b_spline.evaluate(section_parametric_coordinates)
        comp_transl_y = ffd_block_translation_y_b_spline.evaluate(section_parametric_coordinates)
        comp_transl_z = ffd_block_translation_z_b_spline.evaluate(section_parametric_coordinates)

        sectional_parameters = {
                f'{block_name}_translation_x': comp_transl_x,
                f'{block_name}_translation_y': comp_transl_y,
                f'{block_name}_translation_z': comp_transl_z,
        }

        comp_ffd_block_coefficients = ffd_block_sect_param_list[i].evaluate(sectional_parameters, plot=plot)
        comp_coefficients = ffd_block_list[i].evaluate(comp_ffd_block_coefficients, plot=plot)
        
        
        coefficients_list.append(comp_coefficients)
        # b_spline_names_list.append(comp.b_spline_names)
        
        # geometry.assign_coefficients(coefficients=comp_coefficients, b_spline_names=comp.b_spline_names)


    for i, comp in enumerate(hub_disk_components):
        block_name = f'{comp[0].name}_{comp[1].name}'
        disk = comp[0]
        hub = comp[1]
        blade_1 = comp[2]
        blade_2 = comp[3]

        ffd_block_translation_x_sect_coeffs = outputs_dict[f'{block_name}_translation_x_coefficients']
        ffd_block_translation_x_b_spline = bsp.BSpline(name=f'{block_name}_translation_x_bspline', space=linear_b_spline_curve_2_dof_space, 
                                                coefficients=ffd_block_translation_x_sect_coeffs, num_physical_dimensions=1)
        ffd_block_translation_y_sect_coeffs = outputs_dict[f'{block_name}_translation_y_coefficients']
        ffd_block_translation_y_b_spline = bsp.BSpline(name=f'{block_name}_translation_y_bspline', space=constant_b_spline_curve_1_dof_space, 
                                                coefficients=ffd_block_translation_y_sect_coeffs, num_physical_dimensions=1)
        ffd_block_translation_z_sect_coeffs = outputs_dict[f'{block_name}_translation_z_coefficients']
        ffd_block_translation_z_b_spline = bsp.BSpline(name=f'{block_name}_translation_z_bspline', space=constant_b_spline_curve_1_dof_space, 
                                                coefficients=ffd_block_translation_z_sect_coeffs, num_physical_dimensions=1)

        ffd_block_stretch_y_sect_coeffs = outputs_dict[f'{block_name}_stretch_y_coefficients']

        ffd_block_stretch_y_b_spline = bsp.BSpline(name=f'{block_name}_stretch_y_bspline', space=space, coefficients=ffd_block_stretch_y_sect_coeffs, 
                                                        num_physical_dimensions=1)
        
        section_parametric_coordinates = np.linspace(0., 1., ffd_block_sect_param_list_2[i].num_sections).reshape((-1,1))
        comp_transl_x = ffd_block_translation_x_b_spline.evaluate(section_parametric_coordinates)
        comp_transl_y = ffd_block_translation_y_b_spline.evaluate(section_parametric_coordinates)
        comp_transl_z = ffd_block_translation_z_b_spline.evaluate(section_parametric_coordinates)
        comp_stretch_y = ffd_block_stretch_y_b_spline.evaluate(section_parametric_coordinates)

        sectional_parameters = {
                f'{block_name}_translation_x': comp_transl_x,
                f'{block_name}_translation_y': comp_transl_y,
                f'{block_name}_translation_z': comp_transl_z,
                f'{block_name}_stretch_y': comp_stretch_y,
        }


        comp_ffd_block_coefficients = ffd_block_sect_param_list_2[i].evaluate(sectional_parameters, plot=plot)
        comp_coefficients = ffd_block_list_2[i].evaluate(comp_ffd_block_coefficients, plot=plot)
        # geometry.assign_coefficients(coefficients=comp_coefficients, b_spline_names=disk.b_spline_names + hub.b_spline_names)

        coefficients_list.append(comp_coefficients[disk.name + '_coefficients'])
        coefficients_list.append(comp_coefficients[hub.name + '_coefficients'])
        coefficients_list.append(comp_coefficients[blade_1.name + '_coefficients'])
        coefficients_list.append(comp_coefficients[blade_2.name + '_coefficients'])

        # geometry.assign_coefficients(coefficients=comp_coefficients[disk.name + '_coefficients'], b_spline_names=disk.b_spline_names)
        # geometry.assign_coefficients(coefficients=comp_coefficients[hub.name + '_coefficients'], b_spline_names=hub.b_spline_names)
        # geometry.assign_coefficients(coefficients=comp_coefficients[blade_1.name + '_coefficients'], b_spline_names=blade_1.b_spline_names)
        # geometry.assign_coefficients(coefficients=comp_coefficients[blade_2.name + '_coefficients'], b_spline_names=blade_2.b_spline_names)

    coefficients_list.append(pusher_prop_coefficients[pp_disk.name + '_coefficients'])
    coefficients_list.append(pusher_prop_coefficients[pp_hub.name + '_coefficients'])
    coefficients_list.append(pusher_prop_coefficients[pp_blade_1.name + '_coefficients'])
    coefficients_list.append(pusher_prop_coefficients[pp_blade_2.name + '_coefficients'])
    coefficients_list.append(pusher_prop_coefficients[pp_blade_3.name + '_coefficients'])
    coefficients_list.append(pusher_prop_coefficients[pp_blade_4.name + '_coefficients'])


    geometry.assign_coefficients(coefficients=coefficients_list, b_spline_names=b_spline_names_list)
    # geometry.assign_coefficients(coefficients=coefficients_list, b_spline_names=[wing.b_spline_names, ])


    # del outputs_dict
    # gc.collect()

    # enforce_kinematic_geometric_constraints()

    # endregion


# Update all the meshes
wing_meshes = wing_meshes.update(geometry=geometry)
tail_meshes = tail_meshes.update(geometry=geometry)
vtail_meshes =  vtail_meshes.update(geometry=geometry)
box_beam_mesh = box_beam_mesh.update(geometry=geometry)

fueslage_top_points_m3l = geometry.evaluate(fueslage_top_points_parametric)
fueslage_bottom_points_m3l = geometry.evaluate(fueslage_bottom_points_parametric)
fuesleage_mesh = m3l.linspace(fueslage_top_points_m3l.reshape((-1, 3)), fueslage_bottom_points_m3l.reshape((-1, 3)),  int(num_fuselage_height + 1))
fuesleage_mesh.description = 'zero_y'

if plot_meshes:
    geometry.plot_meshes(meshes=[wing_meshes.vlm_mesh, tail_meshes.vlm_mesh, vtail_meshes.vlm_mesh, box_beam_mesh.beam_nodes, fuesleage_mesh])


pp_mesh = pp_mesh.update(geometry=geometry)
rlo_mesh = rlo_mesh.update(geometry=geometry)
rro_mesh = rro_mesh.update(geometry=geometry)
flo_mesh = flo_mesh.update(geometry=geometry)
fro_mesh = fro_mesh.update(geometry=geometry)
rli_mesh = rli_mesh.update(geometry=geometry)
rri_mesh = rri_mesh.update(geometry=geometry)
fli_mesh = fli_mesh.update(geometry=geometry)
fri_mesh = fri_mesh.update(geometry=geometry)

# chord/ twist profiles
chord_cps_numpy = np.array([0.122222, 0.213889, 0.188426, 0.050926]) * 5
chord_cps_numpy_pusher = np.array([0.122222, 0.213889, 0.188426, 0.050926]) * 4.5

twist_cps_numpy = np.deg2rad(np.linspace(35.000000, 15.000000, 4))
twist_cps_numpy_pusher = np.deg2rad(np.linspace(55.000000, 10.000000, 4))

lower_twist_hover = np.deg2rad(1)
upper_twist_hover = np.deg2rad(60)
lower_twist_pusher = np.deg2rad(5)
upper_twist_pusher = np.deg2rad(85)

flo_blade_chord_bsp_cps = fro_blade_chord_bsp_cps = system_model.create_input('front_outer_blade_chord_cps', val=chord_cps_numpy, dv_flag=True, lower=0.1, upper=1.1)
fli_blade_chord_bsp_cps = fri_blade_chord_bsp_cps = system_model.create_input('front_inner_blade_chord_cps', val=chord_cps_numpy, dv_flag=True, lower=0.1, upper=1.1)
rlo_blade_chord_bsp_cps = rro_blade_chord_bsp_cps = system_model.create_input('rear_outer_blade_chord_cps', val=chord_cps_numpy, dv_flag=True, lower=0.1, upper=1.1)
rli_blade_chord_bsp_cps = rri_blade_chord_bsp_cps = system_model.create_input('rear_inner_blade_chord_cps', val=chord_cps_numpy, dv_flag=True, lower=0.1, upper=1.1)

flo_blade_twist_bsp_cps = fro_blade_twist_bsp_cps = system_model.create_input('front_outer_blade_twist_cps', val=twist_cps_numpy, dv_flag=True, lower=lower_twist_hover, upper=upper_twist_hover)
fli_blade_twist_bsp_cps = fri_blade_twist_bsp_cps = system_model.create_input('front_inner_blade_twist_cps', val=twist_cps_numpy, dv_flag=True, lower=lower_twist_hover, upper=upper_twist_hover)
rlo_blade_twist_bsp_cps = rro_blade_twist_bsp_cps = system_model.create_input('rear_outer_blade_twist_cps', val=twist_cps_numpy, dv_flag=True, lower=lower_twist_hover, upper=upper_twist_hover)
rli_blade_twist_bsp_cps = rri_blade_twist_bsp_cps = system_model.create_input('rear_inner_blade_twist_cps', val=twist_cps_numpy, dv_flag=True, lower=lower_twist_hover, upper=upper_twist_hover)

pusher_chord_bsp_cps = system_model.create_input('pusher_chord_bsp_cps', val=chord_cps_numpy_pusher, dv_flag=True, lower=0.08, upper=1.4)
pusher_twist_bsp_cps = system_model.create_input('pusher_twist_bsp_cps', val=twist_cps_numpy_pusher, dv_flag=True, lower=lower_twist_pusher, upper=upper_twist_pusher)

rlo_mesh.chord_cps = rlo_blade_chord_bsp_cps
rlo_mesh.twist_cps = rlo_blade_twist_bsp_cps

rli_mesh.chord_cps = rli_blade_chord_bsp_cps
rli_mesh.twist_cps = rli_blade_twist_bsp_cps

rri_mesh.chord_cps = rri_blade_chord_bsp_cps
rri_mesh.twist_cps = rri_blade_twist_bsp_cps

rro_mesh.chord_cps = rro_blade_chord_bsp_cps
rro_mesh.twist_cps = rro_blade_twist_bsp_cps

flo_mesh.chord_cps = flo_blade_chord_bsp_cps
flo_mesh.twist_cps = flo_blade_twist_bsp_cps

fli_mesh.chord_cps = fli_blade_chord_bsp_cps
fli_mesh.twist_cps = fli_blade_twist_bsp_cps

fri_mesh.chord_cps = fri_blade_chord_bsp_cps
fri_mesh.twist_cps = fri_blade_twist_bsp_cps

fro_mesh.chord_cps = fro_blade_chord_bsp_cps
fro_mesh.twist_cps = fro_blade_twist_bsp_cps

pp_mesh.chord_cps = pusher_chord_bsp_cps
pp_mesh.twist_cps = pusher_twist_bsp_cps

print('New thrust origins')
print(pp_mesh.thrust_origin)
print(rlo_mesh.thrust_origin)
print(rro_mesh.thrust_origin)
print(flo_mesh.thrust_origin)
print(fro_mesh.thrust_origin)
print(rli_mesh.thrust_origin)
print(rri_mesh.thrust_origin)
print(fli_mesh.thrust_origin)
print(fri_mesh.thrust_origin)

print('New radii')
print(pp_mesh.radius)
print(rlo_mesh.radius)
print(rro_mesh.radius)
print(flo_mesh.radius)
print(fro_mesh.radius)
print(rli_mesh.radius)
print(rri_mesh.radius)
print(fli_mesh.radius)
print(fri_mesh.radius)


# region Drag components 

# Component surface areas
component_list = [rlo_boom, rlo_hub, fuselage, wing, h_tail, vtail, pp_hub, rlo_blade_1]
surface_area_list = compute_component_surface_area(
    component_list=component_list,
    geometry=geometry,
    parametric_mesh_grid_num=20,
    plot=False,
)


# Fuselage
fuselage_l1 = geometry.evaluate(fuselage_l1_parametric).reshape((-1, 3))
fuselage_l2 = geometry.evaluate(fuselage_l2_parametric).reshape((-1, 3))
fuselage_length = m3l.norm(fuselage_l2-fuselage_l1)

fuselage_d1 = geometry.evaluate(fuselage_d1_parametric).reshape((-1, 3))
fuselage_d2=  geometry.evaluate(fuselage_d2_parametric).reshape((-1, 3))
fuselage_diameter = m3l.norm(fuselage_d2-fuselage_d1)

fuselage_drag_comp = DragComponent(
    component_type='fuselage',
    wetted_area=surface_area_list[2],
    characteristic_length=fuselage_length,
    characteristic_diameter=fuselage_diameter,
    Q=2.9,
)


# wing
wing_mid_le = geometry.evaluate(wing_mid_le_parametric).reshape((-1, 3))
wing_mid_te = geometry.evaluate(wing_mid_te_parametric).reshape((-1, 3))
wing_mid_chord_length = m3l.norm(wing_mid_le-wing_mid_te)

wing_le_right_mapped_array = geometry.evaluate(wing_le_right_parametric).reshape((-1, 3))
wing_le_left_mapped_array = geometry.evaluate(wing_le_left__parametric).reshape((-1, 3))
wing_span = m3l.norm(wing_le_left_mapped_array - wing_le_right_mapped_array)

wing_drag_comp = DragComponent(
    component_type='wing',
    wetted_area=surface_area_list[3],
    characteristic_length=wing_mid_chord_length,
    thickness_to_chord=0.16,
    x_cm=0.45,
    Q=1.4,
)

# h tail
h_tail_mid_le = geometry.evaluate(h_tail_mid_le_parametric).reshape((-1, 3))
h_tail_mid_te = geometry.evaluate(h_tail_mid_te_parametric).reshape((-1, 3))
h_tail_chord_length = m3l.norm(h_tail_mid_le-h_tail_mid_te)

h_tail_drag_comp = DragComponent(
    component_type='wing',
    wetted_area=surface_area_list[4],
    characteristic_length=h_tail_chord_length,
    thickness_to_chord=0.15,
    x_cm=0.3,
    Q=1.5,

)

# v tail
vtail_mid_le = geometry.evaluate(vtail_mid_le_parametric).reshape((-1, 3))
vtail_mid_te = geometry.evaluate(vtail_mid_te_parametric).reshape((-1, 3))
vtail_chord_length = m3l.norm(vtail_mid_le-vtail_mid_te)

vtail_drag_comp = DragComponent(
    component_type='wing',
    wetted_area=surface_area_list[5],
    characteristic_length=vtail_chord_length,
    thickness_to_chord=0.115,
    x_cm=0.2,
    Q=1.5,
)

# boom
boom_l1 = geometry.evaluate(boom_l1_parametric).reshape((-1, 3))
boom_l2 = geometry.evaluate(boom_l2_parametric).reshape((-1, 3))
boom_length = m3l.norm(boom_l1-boom_l2)

boom_d1 = geometry.evaluate(boom_d1_parametric).reshape((-1, 3))
boom_d2 = geometry.evaluate(boom_d2_parametric).reshape((-1, 3))
boom_diameter = m3l.norm(boom_d1-boom_d2)

boom_drag_comp = DragComponent(
    component_type='boom',
    wetted_area=surface_area_list[0],
    characteristic_diameter=boom_diameter,
    characteristic_length=boom_length,
    multiplicity=8,
    Q=2.,
)

# lift hubs
hub_l1 = geometry.evaluate(hub_l1_parametric)
hub_l2 = geometry.evaluate(hub_l2_parametric)
hub_length = m3l.norm(hub_l1-hub_l2)

hub_drag_comp = DragComponent(
    component_type='nacelle',
    wetted_area=surface_area_list[1],
    characteristic_diameter=hub_length,
    characteristic_length=hub_length,
    multiplicity=8,
    Q=1,

)


# blade 
blade_tip = geometry.evaluate(blade_tip_parametric)
blade_hub = geometry.evaluate(blade_hub_parametric)

blade_length = m3l.norm(blade_tip-blade_hub)
blade_drag_comp = DragComponent(
    component_type='flat_plate',
    characteristic_length=blade_length,
    wetted_area=surface_area_list[-1],
    multiplicity=16,
    Q=2,

)

drag_comp_list = [wing_drag_comp, fuselage_drag_comp, h_tail_drag_comp, vtail_drag_comp,
                  blade_drag_comp, boom_drag_comp, hub_drag_comp]

S_ref = surface_area_list[3] / 2.1
print('wing area', S_ref.value)
h_tail_area = surface_area_list[4] / 2.1
print('tail area', h_tail_area.value)
v_tail_area = surface_area_list[5] / 2.
print('v_tail area', v_tail_area.value)
# 
wing_AR = wing_span**2 / S_ref
print('wing span', wing_span.value)
print('wing AR', wing_AR.value)
# endregion

# if geometry_dv:
if False:
    radii_front_wing_ratio = (front_outer_radius * 1 + front_inner_radius * 1) / (0.5 * wing_span_input)
    radii_rear_wing_ratio = (rear_outer_radius * 1 + rear_inner_radius * 1) / (0.5 * wing_span_input)
    system_model.register_output(radii_front_wing_ratio * 1)
    system_model.register_output(radii_rear_wing_ratio)
    system_model.add_constraint(radii_front_wing_ratio, equals=0.4)
    system_model.add_constraint(radii_rear_wing_ratio, equals=0.4)


# exit()

# 


# cruise_geometry = geometry.copy()