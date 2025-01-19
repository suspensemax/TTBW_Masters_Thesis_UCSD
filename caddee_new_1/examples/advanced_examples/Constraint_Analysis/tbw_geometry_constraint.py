"TBW baseline fixed wing and span location geometry working? backup"

# import numpy as np
import numpy as np
import caddee.api as cd 
import lsdo_geo as lg
import m3l
from caddee import GEOMETRY_FILES_FOLDER
from caddee.utils.helper_functions.geometry_helpers import make_rotor_mesh, make_vlm_camber_mesh, make_1d_box_beam_mesh, compute_component_surface_area, BladeParameters
from caddee.utils.aircraft_models.drag_models.drag_build_up import DragComponent
import lsdo_geo.splines.b_splines as bsp
import gc
gc.enable()
from dataclasses import dataclass, field
import csdl
import math

# @dataclass
# class geometryOutputs: 
#     wing_AR : m3l.Variable
#     # S_ref : m3l.Variable 
#     wing_area : m3l.Variable
#     # wing_span : m3l.Variable

# region True or False
geometry_plot = False
components_plot = False
do_plots = False
print_coordinates = False
plot_htail_mesh = False
plot_left_strut_mesh = False
plot_right_strut_mesh = False

# region Plots 
# VLM 
mesh_flag_wing = False
mesh_flag_wing_oml = False
mesh_flag_htail = False
mesh_flag_htail_oml = False
mesh_flag_left_strut = False
mesh_flag_left_strut_oml = False
mesh_flag_right_strut = False
mesh_flag_right_strut_oml = False
mesh_flag_vtail_helper = False
mesh_flag_fuselage = False
mesh_flag_left_jury_helper = False
mesh_flag_right_jury_helper = False
mesh_flag_gearpod = False
plot_unupdated_vlm_mesh = False
plot_updated_vlm_mesh = False
# beam
wing_beam_mesh_flag = False
left_strut_beam_mesh_flag = False
right_strut_beam_mesh_flag = False
left_jury_beam_mesh_flag = False
right_jury_beam_mesh_flag = False
# endregion

# endregion

# Instantiate system model
system_model = m3l.Model()

# Importing and refitting the geometrywing_oml_mesh
geometry = lg.import_geometry(GEOMETRY_FILES_FOLDER / 'tbw_try.stp', parallelize=True)
geometry.refit(parallelize=True, order=(4, 4))

if geometry_plot:
    geometry.plot()

plot_meshes = False
# geometry_dv = True

# region Declaring Components
wing = geometry.declare_component(component_name='wing', b_spline_search_names=['Wing'])
if components_plot:
    wing.plot()
htail = geometry.declare_component(component_name='htail', b_spline_search_names=['Htail'])
if components_plot:
    htail.plot()
strut = geometry.declare_component(component_name='strut', b_spline_search_names=['Strut'])
if components_plot:
    strut.plot()
strut_right = geometry.declare_component(component_name='strut_right', b_spline_search_names=['Strut, 0'])
if components_plot:
    strut_right.plot()
strut_left = geometry.declare_component(component_name='strut_left', b_spline_search_names=['Strut, 1'])
if components_plot:
    strut_left.plot()
jury = geometry.declare_component(component_name='jury', b_spline_search_names=['Jury'])
if components_plot:
    jury.plot()
jury_right = geometry.declare_component(component_name='jury_right', b_spline_search_names=['Jury, 0'])
if components_plot:
    jury_right.plot()
jury_left = geometry.declare_component(component_name='jury_left', b_spline_search_names=['Jury, 1'])
if components_plot:
    jury_left.plot()
vtail = geometry.declare_component(component_name='vtail', b_spline_search_names=['Vtail'])
if components_plot:
    vtail.plot()
Fuselage = geometry.declare_component(component_name='Fuselage', b_spline_search_names=['Fuselage'])
if components_plot:
    Fuselage.plot()
gearpod = geometry.declare_component(component_name='gearpod', b_spline_search_names=['Gear Pod'])
if components_plot:
    gearpod.plot()
# endregion

# region geometry dv

geometry_dv = False
# geometry_dv = True

wing_span_dv = system_model.create_input(name='wing_span_dv', shape = (1,), val = 170, dv_flag = geometry_dv, lower = 140., upper = 200., scaler= 1.e-1)
# wing_span_dv = system_model.create_input(name='wing_span_dv', shape = (1,), val = 171, dv_flag = geometry_dv, lower = 140., upper = 200., scaler =1.e-2)

# geometry_dv = False
# geometry_dv = True
# wing_tip_chord_left_dv = system_model.create_input(name='wing_tip_chord_left_dv', shape= (1,), val = 10.7, dv_flag = geometry_dv, lower=2.37, upper=5.13)
wing_tip_chord_left_dv = system_model.create_input(name='wing_tip_chord_left_dv', shape= (1,), val = 3.8, dv_flag = geometry_dv, lower=3.37, upper=4.13)
# wing_tip_chord_right_dv = system_model.create_input(name='wing_tip_chord_right_dv', shape= (1,), val = 3.8, dv_flag = geometry_dv, lower=3.37, upper=4.13, scaler =1.e-1)
wing_root_chord_dv = system_model.create_input(name='wing_root_chord_dv', shape= (1,), val = 10.7, dv_flag = geometry_dv, lower=9.64, upper=11.79)
# wing_root_chord_dv = system_model.create_input(name='wing_root_chord_dv', shape= (1,), val = 9.64, dv_flag = geometry_dv, lower=9.64, upper=12.79)
# wing_mid_chord_left_dv = system_model.create_input(name = 'wing_mid_chord_left_dv', shape=(1,), val= 11.68, dv_flag = geometry_dv, lower = 7.70, upper = 11.68)
wing_mid_chord_left_dv = system_model.create_input(name = 'wing_mid_chord_left_dv', shape=(1,), val= 9.5, dv_flag = geometry_dv, lower = 8.70, upper = 10.68)
# wing_mid_chord_left_dv = system_model.create_input(name = 'wing_mid_chord_left_dv', shape=(1,), val= 8.7, dv_flag = geometry_dv, lower = 8.70, upper = 10.68)
# wing_mid_chord_right_dv = system_model.create_input(name = 'wing_mid_chord_right_dv', shape=(1,), val= 9.5, dv_flag = geometry_dv, lower = 8.70, upper = 10.68, scaler =1.e-1)

# left_strut_chord_dv = system_model.create_input(name='left_strut_chord_dv', shape = (1,), val = 1.8, dv_flag = geometry_dv, lower = 1.7, upper = 1.9)
# right_strut_chord_dv = system_model.create_input(name='right_strut_chord_dv', shape = (1,), val = 1.8, dv_flag = geometry_dv, lower = 1.7, upper = 1.9)
# left_jury_chord_dv = system_model.create_input(name='left_strut_chord_dv', shape = (1,), val = 1.8, dv_flag = geometry_dv, lower = 1.7, upper = 1.9)
# right_jury_chord_dv = system_model.create_input(name='right_strut_chord_dv', shape = (1,), val = 1.8, dv_flag = geometry_dv, lower = 1.7, upper = 1.9)

# endregion

# region mesh

# region Wing Mesh

num_spanwise_wing_vlm = 25
num_chordwise_wing_vlm = 7 

point00_wing = np.array([68.035, 85.291, 4.704]) # * ft2m # Right tip leading edge 
wing_le_right = point00_wing
point01_wing = np.array([71.790, 85.291, 4.704]) # * ft2m # Right tip trailing edge
wing_te_right = point01_wing
point10_wing = np.array([57.575, 49.280, 5.647]) # * ft2m # Right leading edge 1
point11_wing = np.array([67.088, 49.280, 5.507]) # * ft2m # Right trailing edge 1
point20_wing = np.array([56.172, 42.646, 5.821]) # * ft2m # Right leading edge 2
point21_wing = np.array([65.891, 42.646, 5.651]) # * ft2m # Right trailing edge 2
point30_wing = np.array([47.231, 0.000, 6.937]) # * ft2m # Center Leading Edge
wing_le_center = point30_wing
point31_wing = np.array([57.953, 0.000, 6.566]) # * ft2m # Center Trailing edge
wing_te_center = point31_wing
point40_wing = np.array([56.172, -42.646, 5.821]) # * ft2m # Left leading edge 2
point41_wing = np.array([65.891, -42.646, 5.651]) # * ft2m # Left trailing edge 2
point50_wing = np.array([57.575, -49.280, 5.647]) # * ft2m # Left leading edge 1
point51_wing = np.array([67.088, -49.280, 5.507]) # * ft2m # Left trailing edge 1
point60_wing = np.array([68.035, -85.291, 4.704]) # * ft2m # Left tip leading edge 
wing_le_left = point60_wing
point61_wing = np.array([71.790, -85.291, 4.704]) # * ft2m # Left tip trailing edge
wing_te_left = point61_wing

wing_meshes = make_vlm_camber_mesh(
    geometry=geometry,
    wing_component=wing,
    num_spanwise=num_spanwise_wing_vlm,
    num_chordwise=num_chordwise_wing_vlm,
    te_right=np.array([73., 85., 5.507]),
    te_left=np.array([73., -85., 5.507]),
    te_center=np.array([57.953, 0.000, 6.566]),
    le_left=np.array([66., -85., 5.647]),
    le_right=np.array([66., 85., 5.647]),
    le_center=np.array([47.231, 0.000, 6.937]),
    grid_search_density_parameter=100,
    le_interp='linear',
    te_interp='linear',
    off_set_x=0.2,
    bunching_cos=True,
    plot=mesh_flag_wing,
    mirror=True,
)

# endregion

# region htail mesh

num_spanwise_htail_vlm = 21
num_chordwise_htail_vlm = 5

num_spanwise_vlm_htail = 9
num_chordwise_vlm_htail = 4

point00_htail = np.array([132.002-10.0, 19.217+4.5, 18.993+3.5]) # * ft2m # Right tip leading edge
point01_htail = np.array([135.993, 19.217, 18.993]) # * ft2m # Right tip trailing edge
point10_htail = np.array([122.905, 0.000, 20.000]) # * ft2m # Center Leading Edge
point11_htail = np.array([134.308, 0.000, 20.000]) # * ft2m # Center Trailing edge
point20_htail = np.array([132.002-10, -19.217-4.5, 18.993+3.5]) # * ft2m # Left tip leading edge
point21_htail = np.array([135.993, -19.217, 18.993]) # * ft2m # Left tip trailing edge

htail_meshes = make_vlm_camber_mesh(
    geometry=geometry,
    wing_component=htail,
    # num_spanwise=num_spanwise_htail_vlm,
    num_spanwise=num_spanwise_vlm_htail,
    # num_chordwise=num_chordwise_htail_vlm,
    num_chordwise=num_chordwise_vlm_htail,
    te_right=point01_htail,
    te_left=point21_htail,
    te_center=point11_htail,
    le_left=point20_htail,
    le_right=point00_htail,
    le_center=point10_htail,
    grid_search_density_parameter=100,
    plot=mesh_flag_htail,
    mirror=True,
)

# endregion

# region strut mesh

# region left strut mesh 

num_spanwise_strut_vlm = 21
num_chordwise_strut_vlm = 5 

vertex00_left_strut = np.array([55.573, -12.641, -4.200]) # left leading 1
vertex01_left_strut = np.array([55.034, -16.277, -3.204]) # left leading 2
vertex02_left_strut = np.array([56.422, -25.365, -0.713]) # left leading 3
vertex03_left_strut = np.array([58.313, -30.818, 0.782]) # left leading 4
vertex04_left_strut = np.array([58.089, -36.271, 2.276]) # left leading 5
vertex05_left_strut = np.array([59.477, -45.359, 4.767]) # left leading 6
vertex06_left_strut = np.array([61.090, -48.994, 5.763]) # left leading 7

vertex10_left_strut = np.array([57.309, -12.641, -4.200]) # left trailing 1
vertex11_left_strut = np.array([58.959, -16.277, -3.204]) # left trailing 2
vertex12_left_strut = np.array([60.348, -25.365, -0.713]) # left trailing 3
vertex13_left_strut = np.array([60.124, -30.818, 0.782]) # left trailing 4
vertex14_left_strut = np.array([62.014, -36.271, 2.276]) # left trailing 5
vertex15_left_strut = np.array([63.403, -45.359, 4.767]) # left trailing 6                
vertex16_left_strut = np.array([62.902, -48.994, 5.763]) # left trailing 7

left_strut_meshes = make_vlm_camber_mesh(
    geometry=geometry,
    wing_component=strut_left,
    num_spanwise=num_spanwise_strut_vlm,
    num_chordwise=num_chordwise_strut_vlm,
    te_right=vertex10_left_strut,
    # te_right=np.array([57.,-13.,-4.2]),
    te_left=vertex16_left_strut,
    # te_left=np.array([63., -49., 5.763]),
    # te_center=vertex13_left_strut,
    # te_center=np.array([60., -30.9, 0.782]),
    le_left=vertex06_left_strut,
    # le_left=np.array([61., -49., 5.763]),
    le_right=vertex00_left_strut,
    # le_right=np.array([55., -13., -4.200]),
    # le_center=vertex03_left_strut,
    # le_center=np.array([58., -30.8, 0.782]),
    grid_search_density_parameter=100,
    plot=mesh_flag_left_strut,
    mirror=False,
)

# endregion
  
# region right strut mesh

num_spanwise_strut_vlm = 21
num_chordwise_strut_vlm = 5 
vertex00_right_strut = np.array([55.573, 12.641, -4.200]) # right leading 1
vertex01_right_strut = np.array([55.034, 16.277, -3.204]) # right leading 2
vertex02_right_strut = np.array([56.422, 25.365, -0.713]) # right leading 3
vertex03_right_strut = np.array([58.313, 30.818, 0.782]) # right leading 4
vertex04_right_strut = np.array([58.089, 36.271, 2.276]) # right leading 5
vertex05_right_strut = np.array([59.477, 45.359, 4.767]) # right leading 6
vertex06_right_strut = np.array([61.090, 48.994, 5.763]) # right leading 7

vertex10_right_strut = np.array([57.309, 12.641, -4.200]) # right trailing 1
vertex11_right_strut = np.array([58.959, 16.277, -3.204]) # right trailing 2
vertex12_right_strut = np.array([60.348, 25.365, -0.713]) # right trailing 3
vertex13_right_strut = np.array([60.124, 30.818, 0.782]) # right trailing 4
vertex14_right_strut = np.array([62.014, 36.271, 2.276]) # right trailing 5
vertex15_right_strut = np.array([63.403, 45.359, 4.767]) # right trailing 6                
vertex16_right_strut = np.array([62.902, 48.994, 5.763]) # right trailing 7

right_strut_meshes = make_vlm_camber_mesh(
    geometry=geometry,
    wing_component=strut_right,
    num_spanwise=num_spanwise_strut_vlm,
    num_chordwise=num_chordwise_strut_vlm,
    te_right=vertex10_right_strut,
    # te_right=np.array([57.,-13.,-4.2]),
    te_left=vertex16_right_strut,
    # te_left=np.array([63., -49., 5.763]),
    # te_center=vertex13_right_strut,
    # te_center=np.array([60., -30.9, 0.782]),
    le_left=vertex06_right_strut,
    # le_left=np.array([61., -49., 5.763]),
    le_right=vertex00_right_strut,
    # le_right=np.array([55., -13., -4.200]),
    # le_center=vertex03_right_strut,
    # le_center=np.array([58., -30.8, 0.782]),
    grid_search_density_parameter=100,
    plot=mesh_flag_right_strut,
    mirror=False,
)

# endregion

# endregion

# region vtail mesh 
num_spanwise_vlm_vtail = 8
num_chordwise_vlm_vtail = 6
vtail_meshes = make_vlm_camber_mesh(
    geometry=geometry,
    wing_component=vtail,
    num_spanwise=num_spanwise_vlm_vtail,
    num_chordwise=num_chordwise_vlm_vtail,
    le_left=np.array([105.343, 0., 3.5,]), 
    le_right=np.array([120.341, 0., 20.754,]),
    te_left=np.array([122.596, 0., 3.5,]),
    te_right=np.array([137.595, 0., 20.754,]),
    plot=mesh_flag_vtail_helper,
    orientation='vertical',
    zero_y=True,
)
# endregion

# region jury mesh 

# region left jury mesh     
num_spanwise_vlm_jury = 5
num_chordwise_vlm_jury = 3
left_jury_meshes = make_vlm_camber_mesh(
    geometry=geometry,
    wing_component=jury_left,
    num_spanwise=num_spanwise_vlm_jury,
    num_chordwise=num_chordwise_vlm_jury,
    le_left=np.array([58.721, -30.818, 0.732,]), 
    le_right=np.array([58.052, -30.818, 6.181]),
    te_left=np.array([59.676, -30.818, 0.732]),
    te_right=np.array([59.237, -30.818, 6.181]),
    plot=mesh_flag_left_jury_helper,
    orientation='vertical',
    zero_y=True,
)
# endregion

# region right jury mesh     
num_spanwise_vlm_jury = 5
num_chordwise_vlm_jury = 3
right_jury_meshes = make_vlm_camber_mesh(
    geometry=geometry,
    wing_component=jury_right,
    num_spanwise=num_spanwise_vlm_jury,
    num_chordwise=num_chordwise_vlm_jury,
    le_left=np.array([58.721, 30.818, 0.732,]), 
    le_right=np.array([58.052, 30.818, 6.181]),
    te_left=np.array([59.676, 30.818, 0.732]),
    te_right=np.array([59.237, 30.818, 6.181]),
    plot=mesh_flag_right_jury_helper,
    orientation='vertical',
    zero_y=True,
)
# endregion

# endregion

# region not working Gearpod mesh 

# num_spanwise_gearpod_vlm_1 = 3
# num_spanwise_gearpod_vlm_2 = 9
# num_spanwise_gearpod_vlm = 2*((num_spanwise_gearpod_vlm_1-1) + (num_spanwise_gearpod_vlm_2-1)) + 1
# num_chordwise_gearpod_vlm = 5

# gearpod_strut_le_right = np.array([55.573, 12.461, -4.20])
# gearpod_strut_te_right = np.array([57.309, 12.641, -4.20])
# gearpod_strut_le_left = np.array([55.573, -12.641, -4.20])
# gearpod_strut_te_left = np.array([57.309, -12.641, -4.20])
# gearpod_le_right = np.array([54.807, 10.258, -4.796])
# gearpod_te_right = np.array([60.148, 10.258, -4.621])
# gearpod_le_left = np.array([54.807, -10.258, -4.796])
# gearpod_te_left = np.array([60.148, -10.258, -4.621])
# # gearpod_le_center = np.array([36.497, 0., -6.284])
# gearpod_le_center = np.array([35.80, 0., -5.25])
# gearpod_te_center = np.array([72.238, 0., -5.250])

# gearpod_leading_edge_points_le_1 = np.linspace(gearpod_strut_le_right, gearpod_le_right, num_spanwise_gearpod_vlm_1)[1:-1]
# gearpod_leading_edge_points_le_2 = np.linspace(gearpod_le_right, gearpod_le_center, num_spanwise_gearpod_vlm_2)[1:-1]
# gearpod_leading_edge_points_le_3 = np.linspace(gearpod_le_center, gearpod_le_left, num_spanwise_gearpod_vlm_2)[1:-1]
# gearpod_leading_edge_points_le_4 = np.linspace(gearpod_le_left, gearpod_strut_le_left, num_spanwise_gearpod_vlm_1)[1:-1]

# gearpod_leading_edge_points = np.vstack([gearpod_strut_le_right, gearpod_leading_edge_points_le_1, gearpod_le_right, gearpod_leading_edge_points_le_2, 
#                                         gearpod_le_center, gearpod_leading_edge_points_le_3, gearpod_le_left, gearpod_leading_edge_points_le_4, 
#                                         gearpod_strut_le_left])

# gearpod_trailing_edge_points_te_1 = np.linspace(gearpod_strut_te_right, gearpod_te_right, num_spanwise_gearpod_vlm_1)[1:-1]
# gearpod_trailing_edge_points_te_2 = np.linspace(gearpod_te_right, gearpod_te_center, num_spanwise_gearpod_vlm_2)[1:-1]
# gearpod_trailing_edge_points_te_3 = np.linspace(gearpod_te_center, gearpod_te_left, num_spanwise_gearpod_vlm_2)[1:-1]
# gearpod_trailing_edge_points_te_4 = np.linspace(gearpod_te_left, gearpod_strut_te_left, num_spanwise_gearpod_vlm_1)[1:-1]

# gearpod_trailing_edge_points = np.vstack([gearpod_strut_te_right, gearpod_trailing_edge_points_te_1, gearpod_te_right, gearpod_trailing_edge_points_te_2, 
#                                         gearpod_te_center, gearpod_trailing_edge_points_te_3, gearpod_te_left, gearpod_trailing_edge_points_te_4, 
#                                         gearpod_strut_te_left])

# gearpod_leading_edge = gearpod.project(gearpod_leading_edge_points, plot=do_plots)
# gearpod_trailing_edge = gearpod.project(gearpod_trailing_edge_points, plot=do_plots)

# gearpod_le_coord = geometry.evaluate(gearpod_leading_edge).reshape((-1, 3))
# gearpod_te_coord = geometry.evaluate(gearpod_trailing_edge).reshape((-1, 3))
# # A user can print the values of m3l variable by calling '.value'
# if print_coordinates:
#     print(gearpod_le_coord.value)
#     print(gearpod_te_coord.value)

# # Getting a linearly spaced (2-D) array between the leading and trailing edge
# gearpod_chord = m3l.linspace(gearpod_le_coord, gearpod_te_coord, num_chordwise_gearpod_vlm)

# # Projecting the 2-D array onto the upper and lower surface of the gearpod to get the camber surface mesh
# gearpod_upper_surface_wireframe_parametric = gearpod.project(gearpod_chord.value + np.array([0., 0., 1.]), direction=np.array([0., 0., 1.]), grid_search_density_parameter=25, plot=False)
# gearpod_lower_surface_wireframe_parametric = gearpod.project(gearpod_chord.value - np.array([0., 0., 1.]), direction=np.array([0., 0., -1.]), grid_search_density_parameter=25, plot=False)

# gearpod_upper_surface_wireframe = geometry.evaluate(gearpod_upper_surface_wireframe_parametric).reshape((num_chordwise_gearpod_vlm, num_spanwise_gearpod_vlm, 3))
# gearpod_lower_surface_wireframe = geometry.evaluate(gearpod_lower_surface_wireframe_parametric).reshape((num_chordwise_gearpod_vlm, num_spanwise_gearpod_vlm, 3))

# gearpod_camber_surface = m3l.linspace(gearpod_upper_surface_wireframe, gearpod_lower_surface_wireframe, 1)#.reshape((-1, 3))

# # Optionally, the resulting camber surface mesh can be plotted
# if mesh_flag_gearpod:
#     geometry.plot_meshes(meshes=gearpod_camber_surface, mesh_plot_types=['wireframe'], mesh_opacity=1., mesh_color='#F5F0E6')

# # endregion

# region not working Fuselage mesh 

# num_fuselage_len = 20
# num_fuselage_height = 4
# nose = np.array([0., 0., -2.8])
# rear = np.array([124.750, 0., 2.019])
# nose_points_parametric = Fuselage.project(nose, grid_search_density_parameter=20)
# rear_points_parametric = Fuselage.project(rear)

# nose_points_m3l = geometry.evaluate(nose_points_parametric)
# rear_points_m3l = geometry.evaluate(rear_points_parametric)

# fuselage_linspace = m3l.linspace(nose_points_m3l, rear_points_m3l, num_fuselage_len)

# fueslage_top_points_parametric = Fuselage.project(fuselage_linspace.value + np.array([0., 0., 3]), direction=np.array([0.,0.,-1.]), plot=False, grid_search_density_parameter=20)
# fueslage_bottom_points_parametric = Fuselage.project(fuselage_linspace.value - np.array([0., 0., 3]), direction=np.array([0.,0.,1.]), plot=False, grid_search_density_parameter=20)

# fueslage_top_points_m3l = geometry.evaluate(fueslage_top_points_parametric)
# fueslage_bottom_points_m3l = geometry.evaluate(fueslage_bottom_points_parametric)

# fuesleage_mesh = m3l.linspace(fueslage_top_points_m3l.reshape((-1, 3)), fueslage_bottom_points_m3l.reshape((-1, 3)),  int(num_fuselage_height + 1))
# fuesleage_mesh.description = 'zero_y'

# if mesh_flag_fuselage:
#     geometry.plot_meshes(meshes=fuesleage_mesh, mesh_plot_types=['wireframe'], mesh_opacity=1., mesh_color='#F5F0E6')
# # endregion

# if plot_unupdated_vlm_mesh: 
#     geometry.plot_meshes(meshes=[wing_meshes.vlm_mesh, htail_meshes.vlm_mesh, right_strut_meshes.vlm_mesh, left_strut_meshes.vlm_mesh, vtail_meshes.vlm_mesh, fuesleage_mesh, left_jury_meshes.vlm_mesh, right_jury_meshes.vlm_mesh, gearpod_camber_surface])

# endregion

# endregion

# endregion

# region beam mesh 
ft_to_m = 0.3048

# wing coordinates
wing00 = np.array([68.035, 85.291, 4.704]) # * ft2m # Right tip leading edge
wing01 = np.array([71.790, 85.291, 4.704]) # * ft2m # Right tip trailing edge
wing10 = np.array([47.231, 0.000, 6.937])  # * ft2m # Center Leading Edge
wing11 = np.array([57.953, 0.000, 6.566]) # * ft2m # Center Trailing edge
wing20 = np.array([68.035, -85.291, 4.704]) # * ft2m # Left tip leading edge
wing21 = np.array([71.790, -85.291, 4.704]) # * ft2m # Left tip trailing edge

# htail coordinates
htail00 = np.array([132.002-10.0, 19.217+4.5, 18.993+3.5]) # * ft2m # Right tip leading edge
htail01 = np.array([135.993, 19.217, 18.993]) # * ft2m # Right tip trailing edge
htail10 = np.array([122.905, 0.000, 20.000]) # * ft2m # Center Leading Edge
htail11 = np.array([134.308, 0.000, 20.000]) # * ft2m # Center Trailing edge
htail20 = np.array([132.002-10, -19.217-4.5, 18.993+3.5]) # * ft2m # Left tip leading edge
htail21 = np.array([135.993, -19.217, 18.993]) # * ft2m # Left tip trailing edge

# strut coordinates
strut00 = np.array([55.573, -12.641, -4.200]) # left leading 1
strut06 = np.array([61.090, -48.994, 5.763]) # left leading 7
strut10 = np.array([57.309, -12.641, -4.200]) # left trailing 1
strut16 = np.array([62.902, -48.994, 5.763]) # left trailing 7
strut20 = np.array([55.573, 12.641, -4.200]) # right leading 1
strut26 = np.array([61.090, 48.994, 5.763]) # right leading 7
strut30 = np.array([57.309, 12.641, -4.200]) # right trailing 1
strut36 = np.array([62.902, 48.994, 5.763]) # right trailing 7

# region wing beam mesh 
    
num_wing_beam = 21

# wing_beam_mesh_flag = True
# wing_box_beam_mesh = make_1d_box_beam_mesh(
#     geometry=geometry,
#     wing_component=wing,
#     num_beam_nodes=num_wing_beam,
#     te_right=wing01,
#     te_left=wing21,
#     te_center=wing11,
#     le_left=wing20,
#     le_right=wing00,
#     le_center=wing10,
#     beam_width=0.5,
#     node_center=0.5,
#     plot=wing_beam_mesh_flag,
#     le_interp = 'linear',
#     te_interp = 'linear',
# )

wing_beam_mesh_flag = False
# wing_beam_mesh_flag = True
wing_box_beam_mesh = make_1d_box_beam_mesh(
    geometry=geometry,
    wing_component=wing,
    num_beam_nodes=num_wing_beam,
    # te_right=np.array([73., 85., 5.507]),
    te_right=wing_te_right,
    # te_left=np.array([73., -85., 5.507]),
    te_left=wing_te_left,
    # te_center=np.array([57.953, 0.000, 6.566]),
    te_center=wing_te_center,
    # le_left=np.array([66., -85., 5.647]),
    le_left=wing_le_left,
    # le_right=np.array([66., 85., 5.647]),
    le_right=wing_le_right,
    # le_center=np.array([47.231, 0.000, 6.937]),
    le_center=wing_le_center,
    beam_width=0.5,
    node_center=0.35,
    front_spar_location = 0.25,
    rear_spar_location = 0.75, 
    plot=wing_beam_mesh_flag,
    le_interp='ellipse',
    te_interp='linear',
    horizontal = 'yes',
)

# endregion

# region strut beam mesh
num_strut_beam = 9
# num_strut_beam = 21
# left_strut_beam_mesh_flag = True
# region left strut beam mesh

left_strut_box_beam_mesh = make_1d_box_beam_mesh(
    geometry=geometry,
    wing_component=strut,
    num_beam_nodes=num_strut_beam,
    te_right=strut10,
    te_left=strut16,
    #te_center=vertex13_left_strut,
    le_left=strut06,
    le_right=strut00,
    #le_center=vertex03_left_strut,
    beam_width=0.5,
    node_center=0.35,
    front_spar_location = 0.25,
    rear_spar_location = 0.75, 
    plot=left_strut_beam_mesh_flag,
    le_interp = 'linear',
    te_interp = 'linear',
    horizontal = 'yes',
)
# endregion

# region right strut beam mesh
# right_strut_beam_mesh_flag = True
right_strut_box_beam_mesh = make_1d_box_beam_mesh(
    geometry=geometry,
    wing_component=strut,
    num_beam_nodes=num_strut_beam,
    te_right=strut30,
    te_left=strut36,
    #te_center=vertex13_left_strut,
    le_left=strut26,
    le_right=strut20,
    #le_center=vertex03_left_strut,
    beam_width=0.5,
    node_center=0.35,
    front_spar_location = 0.25,
    rear_spar_location = 0.75, 
    plot=right_strut_beam_mesh_flag,
    le_interp = 'linear',
    te_interp = 'linear',
    horizontal = 'yes',
)
# endregion
# endregion

# region jury beam mesh

# region left jury beam mesh
num_jury_beam = 3
# left_jury_beam_mesh_flag = True
left_jury_box_beam_mesh = make_1d_box_beam_mesh(
    geometry=geometry,
    wing_component=jury,
    num_beam_nodes=num_jury_beam,
    te_right=np.array([59.237, -30.818, 6.181]) ,
    te_left=np.array([59.676, -30.818, 0.732]) ,
    le_left=np.array([58.721, -30.818, 0.732,]) ,
    le_right=np.array([58.052, -30.818, 6.181]) ,
    beam_width=0.5,
    node_center=0.35,
    front_spar_location = 0.25,
    rear_spar_location = 0.75, 
    plot=left_jury_beam_mesh_flag,
    le_interp = 'linear',
    te_interp = 'linear',
    horizontal = 'no',
)
# endregion

# region right jury beam mesh
num_jury_beam = 3
# right_jury_beam_mesh_flag = True
right_jury_box_beam_mesh = make_1d_box_beam_mesh(
    geometry=geometry,
    wing_component=jury,
    num_beam_nodes=num_jury_beam,
    te_right=np.array([59.237, 30.818, 6.181]) ,
    te_left=np.array([59.676, 30.818, 0.732]) ,
    le_left=np.array([58.721, 30.818, 0.732,]) ,
    le_right=np.array([58.052, 30.818, 6.181]) ,
    beam_width=0.5,
    node_center=0.35,
    front_spar_location = 0.25,
    rear_spar_location = 0.75, 
    plot=right_jury_beam_mesh_flag,
    le_interp = 'linear',
    te_interp = 'linear',
    horizontal = 'no',
)
# endregion
# endregion

# endregion


def calculate_angle(point1, point2):
    # Calculate the vectors from the origin to each point
    vector1 = np.array(point1)
    vector2 = np.array(point2)
    
    # Calculate the dot product and magnitudes of the vectors
    dot_product = np.dot(vector1, vector2)
    magnitude1 = np.linalg.norm(vector1)
    magnitude2 = np.linalg.norm(vector2)
    
    # Calculate the angle in radians
    angle_radians = np.arccos(dot_product / (magnitude1 * magnitude2))
    
    # Convert the angle to degrees
    angle_degrees = np.degrees(angle_radians)
    
    return angle_degrees

# Example points
point1 = point40_wing
point2 = vertex05_left_strut

# Calculate the angle between the two points
angle_beta_not_m3l = calculate_angle(point1, point2)
angle_beta = system_model.create_input(name = 'angle_beta', shape = (1,), val = angle_beta_not_m3l)

# Component surface areas
component_list = [Fuselage, wing, htail, vtail, jury, gearpod, strut]
surface_area_list = compute_component_surface_area(
    component_list=component_list,
    geometry=geometry,
    parametric_mesh_grid_num=20,
    plot=False,
)

S_ref_area = surface_area_list[1] / 1.969576502
h_tail_area = surface_area_list[2] / 2.1
v_tail_area = surface_area_list[3] / 2.18
jury_area = surface_area_list[4] / 2.11
strut_area = surface_area_list[6] / 2.07
wing_AR = ((wing_span_dv)**2 )/ S_ref_area
print('wing area', S_ref_area.value)
print('htail area', h_tail_area.value)
print('v_tail area', v_tail_area.value)
print('jury area', jury_area.value)
print('strut area', strut_area.value)
print('wing_AR', wing_AR.value)