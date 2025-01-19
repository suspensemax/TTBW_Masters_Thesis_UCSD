'''Example TBW geometry setup -- Structural analysis (Final? maybe?)'''

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
geometry.refit(parallelize=True, order=(5, 5))
if geometry_plot:
    geometry.plot()

plot_meshes = False
geometry_dv = False


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
jury = geometry.declare_component(component_name='jury', b_spline_search_names=['Jury'])
if components_plot:
    jury.plot()
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

wing_span_dv = system_model.create_input(name='wing_span_dv', shape = (1,), val = 170.64)
wing_tip_chord_left_dv = system_model.create_input(name='wing_tip_chord_left_dv', shape= (1,), val = 3.8)
wing_root_chord_dv = system_model.create_input(name='wing_root_chord_dv', shape= (1,), val = 10.73)
wing_mid_chord_left_dv = system_model.create_input(name = 'wing_mid_chord_left_dv', shape=(1,), val= 9.5)

# endregion

# region mesh

# region Wing Mesh

# num_spanwise_wing_vlm = 25
# num_chordwise_wing_vlm = 14 
num_spanwise_wing_vlm = 21
num_chordwise_wing_vlm = 5 

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

# mesh_flag_wing = True
wing_meshes = make_vlm_camber_mesh(
    geometry=geometry,
    wing_component=wing,
    num_spanwise=num_spanwise_wing_vlm,
    num_chordwise=num_chordwise_wing_vlm,
    # te_right=np.array([73., 85., 5.507]),
    te_right=wing_te_right,
    # te_left=np.array([73., -85., 5.507]),
    te_left=wing_te_left,
    # te_center=np.array([57.953, 0.000, 6.566]),
    te_center=wing_te_center,
    # le_left=np.array([66., -85., 5.647]),ss
    le_left=wing_le_left,
    # le_right=np.array([66., 85., 5.647]),
    le_right=wing_le_right,
    # le_center=np.array([47.231, 0.000, 6.937]),
    le_center=wing_le_center,
    grid_search_density_parameter=150,
    le_interp='ellipse',
    te_interp='linear',
    off_set_x=0.2,
    bunching_cos=False,
    plot=mesh_flag_wing,
    mirror=True,
)

manual = False

if manual: 

    num_spanwise_wing_vlm_1 = 6
    num_spanwise_wing_vlm_3 = 8

    wing_leading_edge_points_le_1 = np.linspace(point00_wing, point10_wing, num_spanwise_wing_vlm_1)[1:-1]
    wing_leading_edge_points_le_2 = np.linspace(point10_wing, point30_wing, num_spanwise_wing_vlm_3)[1:-1]
    wing_leading_edge_points_le_3 = np.linspace(point30_wing, point50_wing, num_spanwise_wing_vlm_3)[1:-1]
    wing_leading_edge_points_le_4 = np.linspace(point50_wing, point60_wing, num_spanwise_wing_vlm_1)[1:-1]


    wing_leading_edge_points = np.vstack([point00_wing, wing_leading_edge_points_le_1, point10_wing, wing_leading_edge_points_le_2, 
                                            point30_wing, wing_leading_edge_points_le_3, point50_wing, wing_leading_edge_points_le_4,
                                            point60_wing])


    # wing_leading_edge_points = np.concatenate((np.linspace(point00_wing, point30_wing, int(num_spanwise_vlm/2+1))[0:-1_wing,:], np.linspace(point30_wing, point60_wing, int(num_spanwise_vlm/2+1))), axis=0)

    wing_trailing_edge_points_te_1 = np.linspace(point01_wing, point11_wing, num_spanwise_wing_vlm_1)[1:-1]
    wing_trailing_edge_points_te_2 = np.linspace(point11_wing, point31_wing, num_spanwise_wing_vlm_3)[1:-1]
    wing_trailing_edge_points_te_3 = np.linspace(point31_wing, point51_wing, num_spanwise_wing_vlm_3)[1:-1]
    wing_trailing_edge_points_te_4 = np.linspace(point51_wing, point61_wing, num_spanwise_wing_vlm_1)[1:-1]

    wing_trailing_edge_points = np.vstack([point01_wing, wing_trailing_edge_points_te_1, point11_wing, wing_trailing_edge_points_te_2, 
                                            point31_wing, wing_trailing_edge_points_te_3, point51_wing, wing_trailing_edge_points_te_4,
                                            point61_wing])

    # wing_trailing_edge_points = np.concatenate((np.linspace(point01_wing, point31_wing, int(num_spanwise_vlm/2+1))[0:-1,:], np.linspace(point31_wing, point61_wing, int(num_spanwise_vlm/2+1))), axis=0)

    wing_leading_edge = wing.project(wing_leading_edge_points, plot=do_plots)
    wing_trailing_edge = wing.project(wing_trailing_edge_points, plot=do_plots)

    wing_le_coord = geometry.evaluate(wing_leading_edge).reshape((-1, 3))
    wing_te_coord = geometry.evaluate(wing_trailing_edge).reshape((-1, 3))
    # A user can print the values of m3l variable by calling '.value'
    if print_coordinates:
        print(wing_le_coord.value)
        print(wing_te_coord.value)

    # Getting a linearly spaced (2-D) array between the leading and trailing edge
    wing_chord = m3l.linspace(wing_le_coord, wing_te_coord, num_chordwise_wing_vlm)

    # Projecting the 2-D array onto the upper and lower surface of the wing to get the camber surface mesh
    wing_upper_surface_wireframe_parametric = wing.project(wing_chord.value + np.array([0., 0., 1.]), direction=np.array([0., 0., 1.]), grid_search_density_parameter=25, plot=False)
    wing_lower_surface_wireframe_parametric = wing.project(wing_chord.value - np.array([0., 0., 1.]), direction=np.array([0., 0., -1.]), grid_search_density_parameter=25, plot=False)
    wing_upper_surface_wireframe = geometry.evaluate(wing_upper_surface_wireframe_parametric).reshape((num_chordwise_wing_vlm, num_spanwise_wing_vlm, 3))
    wing_lower_surface_wireframe = geometry.evaluate(wing_lower_surface_wireframe_parametric).reshape((num_chordwise_wing_vlm, num_spanwise_wing_vlm, 3))

    wing_camber_surface = m3l.linspace(wing_upper_surface_wireframe, wing_lower_surface_wireframe, 1)#.reshape((-1, 3))
    
    # Optionally, the resulting camber surface mesh can be plotted
    if mesh_flag_wing:
        geometry.plot_meshes(meshes=wing_camber_surface, mesh_plot_types=['wireframe'], mesh_opacity=1., mesh_color='#F5F0E6')

    parametric_mesh_grid_num = 20
    surfaces = wing.b_spline_names
    wing_oml_para_mesh = []
    for name in surfaces:
        for u in np.linspace(0,1,parametric_mesh_grid_num):
            for v in np.linspace(0,1,parametric_mesh_grid_num):
                wing_oml_para_mesh.append((name, np.array([u,v]).reshape((1,2))))

    wing_oml_mesh = geometry.evaluate(wing_oml_para_mesh).reshape((-1, 3))
    # print(oml_mesh.shape)

    if mesh_flag_wing_oml:
        geometry.plot_meshes(meshes=wing_oml_mesh, mesh_plot_types=['wireframe'], mesh_opacity=1., mesh_color='#F5F0E6')

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

manual = False    

if manual:

    num_spanwise_htail_vlm = 11
    num_chordwise_htail_vlm = 5
    point00_htail = np.array([132.002-10.0, 19.217+4.5, 18.993+3.5]) # * ft2m # Right tip leading edge
    point01_htail = np.array([135.993, 19.217, 18.993]) # * ft2m # Right tip trailing edge
    point10_htail = np.array([122.905, 0.000, 20.000]) # * ft2m # Center Leading Edge
    point11_htail = np.array([134.308, 0.000, 20.000]) # * ft2m # Center Trailing edge
    point20_htail = np.array([132.002-10, -19.217-4.5, 18.993+3.5]) # * ft2m # Left tip leading edge
    point21_htail = np.array([135.993, -19.217, 18.993]) # * ft2m # Left tip trailing edge

    htail_leading_edge_points = np.linspace(point00_htail, point20_htail, num_spanwise_htail_vlm)
    htail_trailing_edge_points = np.linspace(point01_htail, point21_htail, num_spanwise_htail_vlm)

    htail_leading_edge = htail.project(htail_leading_edge_points, plot=plot_htail_mesh)
    htail_trailing_edge = htail.project(htail_trailing_edge_points, plot=plot_htail_mesh)

    htail_le_coord = geometry.evaluate(htail_leading_edge).reshape((-1, 3))
    htail_te_coord = geometry.evaluate(htail_trailing_edge).reshape((-1, 3))
    # A user can print the values of m3l variable by calling '.value'
    if print_coordinates:
        print(htail_le_coord.value)
        print(htail_te_coord.value)

    # Getting a linearly spaced (2-D) array between the leading and trailing edge
    htail_chord = m3l.linspace(htail_le_coord, htail_te_coord, num_chordwise_htail_vlm)

    # Projecting the 2-D array onto the upper and lower surface of the wing to get the camber surface mesh
    htail_upper_surface_wireframe_parametric = htail.project(htail_chord.value + np.array([0., 0., 1.]), direction=np.array([0., 0., 1.]), grid_search_density_parameter=25, plot=False)
    htail_lower_surface_wireframe_parametric = htail.project(htail_chord.value - np.array([0., 0., 1.]), direction=np.array([0., 0., -1.]), grid_search_density_parameter=25, plot=False)
    htail_upper_surface_wireframe = geometry.evaluate(htail_upper_surface_wireframe_parametric).reshape((num_chordwise_htail_vlm, num_spanwise_htail_vlm, 3))
    htail_lower_surface_wireframe = geometry.evaluate(htail_lower_surface_wireframe_parametric).reshape((num_chordwise_htail_vlm, num_spanwise_htail_vlm, 3))

    htail_camber_surface = m3l.linspace(htail_upper_surface_wireframe, htail_lower_surface_wireframe, 1)#.reshape((-1, 3))

    # Optionally, the resulting camber surface mesh can be plotted
    if mesh_flag_htail:
        geometry.plot_meshes(meshes=htail_camber_surface, mesh_plot_types=['wireframe'], mesh_opacity=1., mesh_color='#F5F0E6')

    parametric_mesh_grid_num = 20
    surfaces = htail.b_spline_names
    htail_oml_para_mesh = []
    for name in surfaces:
        for u in np.linspace(0,1,parametric_mesh_grid_num):
            for v in np.linspace(0,1,parametric_mesh_grid_num):
                htail_oml_para_mesh.append((name, np.array([u,v]).reshape((1,2))))

    htail_oml_mesh = geometry.evaluate(htail_oml_para_mesh).reshape((-1, 3))
    # print(oml_mesh.shape)

    if mesh_flag_htail_oml:
        geometry.plot_meshes(meshes=htail_oml_mesh, mesh_plot_types=['wireframe'], mesh_opacity=1., mesh_color='#F5F0E6')

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

# mesh_flag_left_strut = True
left_strut_meshes = make_vlm_camber_mesh(
    geometry=geometry,
    wing_component=strut,
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

manual = False

if manual:
    strut_small_vlm = 3
    strut_middle_vlm = 6
    num_spanwise_strut_vlm = 21
    num_chordwise_strut_vlm = 5 

    vertex00_left_strut = np.array([55.573, -12.641, -4.200]) # left leading 1
    vertex01_left_strut = np.array([55.034, -16.277, -3.204]) # left leading 2
    vertex02_left_strut = np.array([56.422, -25.365, -0.713]) # left leading 3
    vertex03_left_strut = np.array([58.313, -30.818, 0.782]) # left leading 4
    vertex04_left_strut = np.array([58.089, -36.271, 2.276]) # left leading 5
    vertex05_left_strut = np.array([59.477, -45.359, 4.767]) # left leading 6
    vertex06_left_strut = np.array([61.090, -48.994, 5.763]) # left leading 7

    left_strut_leading_edge_points_le_1 = np.linspace(vertex00_left_strut, vertex01_left_strut, strut_small_vlm)[1:-1]
    left_strut_leading_edge_points_le_2 = np.linspace(vertex01_left_strut, vertex02_left_strut, strut_middle_vlm)[1:-1]
    left_strut_leading_edge_points_le_3 = np.linspace(vertex02_left_strut, vertex03_left_strut, strut_small_vlm + 1)[1:-1]
    left_strut_leading_edge_points_le_4 = np.linspace(vertex03_left_strut, vertex04_left_strut, strut_small_vlm + 1)[1:-1]
    left_strut_leading_edge_points_le_5 = np.linspace(vertex04_left_strut, vertex05_left_strut, strut_middle_vlm)[1:-1]
    left_strut_leading_edge_points_le_6_not_working = np.linspace(vertex05_left_strut, vertex06_left_strut, strut_small_vlm)[1:-1]
    left_strut_leading_edge_points_le_6 = np.array([59.2835 ,-47.65 ,  5.265])
    left_strut_le_joined_vertices = np.vstack([vertex00_left_strut, left_strut_leading_edge_points_le_1, vertex01_left_strut, left_strut_leading_edge_points_le_2, 
                                            vertex02_left_strut, left_strut_leading_edge_points_le_3, vertex03_left_strut, left_strut_leading_edge_points_le_4,
                                            vertex04_left_strut, left_strut_leading_edge_points_le_5, vertex05_left_strut,left_strut_leading_edge_points_le_6, vertex06_left_strut])

    left_strut_leading_edge = strut.project(left_strut_le_joined_vertices, plot=plot_left_strut_mesh)

    vertex10_left_strut = np.array([57.309, -12.641, -4.200]) # left trailing 1
    vertex11_left_strut = np.array([58.959, -16.277, -3.204]) # left trailing 2
    vertex12_left_strut = np.array([60.348, -25.365, -0.713]) # left trailing 3
    vertex13_left_strut = np.array([60.124, -30.818, 0.782]) # left trailing 4
    vertex14_left_strut = np.array([62.014, -36.271, 2.276]) # left trailing 5
    vertex15_left_strut = np.array([63.403, -45.359, 4.767]) # left trailing 6                
    vertex16_left_strut = np.array([62.902, -48.994, 5.763]) # left trailing 7

    left_strut_trailing_edge_points_te_1 = np.linspace(vertex10_left_strut, vertex11_left_strut, strut_small_vlm)[1:-1]
    left_strut_trailing_edge_points_te_2 = np.linspace(vertex11_left_strut, vertex12_left_strut, strut_middle_vlm)[1:-1]
    left_strut_trailing_edge_points_te_3 = np.linspace(vertex12_left_strut, vertex13_left_strut, strut_small_vlm + 1)[1:-1]
    left_strut_trailing_edge_points_te_4 = np.linspace(vertex13_left_strut, vertex14_left_strut, strut_small_vlm + 1)[1:-1]
    left_strut_trailing_edge_points_te_5 = np.linspace(vertex14_left_strut, vertex15_left_strut, strut_middle_vlm)[1:-1]
    left_strut_trailing_edge_points_te_6 = np.linspace(vertex15_left_strut, vertex16_left_strut, strut_small_vlm)[1:-1]

    left_strut_te_joined_vertices = np.vstack([vertex10_left_strut, left_strut_trailing_edge_points_te_1, vertex11_left_strut, left_strut_trailing_edge_points_te_2, 
                                            vertex12_left_strut, left_strut_trailing_edge_points_te_3, vertex13_left_strut, left_strut_trailing_edge_points_te_4,
                                            vertex14_left_strut, left_strut_trailing_edge_points_te_5, vertex15_left_strut, left_strut_trailing_edge_points_te_6, vertex16_left_strut])

    left_strut_trailing_edge = strut.project(left_strut_te_joined_vertices, plot=plot_left_strut_mesh)

    left_strut_le_coord = geometry.evaluate(left_strut_leading_edge).reshape((-1, 3))
    left_strut_te_coord = geometry.evaluate(left_strut_trailing_edge).reshape((-1, 3))
    # A user can print the values of m3l variable by calling '.value'
    if print_coordinates:
        print(left_strut_le_coord.value)
        print(left_strut_te_coord.value)

    # Getting a linearly spaced (2-D) array between the leading and trailing edge
    left_strut_chord = m3l.linspace(left_strut_le_coord, left_strut_te_coord, num_chordwise_strut_vlm)

    # Projecting the 2-D array onto the upper and lower surface of the wing to get the camber surface mesh
    left_strut_upper_surface_wireframe_parametric = strut.project(left_strut_chord.value + np.array([0., 0., 1.]), direction=np.array([0., 0., 1.]), grid_search_density_parameter=25, plot=False)
    left_strut_lower_surface_wireframe_parametric = strut.project(left_strut_chord.value - np.array([0., 0., 1.]), direction=np.array([0., 0., -1.]), grid_search_density_parameter=25, plot=False)
    left_strut_upper_surface_wireframe = geometry.evaluate(left_strut_upper_surface_wireframe_parametric).reshape((num_chordwise_strut_vlm, num_spanwise_strut_vlm, 3))
    left_strut_lower_surface_wireframe = geometry.evaluate(left_strut_lower_surface_wireframe_parametric).reshape((num_chordwise_strut_vlm, num_spanwise_strut_vlm, 3))

    left_strut_camber_surface = m3l.linspace(left_strut_upper_surface_wireframe, left_strut_lower_surface_wireframe, 1)#.reshape((-1, 3))

    # Optionally, the resulting camber surface mesh can be plotted
    if mesh_flag_left_strut:
        geometry.plot_meshes(meshes=left_strut_camber_surface, mesh_plot_types=['wireframe'], mesh_opacity=1., mesh_color='#F5F0E6')

    parametric_mesh_grid_num = 20
    surfaces = strut.b_spline_names
    left_strut_oml_para_mesh = []
    for name in surfaces:
        for u in np.linspace(0,1,parametric_mesh_grid_num):
            for v in np.linspace(0,1,parametric_mesh_grid_num):
                left_strut_oml_para_mesh.append((name, np.array([u,v]).reshape((1,2))))

    left_strut_oml_mesh = geometry.evaluate(left_strut_oml_para_mesh).reshape((-1, 3))
    # print(oml_mesh.shape)

    if mesh_flag_left_strut_oml:
        geometry.plot_meshes(meshes=left_strut_oml_mesh, mesh_plot_types=['wireframe'], mesh_opacity=1., mesh_color='#F5F0E6')

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
    wing_component=strut,
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

manual = False

if manual:        

    strut_small_vlm = 3
    strut_middle_vlm = 6
    vertex00_right_strut = np.array([55.573, 12.641, -4.200]) # right leading 1
    vertex01_right_strut = np.array([55.034, 16.277, -3.204]) # right leading 2
    vertex02_right_strut = np.array([56.422, 25.365, -0.713]) # right leading 3
    vertex03_right_strut = np.array([58.313, 30.818, 0.782]) # right leading 4
    vertex04_right_strut = np.array([58.089, 36.271, 2.276]) # right leading 5
    vertex05_right_strut = np.array([59.477, 45.359, 4.767]) # right leading 6
    vertex06_right_strut = np.array([61.090, 48.994, 5.763]) # right leading 7

    right_strut_leading_edge_points_le_1 = np.linspace(vertex00_right_strut, vertex01_right_strut, strut_small_vlm)[1:-1]
    right_strut_leading_edge_points_le_2 = np.linspace(vertex01_right_strut, vertex02_right_strut, strut_middle_vlm)[1:-1]
    right_strut_leading_edge_points_le_3 = np.linspace(vertex02_right_strut, vertex03_right_strut, strut_small_vlm + 1)[1:-1]
    right_strut_leading_edge_points_le_4 = np.linspace(vertex03_right_strut, vertex04_right_strut, strut_small_vlm + 1)[1:-1]
    right_strut_leading_edge_points_le_5 = np.linspace(vertex04_right_strut, vertex05_right_strut, strut_middle_vlm)[1:-1]
    right_strut_leading_edge_points_le_6 = np.linspace(vertex05_right_strut, vertex06_right_strut, strut_small_vlm)[1:-1]

    vertex10_right_strut = np.array([57.309, 12.641, -4.200]) # right trailing 1
    vertex11_right_strut = np.array([58.959, 16.277, -3.204]) # right trailing 2
    vertex12_right_strut = np.array([60.348, 25.365, -0.713]) # right trailing 3
    vertex13_right_strut = np.array([60.124, 30.818, 0.782]) # right trailing 4
    vertex14_right_strut = np.array([62.014, 36.271, 2.276]) # right trailing 5
    vertex15_right_strut = np.array([63.403, 45.359, 4.767]) # right trailing 6                
    vertex16_right_strut = np.array([62.902, 48.994, 5.763]) # right trailing 7

    right_strut_trailing_edge_points_te_1 = np.linspace(vertex10_right_strut, vertex11_right_strut, strut_small_vlm)[1:-1]
    right_strut_trailing_edge_points_te_2 = np.linspace(vertex11_right_strut, vertex12_right_strut, strut_middle_vlm)[1:-1]
    right_strut_trailing_edge_points_te_3 = np.linspace(vertex12_right_strut, vertex13_right_strut, strut_small_vlm + 1)[1:-1]
    right_strut_trailing_edge_points_te_4 = np.linspace(vertex13_right_strut, vertex14_right_strut, strut_small_vlm + 1)[1:-1]
    right_strut_trailing_edge_points_te_5 = np.linspace(vertex14_right_strut, vertex15_right_strut, strut_middle_vlm)[1:-1]
    right_strut_trailing_edge_points_te_6 = np.linspace(vertex15_right_strut, vertex16_right_strut, strut_small_vlm)[1:-1]

    right_strut_le_joined_vertices = np.vstack([vertex00_right_strut, right_strut_leading_edge_points_le_1, vertex01_right_strut, right_strut_leading_edge_points_le_2, 
                                            vertex02_right_strut, right_strut_leading_edge_points_le_3, vertex03_right_strut, right_strut_leading_edge_points_le_4,
                                            vertex04_right_strut, right_strut_leading_edge_points_le_5, vertex05_right_strut, right_strut_leading_edge_points_le_6, vertex06_right_strut])

    right_strut_te_joined_vertices = np.vstack([vertex10_right_strut, right_strut_trailing_edge_points_te_1, vertex11_right_strut, right_strut_trailing_edge_points_te_2, 
                                            vertex12_right_strut, right_strut_trailing_edge_points_te_3, vertex13_right_strut, right_strut_trailing_edge_points_te_4,
                                            vertex14_right_strut, right_strut_trailing_edge_points_te_5, vertex15_right_strut, right_strut_trailing_edge_points_te_6, vertex16_right_strut])

    right_strut_trailing_edge = strut.project(right_strut_te_joined_vertices, plot=plot_right_strut_mesh)
    right_strut_leading_edge = strut.project(right_strut_le_joined_vertices, plot=plot_right_strut_mesh)

    right_strut_le_coord = geometry.evaluate(right_strut_leading_edge).reshape((-1, 3))
    right_strut_te_coord = geometry.evaluate(right_strut_trailing_edge).reshape((-1, 3))
    # A user can print the values of m3l variable by calling '.value'
    if print_coordinates:
        print(right_strut_le_coord.value)
        print(right_strut_te_coord.value)

    # Getting a linearly spaced (2-D) array between the leading and trailing edge
    right_strut_chord = m3l.linspace(right_strut_le_coord, right_strut_te_coord, num_chordwise_strut_vlm)

    # Projecting the 2-D array onto the upper and lower surface of the wing to get the camber surface mesh
    right_strut_upper_surface_wireframe_parametric = strut.project(right_strut_chord.value + np.array([0., 0., 1.]), direction=np.array([0., 0., 1.]), grid_search_density_parameter=25, plot=False)
    right_strut_lower_surface_wireframe_parametric = strut.project(right_strut_chord.value - np.array([0., 0., 1.]), direction=np.array([0., 0., -1.]), grid_search_density_parameter=25, plot=False)
    right_strut_upper_surface_wireframe = geometry.evaluate(right_strut_upper_surface_wireframe_parametric).reshape((num_chordwise_strut_vlm, num_spanwise_strut_vlm, 3))
    right_strut_lower_surface_wireframe = geometry.evaluate(right_strut_lower_surface_wireframe_parametric).reshape((num_chordwise_strut_vlm, num_spanwise_strut_vlm, 3))

    right_strut_camber_surface = m3l.linspace(right_strut_upper_surface_wireframe, right_strut_lower_surface_wireframe, 1)#.reshape((-1, 3))

    # Optionally, the resulting camber surface mesh can be plotted
    if mesh_flag_right_strut:
        geometry.plot_meshes(meshes=right_strut_camber_surface, mesh_plot_types=['wireframe'], mesh_opacity=1., mesh_color='#F5F0E6')

    parametric_mesh_grid_num = 20
    surfaces = strut.b_spline_names
    right_strut_oml_para_mesh = []
    for name in surfaces:
        for u in np.linspace(0,1,parametric_mesh_grid_num):
            for v in np.linspace(0,1,parametric_mesh_grid_num):
                right_strut_oml_para_mesh.append((name, np.array([u,v]).reshape((1,2))))

    right_strut_oml_mesh = geometry.evaluate(right_strut_oml_para_mesh).reshape((-1, 3))
    # print(oml_mesh.shape)

    if mesh_flag_right_strut_oml:
        geometry.plot_meshes(meshes=right_strut_oml_mesh, mesh_plot_types=['wireframe'], mesh_opacity=1., mesh_color='#F5F0E6')

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
    wing_component=jury,
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
    wing_component=jury,
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


if plot_unupdated_vlm_mesh: 
    geometry.plot_meshes(meshes=[wing_meshes.vlm_mesh, htail_meshes.vlm_mesh, right_strut_meshes.vlm_mesh, left_strut_meshes.vlm_mesh, vtail_meshes.vlm_mesh, left_jury_meshes.vlm_mesh, right_jury_meshes.vlm_mesh])
plot_unupdated_oml_mesh = False
if plot_unupdated_oml_mesh: 
    geometry.plot_meshes(meshes=[wing_meshes.oml_mesh,])

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

# region projections for drag components
    
# Wing 
wing_mid_le_parametric = wing.project(point10_wing)
wing_mid_te_parametric = wing.project(point11_wing)

wing_root_le_parametric = wing.project(point30_wing)
wing_root_te_parametric = wing.project(point31_wing)

wing_tip_le_parametric = wing.project(point00_wing)
wing_tip_te_parametric = wing.project(point01_wing)

wing_le_right_parametric = wing.project(point00_wing)
wing_le_left__parametric = wing.project(point60_wing)

# htail
h_tail_mid_le_parametric = htail.project(point10_htail)
h_tail_mid_te_parametric = htail.project(point11_htail)

# strut
# left strut
left_strut_mid_le_parametric = strut.project(vertex03_left_strut)
left_strut_mid_te_parametric = strut.project(vertex13_left_strut)
# right strut
right_strut_mid_le_parametric = strut.project(vertex03_right_strut)
right_strut_mid_te_parametric = strut.project(vertex13_right_strut)

# vtail
vtail_mid_le_parametric = vtail.project(np.array([112.842, 0.0, 12.127]))
vtail_mid_te_parametric = vtail.project(np.array([130.0955, 0.0, 12.127]))

# # fuselage
# fuselage_l1_parametric = Fuselage.project(nose)
# fuselage_l2_parametric = Fuselage.project(rear)

# fuselage_d1_parametric = Fuselage.project(np.array([50.577, -5.974, 1.843]))
# fuselage_d2_parametric = Fuselage.project(np.array([50.577, 5.974, 1.843]))

# jury 
# left jury
left_jury_mid_le_parametric = jury.project(np.array([58.3865 , -30.818, 3.4565]))
left_jury_mid_te_parametric = jury.project(np.array([59.4565 , -30.818, 3.4565]))
# right jury
right_jury_mid_le_parametric = jury.project(np.array([58.3865 , 30.818, 3.4565]))
right_jury_mid_te_parametric = jury.project(np.array([59.4565 , 30.818, 3.4565]))

# gearpod
# gearpod_l1_parametric = gearpod.project(gearpod_le_center)
# gearpod_l2_parametric = gearpod.project(gearpod_te_center)

# gearpod_d1_parametric = gearpod.project(gearpod_strut_le_right)
# gearpod_d2_parametric = gearpod.project(gearpod_strut_le_left)

# endregion

# region Drag components 
    
# Component surface areas
component_list = [Fuselage, wing, htail, vtail, jury, gearpod, strut]
surface_area_list = compute_component_surface_area(
    component_list=component_list,
    geometry=geometry,
    parametric_mesh_grid_num=20,
    plot=False,
)

# # Fuselage
# fuselage_l1 = geometry.evaluate(fuselage_l1_parametric).reshape((-1, 3))
# fuselage_l2 = geometry.evaluate(fuselage_l2_parametric).reshape((-1, 3))
# fuselage_length = m3l.norm(fuselage_l2-fuselage_l1)

# fuselage_d1 = geometry.evaluate(fuselage_d1_parametric).reshape((-1, 3))
# fuselage_d2=  geometry.evaluate(fuselage_d2_parametric).reshape((-1, 3))
# fuselage_diameter = m3l.norm(fuselage_d2-fuselage_d1)

# fuselage_drag_comp = DragComponent(
#     component_type='fuselage',
#     wetted_area=surface_area_list[0],
#     characteristic_length=fuselage_length,
#     characteristic_diameter=fuselage_diameter,
#     Q=2.9,
# )

# wing
wing_mid_le = geometry.evaluate(wing_mid_le_parametric).reshape((-1, 3))
wing_mid_te = geometry.evaluate(wing_mid_te_parametric).reshape((-1, 3))
wing_mid_chord_length = m3l.norm(wing_mid_le-wing_mid_te)

wing_tip_le = geometry.evaluate(wing_tip_le_parametric).reshape((-1, 3))
wing_tip_te = geometry.evaluate(wing_tip_te_parametric).reshape((-1, 3))
wing_tip_chord_length = m3l.norm(wing_tip_le-wing_tip_te)

wing_root_le = geometry.evaluate(wing_root_le_parametric).reshape((-1, 3))
wing_root_te = geometry.evaluate(wing_root_te_parametric).reshape((-1, 3))
wing_root_chord_length = m3l.norm(wing_root_le-wing_root_te)

wing_le_right_mapped_array = geometry.evaluate(wing_le_right_parametric).reshape((-1, 3))
wing_le_left_mapped_array = geometry.evaluate(wing_le_left__parametric).reshape((-1, 3))
wing_span = m3l.norm(wing_le_left_mapped_array - wing_le_right_mapped_array)

wing_drag_comp = DragComponent(
    component_type='wing',
    wetted_area=surface_area_list[1],
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
    component_type='tail',
    wetted_area=surface_area_list[2],
    characteristic_length=h_tail_chord_length,
    thickness_to_chord=0.15,
    x_cm=0.3,
    Q=1.5,
)

# # v tail
# vtail_mid_le = geometry.evaluate(vtail_mid_le_parametric).reshape((-1, 3))
# vtail_mid_te = geometry.evaluate(vtail_mid_te_parametric).reshape((-1, 3))
# vtail_chord_length = m3l.norm(vtail_mid_le-vtail_mid_te)

# vtail_drag_comp = DragComponent(
#     component_type='tail',
#     wetted_area=surface_area_list[3],
#     characteristic_length=vtail_chord_length,
#     thickness_to_chord=0.115,
#     x_cm=0.2,
#     Q=1.5,
# )

# jury
# left jury
left_jury_mid_le = geometry.evaluate(left_jury_mid_le_parametric).reshape((-1, 3))
left_jury_mid_te = geometry.evaluate(left_jury_mid_te_parametric).reshape((-1, 3))
left_jury_chord_length = m3l.norm(left_jury_mid_le-left_jury_mid_te)

left_jury_drag_comp = DragComponent(
    component_type='pylon',
    wetted_area=surface_area_list[4],
    characteristic_length=left_jury_chord_length,
    thickness_to_chord=0.115,
    x_cm=0.2,
    Q=1.5,
)

# right jury
right_jury_mid_le = geometry.evaluate(right_jury_mid_le_parametric).reshape((-1, 3))
right_jury_mid_te = geometry.evaluate(right_jury_mid_te_parametric).reshape((-1, 3))
right_jury_chord_length = m3l.norm(right_jury_mid_le-right_jury_mid_te)

right_jury_drag_comp = DragComponent(
    component_type='pylon',
    wetted_area=surface_area_list[4],
    characteristic_length=right_jury_chord_length,
    thickness_to_chord=0.115,
    x_cm=0.2,
    Q=1.5,
)

# gearpod
# gearpod_l1 = geometry.evaluate(gearpod_l1_parametric).reshape((-1, 3))
# gearpod_l2 = geometry.evaluate(gearpod_l2_parametric).reshape((-1, 3))
# gearpod_length = m3l.norm(gearpod_l2-gearpod_l1)

# gearpod_d1 = geometry.evaluate(gearpod_d1_parametric).reshape((-1, 3))
# gearpod_d2=  geometry.evaluate(gearpod_d2_parametric).reshape((-1, 3))
# gearpod_diameter = m3l.norm(gearpod_d2-gearpod_d1)

# gearpod_drag_comp = DragComponent(
#     component_type='fuselage',
#     wetted_area=surface_area_list[5],
#     characteristic_length=gearpod_length,
#     characteristic_diameter=gearpod_diameter,
#     Q=2.9,
# )

# strut
# left strut
left_strut_mid_le = geometry.evaluate(left_strut_mid_le_parametric).reshape((-1, 3))
left_strut_mid_te = geometry.evaluate(left_strut_mid_te_parametric).reshape((-1, 3))
left_strut_chord_length = m3l.norm(left_strut_mid_le-left_strut_mid_te)

left_strut_drag_comp = DragComponent(
    component_type='strut',
    wetted_area=surface_area_list[6],
    characteristic_length=left_strut_chord_length,
    thickness_to_chord=0.115,
    x_cm=0.2,
    Q=1.5,
)

# right strut
right_strut_mid_le = geometry.evaluate(right_strut_mid_le_parametric).reshape((-1, 3))
right_strut_mid_te = geometry.evaluate(right_strut_mid_te_parametric).reshape((-1, 3))
right_strut_chord_length = m3l.norm(right_strut_mid_le-right_strut_mid_te)

right_strut_drag_comp = DragComponent(
    component_type='strut',
    wetted_area=surface_area_list[6],
    characteristic_length=right_strut_chord_length,
    thickness_to_chord=0.115,
    x_cm=0.2,
    Q=1.5,
)

# drag_comp_list = [wing_drag_comp, fuselage_drag_comp, h_tail_drag_comp, vtail_drag_comp,
#                   left_jury_drag_comp, right_jury_drag_comp, left_strut_drag_comp, 
#                   right_strut_drag_comp, gearpod_drag_comp]
# drag_comp_list = [wing_drag_comp, fuselage_drag_comp, h_tail_drag_comp, vtail_drag_comp,
#                   left_jury_drag_comp, right_jury_drag_comp, left_strut_drag_comp, 
#                   right_strut_drag_comp]

# drag_comp_list = [wing_drag_comp, h_tail_drag_comp, vtail_drag_comp,
#                   left_jury_drag_comp, right_jury_drag_comp, left_strut_drag_comp, 
#                   right_strut_drag_comp]

drag_comp_list = [wing_drag_comp, h_tail_drag_comp,
                  left_jury_drag_comp, right_jury_drag_comp, left_strut_drag_comp, 
                  right_strut_drag_comp]

S_ref_area = surface_area_list[1] / 1.969576502
print('wing area', S_ref_area.value)
h_tail_area = surface_area_list[2] / 2.1
print('htail area', h_tail_area.value)
v_tail_area = surface_area_list[3] / 2.18
print('v_tail area', v_tail_area.value)
jury_area = surface_area_list[4] / 2.11
print('jury area', jury_area.value)
strut_area = surface_area_list[6] / 2.07
print('strut area', strut_area.value)
# 
# wing_AR_normal_value = ((wing_span.value)**2 )/ S_ref_area.value
# wing_AR_normal_value = ((wing_span)**2 )/ S_ref_area
wing_AR = ((wing_span_dv)**2 )/ S_ref_area
# wing_AR = m3l.Variable(name='wing_AR',shape= (1,1),value=wing_AR_normal_value)
print('wing span', wing_span.value)
print('Wing_AR',wing_AR.value)
# endregion

# ft_to_m = 0.3048

# print(wing_box_beam_mesh.width )
# print(left_strut_box_beam_mesh.width * ft_to_m)
# print(right_strut_box_beam_mesh.width * ft_to_m)