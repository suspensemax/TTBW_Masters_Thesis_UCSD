import m3l
import numpy as np
from dataclasses import dataclass, field
import lsdo_geo as lg
from scipy.interpolate import interp1d
from typing import List, Union
from lsdo_rotor import RotorMeshes




@dataclass
class BladeParameters:
    """
    Data class for specifying blade geometric paramters
    """
    blade_component : lg.BSplineSubSet
    point_on_leading_edge : np.ndarray
    num_spanwise_vlm : int = None
    num_chordwise_vlm : int = None


@dataclass
class SimpleBoxBeamMesh:
    """
    Data class for box beam meshes
    """
    height : m3l.Variable
    width : m3l.Variable
    beam_nodes : m3l.Variable

    _top_parametric : list = None,
    _bot_parametric : list = None
    _te_parametric : list = None
    _le_parametric : list = None

    _num_beam_nodes : int = None
    _node_center : list = None

    def update(self, geometry):
        """
        Re-evaluate beam nodes, and height/width at beam nodes (all as m3l variable).
        This method should be called only if there's FFD that changes the geometry

        Return : SimpleBoxBeamMesh
            An updated version of self (meaning updated data class)
        """
        num_beam_nodes = self._num_beam_nodes
        node_center = self._node_center

        te = geometry.evaluate(self._te_parametric).reshape((-1, 3))
        le = geometry.evaluate(self._le_parametric).reshape((-1, 3))

        beam_nodes = m3l.linear_combination(le, te, 1, start_weights=np.ones((num_beam_nodes, ))*(1-node_center), stop_weights=np.ones((num_beam_nodes, ))*node_center).reshape((num_beam_nodes, 3))
        width = m3l.norm((le - te)*0.5)
        offset = np.array([0,0,0.5])

        top = geometry.evaluate(self._top_parametric).reshape((num_beam_nodes, 3))
        bot = geometry.evaluate(self._bot_parametric).reshape((num_beam_nodes, 3))
        height = m3l.norm((top - bot)*1)

        self.height = height
        self.width = width
        self.beam_nodes = beam_nodes

        return self



@ dataclass
class LiftingSurfaceMeshes:
    """Data class for wing meshes"""
    vlm_mesh : m3l.Variable
    oml_mesh : m3l.Variable
    ml_mesh : m3l.Variable = None

    _upper_surface_parametric : list = None
    _lower_surface_parametric : list = None
    
    def update(self, geometry):
        """Evaluates the camber mesh (e.g., after FFD)"""

        if len(self.vlm_mesh.shape) == 4:
            num_chordwise = self.vlm_mesh.shape[1]
            num_spanwise = self.vlm_mesh.shape[2]
        elif len(self.vlm_mesh.shape) == 3:
            num_chordwise = self.vlm_mesh.shape[0]
            num_spanwise = self.vlm_mesh.shape[1]
        else:
            print(self.vlm_mesh.shape)
            raise NotImplementedError
        
        upper_surface_wireframe = geometry.evaluate(self._upper_surface_parametric).reshape((num_chordwise, num_spanwise, 3))
        lower_surface_wireframe = geometry.evaluate(self._lower_surface_parametric).reshape((num_chordwise, num_spanwise, 3))
        wing_camber_surface = m3l.linspace(upper_surface_wireframe, lower_surface_wireframe, 1)

        wing_camber_surface.description = self.vlm_mesh.description

        self.vlm_mesh = wing_camber_surface

        return self

def make_rotor_mesh(
        geometry : lg.Geometry,
        num_radial : int, 
        disk_component : lg.BSplineSubSet,
        origin : np.ndarray, 
        y1 : np.ndarray,
        y2 : np.ndarray,
        z1 : np.ndarray,
        z2 : np.ndarray,
        blade_geometry_parameters : List[BladeParameters] = [],
        num_tangential : int = 30,
        norm_hub_radius : float = 0.2,
        create_disk_mesh = False,
        plot : bool = False,
        grid_search_density_parameter : int = 25,
        boom_is_thrust_origin : lg.BSplineSubSet=None,
        radius : m3l.Variable=None,
) -> RotorMeshes: 
    """
    Helper function to automate generation of rotor meshes
    """
    # Disk origin
    if boom_is_thrust_origin is not None:
        disk_origin_parametric = boom_is_thrust_origin.project(origin, grid_search_density_parameter=grid_search_density_parameter, plot=plot)
        disk_origin = geometry.evaluate(disk_origin_parametric)
    else:
        disk_origin_parametric = disk_component.project(origin, grid_search_density_parameter=grid_search_density_parameter, plot=plot)
        disk_origin = geometry.evaluate(disk_origin_parametric)
    
    # In-plane 1

    y11_parametric = disk_component.project(y1, plot=plot)
    y11 = geometry.evaluate(y11_parametric)
    y12_parametric = disk_component.project(y2, plot=plot)
    y12 = geometry.evaluate(y12_parametric)
    disk_in_plane_y = y11 - y12

    # In-plane 2
    y21_parametric = disk_component.project(z1, plot=plot) 
    y21 = geometry.evaluate(y21_parametric)
    y22_parametric = disk_component.project(z2, plot=plot)
    y22 = geometry.evaluate(y22_parametric)
    disk_in_plane_x = y21 - y22

    if radius is None:
        rotor_radius = m3l.norm(disk_in_plane_y / 2)
        # rotor_radius = m3l.norm((y12 - y11)/2)
        # rotor_radius = m3l.norm(y12 - y11) / 2IM
        
        rotor_radius_2 = m3l.norm(disk_in_plane_x / 2)
        # rotor_radius_2 = m3l.norm((y22 - y21)/2)
        # rotor_radius_2 = m3l.norm(y22 - y21) / 2
    
    else:
        rotor_radius = radius
        rotor_radius_2 = radius

    thrust_vector = m3l.cross(disk_in_plane_x, disk_in_plane_y)
    thrust_unit_vector = thrust_vector / m3l.norm(thrust_vector)

    rotor_mesh = RotorMeshes(
        thrust_origin=disk_origin,
        thrust_vector=thrust_unit_vector,
        radius=rotor_radius,
        in_plane_1=disk_in_plane_x,
        in_plane_2=disk_in_plane_y,

        _disk_origin_parametric=disk_origin_parametric,
        _thrust_origin_prametric=disk_origin_parametric,
        _y11_parametric=y11_parametric,
        _y12_parametric=y12_parametric,
        _y21_parametric=y21_parametric,
        _y22_parametric=y22_parametric,
        _radius_2=rotor_radius_2,
    )

    if create_disk_mesh:
        v1 = disk_in_plane_y / m3l.norm(disk_in_plane_y)
        v2 = disk_in_plane_x / m3l.norm(disk_in_plane_x)
        p = disk_origin
        
        if num_radial %2 != 0:
            raise ValueError(f"Odd number 'num_radial' not yet implemented. Must be an even number for now")
        
        radii = np.linspace(norm_hub_radius * rotor_radius.value, rotor_radius.value, num_radial)
        angles = np.linspace(0, 2*np.pi, num_tangential)
        
        cartesian = np.zeros((num_radial, num_tangential, 3))
        for i in range(num_radial):
            for j in range(num_tangential):
                # Equation of a circle (circular plane) in 3D space that does not have to be aligned with any 2 cartesian axes
                cartesian[i, j, :] = p.value + radii[i] * np.cos(angles[j]) * v1.value + radii[i] * np.sin(angles[j]) * v2.value
        
        disk_mesh_parametric = disk_component.project(cartesian, plot=plot)
        disk_mesh_physical = geometry.evaluate(disk_mesh_parametric)
        rotor_mesh.disk_mesh_physical=disk_mesh_physical
        rotor_mesh.disk_mesh_parametric=disk_mesh_parametric

    if blade_geometry_parameters:
        counter = 0
        for blade in blade_geometry_parameters:
            comp = blade.blade_component
            p_o_le = blade.point_on_leading_edge
            num_spanwise = blade.num_spanwise_vlm
            
            num_chordwise = blade.num_chordwise_vlm

            b_spline_patch = comp.project(p_o_le, plot=plot)[0][0]

            if counter == 0:

                # le_tip = blade.le_tip
                # le_hub = blade.le_hub
                # le_center = blade.le_center
                # te_center = blade.te_center

                # ec = (le_tip - le_hub) / 2 + le_hub
                # u = le_tip - ec
                # v = le_center- ec

                # half_ellipse = np.zeros((num_radial, 3))
                # angles = np.linspace(0, np.pi, num_radial)
                # for i in range(num_radial):
                #     alpha = 2 * angles[i]
                #     theta = 2 * alpha / num_radial
                #     # half_ellipse[i, :] = (ec + np.sin(alpha-theta) * u + np.sin(theta) * v) / np.sin(alpha)
                #     half_ellipse[i, :] = ec + np.cos(angles[i]) * u + np.sin(angles[i]) * v

                # # print(half_ellipse)
                # straight_line = np.linspace(le_hub, le_tip, num_radial)


                linspace_parametric = np.hstack((np.linspace(0, 0.55, int(num_radial/2)), np.linspace(0.7, 1, int(num_radial/2))))
                le_list = []
                te_list = []

                
                for i in range(num_radial):
                    le_list.append((b_spline_patch, np.array([[linspace_parametric[i], 1]])))
                    te_list.append((b_spline_patch, np.array([[linspace_parametric[i], 0]])))

                le = geometry.evaluate(le_list).reshape((-1, 3))
                te = geometry.evaluate(te_list).reshape((-1, 3))
                # le_minus_te = le-te
                le_minus_te = te-le

                normal_exp = m3l.expand(thrust_unit_vector, new_shape=(num_radial, 3), indices='i->ji')
                twist_profile = np.pi/2 - m3l.arccos(m3l.dot(normal_exp, le_minus_te, axis=1)/ m3l.norm(le_minus_te, axes=(1, )))
                chord_profile = m3l.norm(le_minus_te)

                # print(chord_profile)
                # print(twist_profile)

                rotor_mesh.chord_profile = chord_profile
                rotor_mesh.twist_profile = twist_profile

            if num_spanwise is not None:
                if num_spanwise %2 != 0:
                    raise ValueError(f"Odd number for 'num_spanwise' not yet implemented. Must be even number for now")
                linspace_parametric_vlm = np.hstack((np.linspace(0, 0.55, int(num_spanwise/2)), np.linspace(0.7, 1, int(num_spanwise/2))))

                le_list_vlm = []
                te_list_vlm = []
                for i in range(num_spanwise):
                    le_list_vlm.append((b_spline_patch, np.array([[linspace_parametric_vlm[i], 1]])))
                    te_list_vlm.append((b_spline_patch, np.array([[linspace_parametric_vlm[i], 0]])))

                le_vlm = geometry.evaluate(le_list_vlm).reshape((-1, 3))
                te_vlm = geometry.evaluate(te_list_vlm).reshape((-1, 3))

                chord_surface = m3l.linspace(le_vlm, te_vlm, num_chordwise)# .reshape((-1, 3))
                if num_chordwise > 2:
                    upper_surface = geometry.evaluate(comp.project(chord_surface.value + thrust_vector.value, direction=thrust_unit_vector.value, plot=plot)).reshape((num_chordwise, num_spanwise, 3))
                    lower_surface = geometry.evaluate(comp.project(chord_surface.value - thrust_vector.value, direction=-1 * thrust_unit_vector.value, plot=plot)).reshape((num_chordwise, num_spanwise, 3))
                    camber_surface = m3l.linspace(upper_surface, lower_surface, 1)
                else: 
                    camber_surface = chord_surface.reshape((num_chordwise, num_spanwise, 3))

                if plot:
                    # geometry.plot_meshes(meshes=camber_surface, mesh_plot_types=['wireframe'], mesh_opacity=1., mesh_color='#F5F0E6')
                    geometry.plot_meshes(meshes=camber_surface, mesh_plot_types=['wireframe'], mesh_opacity=1., mesh_color='#C8102E')


                rotor_mesh.vlm_meshes.append(camber_surface)

            counter += 1


            

    return rotor_mesh


def make_vlm_camber_mesh(
        geometry : lg.Geometry,
        wing_component : lg.BSplineSubSet,
        num_spanwise : int,
        num_chordwise : int, 
        le_right : np.ndarray,
        le_left : np.ndarray,
        te_right : np.ndarray,
        te_left : np.ndarray,
        le_center : np.ndarray = None,
        te_center : np.ndarray = None,
        plot: bool=False,
        off_set_x: Union[int, float]=None, 
        grid_search_density_parameter : int = 50,
        le_interp : str = 'ellipse',
        te_interp : str = 'ellipse',
        parametric_mesh_grid_num : int = 20,
        orientation : str = 'horizontal',
        bunching_cos : bool=False,
        actuation_axis : List[np.ndarray] = [],
        actuation_angle : Union[m3l.Variable, float, int] = None,
        mirror : bool = False,
        zero_y : bool = False,
    
) -> LiftingSurfaceMeshes: 
    """
    Helper function to create a VLM camber mesh
    """
    if orientation not in ['vertical', 'horizontal']:
        raise ValueError(f"Uknown keyword argument {orientation}. Acceptable arguments are 'horizontal', and 'verical'")

    if le_center is not None:
        x = np.array([le_left[0], le_center[0], le_right[0]])
        y = np.array([le_left[1], le_center[1], le_right[1]])
        z = np.array([le_left[2], le_center[2], le_right[2]])
        fz = interp1d(y, z, kind='linear')
        array_to_project = np.zeros((num_spanwise, 3))
        
        interp_y = np.linspace(y[0], y[2], num_spanwise)
        
        if le_interp == 'ellipse':
            # Parameters for ellipse
            h = le_right[0]
            b = h - le_center[0]
            a = le_right[1]

            array_to_project[:, 0] = -(b**2 * (1 - interp_y**2/a**2))**0.5 + h
            array_to_project[:, 1] = interp_y
            array_to_project[:, 2] = fz(interp_y)

        elif le_interp == 'linear':
            fx = interp1d(y, x, kind='linear')
            array_to_project[:, 0] = fx(interp_y)
            array_to_project[:, 1] = interp_y
            array_to_project[:, 2] = fz(interp_y)

        elif le_interp == 'quadratic':
            fx = interp1d(y, x, kind='quadratic')
            array_to_project[:, 0] = fx(interp_y)
            array_to_project[:, 1] = interp_y
            array_to_project[:, 2] = fz(interp_y)

        elif le_interp not in ['ellipse', 'linear', 'quadratic']:
            raise Exception(f"Unknown interpolation type '{le_interp}'. Available options are 'ellipse', 'linear', 'quadratic'.")
        else:
            raise NotImplementedError
        
        if off_set_x:
            array_to_project[:, 0] -= off_set_x

        le = geometry.evaluate(wing_component.project(array_to_project, plot=plot, grid_search_density_parameter=grid_search_density_parameter)).reshape((-1, 3))
    
    else:
        le = geometry.evaluate(wing_component.project(np.linspace(le_left, le_right, num_spanwise), plot=plot, grid_search_density_parameter=grid_search_density_parameter)).reshape((-1, 3))


    if te_center is not None:
        x = np.array([te_left[0], te_center[0], te_right[0]])
        y = np.array([te_left[1], te_center[1], te_right[1]])
        z = np.array([te_left[2], te_center[2], te_right[2]])
        fz = interp1d(y, z, kind='linear')
        array_to_project = np.zeros((num_spanwise, 3))
        interp_y = np.linspace(y[0], y[2], num_spanwise)

        if te_interp == 'ellipse':
            # Parameters for ellipse
            h = te_right[0]
            b = h - te_center[0]
            a = te_right[1]

            array_to_project[:, 0] = (b**2 * (1 - interp_y**2/a**2))**0.5 + h
            array_to_project[:, 1] = interp_y
            array_to_project[:, 2] = fz(interp_y)

        elif te_interp == 'linear':
            fx = interp1d(y, x, kind='linear')
            array_to_project[:, 0] = fx(interp_y)
            array_to_project[:, 1] = interp_y
            array_to_project[:, 2] = fz(interp_y)

        elif te_interp == 'quadratic':
            fx = interp1d(y, x, kind='quadratic')
            array_to_project[:, 0] = fx(interp_y)
            array_to_project[:, 1] = interp_y
            array_to_project[:, 2] = fz(interp_y)

        elif le_interp not in ['ellipse', 'linear', 'quadratic']:
            raise Exception(f"Unknown interpolation type '{le_interp}'. Available options are 'ellipse', 'linear', 'quadratic'.")
        else:
            raise NotImplementedError
        
        if off_set_x:
            array_to_project[:, 0] += off_set_x

        te = geometry.evaluate(wing_component.project(array_to_project, plot=plot, grid_search_density_parameter=grid_search_density_parameter))#.reshape((-1, 3))
    
    else:
        te = geometry.evaluate(wing_component.project(np.linspace(te_left, te_right, num_spanwise), plot=plot, grid_search_density_parameter=grid_search_density_parameter))#.reshape((-1, 3))

    wing_chord_surface = m3l.linspace(le, te, num_chordwise)
    if plot:
        # geometry.plot_meshes(wing_chord_surface)
        # geometry.plot_meshes(meshes=wing_chord_surface, mesh_plot_types=['wireframe'], mesh_opacity=1., mesh_color='#FF6F61')
        geometry.plot_meshes(meshes=wing_chord_surface, mesh_plot_types=['wireframe'], mesh_opacity=1., mesh_color='#003366')


    if bunching_cos:
        chord_surface_ml = wing_chord_surface
        i_vec = np.arange(0, len(chord_surface_ml.value))
        x_range = np.linspace(0, 1, num_chordwise)

        half_cos =  1-np.cos(i_vec * np.pi/(2 * (len(x_range)-1)))
        x_interp_x = wing_chord_surface.value[0,:, 0].reshape(num_spanwise, 1) - ((chord_surface_ml.value[0, :, 0] - chord_surface_ml.value[-1, :, 0]).reshape(num_spanwise, 1) * half_cos.reshape(1,num_chordwise))
        x_interp_y = wing_chord_surface.value[0,:, 1].reshape(num_spanwise, 1) - ((chord_surface_ml.value[0, :, 1] - chord_surface_ml.value[-1, :, 1]).reshape(num_spanwise, 1) * half_cos.reshape(1,num_chordwise))
        x_interp_z = wing_chord_surface.value[0,:, 2].reshape(num_spanwise, 1) - ((chord_surface_ml.value[0, :, 2] - chord_surface_ml.value[-1, :, 2]).reshape(num_spanwise, 1) * half_cos.reshape(1,num_chordwise))

        new_chord_surface = np.zeros((num_chordwise, num_spanwise, 3))
        new_chord_surface[:, :, 0] = x_interp_x.T
        new_chord_surface[:, :, 1] = x_interp_y.T
        new_chord_surface[:, :, 2] = x_interp_z.T

        if orientation == 'horizontal':
            wing_upper_surface_wireframe_parametric = wing_component.project(new_chord_surface + np.array([0., 0., 0.5]), direction=np.array([0., 0., -1.]), grid_search_density_parameter=25, plot=plot)
            wing_lower_surface_wireframe_parametric = wing_component.project(new_chord_surface - np.array([0., 0., 0.5]), direction=np.array([0., 0., 1.]), grid_search_density_parameter=35, plot=plot)
        elif orientation == 'vertical':
            raise NotImplementedError

    else:
        if orientation == 'horizontal':
            wing_upper_surface_wireframe_parametric = wing_component.project(wing_chord_surface.value + np.array([0., 0., 1.5]), direction=np.array([0., 0., -1.]), grid_search_density_parameter=30, plot=plot)
            wing_lower_surface_wireframe_parametric = wing_component.project(wing_chord_surface.value - np.array([0., 0., 0.9]), direction=np.array([0., 0., 1.]), grid_search_density_parameter=30, plot=plot)
        elif orientation == 'vertical':
            wing_upper_surface_wireframe_parametric = wing_component.project(wing_chord_surface.value + np.array([0., 1., 0.]), direction=np.array([0., 1., 0.]), grid_search_density_parameter=26, plot=plot)
            wing_lower_surface_wireframe_parametric = wing_component.project(wing_chord_surface.value - np.array([0., 1., 0.]), direction=np.array([0., -1., 0.]), grid_search_density_parameter=26, plot=plot)

    # actuated_geometry = geometry.rotate(...)
   
    wing_upper_surface_wireframe = geometry.evaluate(wing_upper_surface_wireframe_parametric).reshape((num_chordwise, num_spanwise, 3))
    wing_lower_surface_wireframe = geometry.evaluate(wing_lower_surface_wireframe_parametric).reshape((num_chordwise, num_spanwise, 3))

    wing_camber_surface = m3l.linspace(wing_upper_surface_wireframe, wing_lower_surface_wireframe, 1)#.reshape((-1, 3))
    if mirror:
        wing_camber_surface.description = 'mirror'

    if zero_y:
        wing_camber_surface.description = 'zero_y'

    if plot:
        # geometry.plot_meshes(meshes=wing_camber_surface, mesh_plot_types=['wireframe'], mesh_opacity=1., mesh_color='#F5F0E6')
        geometry.plot_meshes(meshes=wing_camber_surface, mesh_plot_types=['wireframe'], mesh_opacity=1., mesh_color='#FF6F61')

    # Actuations
    if not actuation_axis and (actuation_angle is None):
        pass
    elif actuation_axis and (actuation_angle is None):
        raise ValueError("Specified 'actuation_axis' but no 'actuation_angle'")
    elif (actuation_angle is not None) and (len(actuation_axis) == 0):
        raise ValueError(("Specified 'actuation_angle' but not 'actuation_axis'"))
    else:
        if len(actuation_axis) != 2:
            raise ValueError(f"'actuation_axis' must be a list of length 2 (containing two vectors)")
        
        axis_origin = actuation_axis[0]
        axis_vector = actuation_axis[1] #- axis_origin

        # wing_component.rotate(axis_origin=axis_origin, axis_vector=axis_vector, angles=actuation_angle, units='degrees')
        geometry.rotate(axis_origin=axis_origin, axis_vector=axis_vector, angles=actuation_angle, units='degrees', b_splines=wing_component.b_spline_names)
        if plot:
            geometry.plot()

    # OML parametric mesh
    surfaces = wing_component.b_spline_names
    oml_para_mesh = []
    for name in surfaces:
        for u in np.linspace(0,1,parametric_mesh_grid_num):
            for v in np.linspace(0,1,parametric_mesh_grid_num):
                oml_para_mesh.append((name, np.array([u,v]).reshape((1,2))))

    oml_mesh = geometry.evaluate(oml_para_mesh).reshape((-1, 3))
    # print(oml_mesh.shape)
    # 
    meshes = LiftingSurfaceMeshes(
        vlm_mesh=wing_camber_surface,
        oml_mesh=oml_mesh,
        _upper_surface_parametric=wing_upper_surface_wireframe_parametric,
        _lower_surface_parametric=wing_lower_surface_wireframe_parametric,
    )


    return meshes


def make_1d_box_beam_mesh(
        geometry : lg.Geometry,
        wing_component : lg.BSplineSubSet,
        num_beam_nodes : int,
        le_right : np.ndarray,
        le_left : np.ndarray,
        te_right : np.ndarray,
        te_left : np.ndarray,
        beam_width : float,
        node_center : float,
        front_spar_location : float, 
        rear_spar_location : float, 
        le_center : np.ndarray = None,
        te_center : np.ndarray = None,
        plot: bool=False,
        grid_search_density_parameter : int = 50,
        le_interp : str = 'ellipse',
        te_interp : str = 'ellipse',
        horizontal : str = 'yes',
) -> SimpleBoxBeamMesh: 
    """
    Helper function to create a simple 1-D box beam mesh with the beam node being 
    centered in the middle of the wing box       
    """

    if num_beam_nodes %2 == 0:
        raise ValueError("Number of beam nodes should be odd such that there is always a node at the center (of the fuselage)")

    if le_center is not None:
        x = np.array([le_left[0], le_center[0], le_right[0]])
        y = np.array([le_left[1], le_center[1], le_right[1]])
        z = np.array([le_left[2], le_center[2], le_right[2]])
        fz = interp1d(y, z, kind='linear')
        array_to_project = np.zeros((num_beam_nodes, 3))
        interp_y = np.linspace(y[0], y[2], num_beam_nodes)
        
        if le_interp == 'ellipse':
            # Parameters for ellipse
            h = le_right[0]
            b = h - le_center[0]
            a = le_right[1]

            array_to_project[:, 0] = -(b**2 * (1 - interp_y**2/a**2))**0.5 + h
            array_to_project[:, 1] = interp_y
            array_to_project[:, 2] = fz(interp_y)

        elif le_interp == 'linear':
            fx = interp1d(y, x, kind='linear')
            array_to_project[:, 0] = fx(interp_y)
            array_to_project[:, 1] = interp_y
            array_to_project[:, 2] = fz(interp_y)

        elif le_interp == 'quadratic':
            fx = interp1d(y, x, kind='quadratic')
            array_to_project[:, 0] = fx(interp_y)
            array_to_project[:, 1] = interp_y
            array_to_project[:, 2] = fz(interp_y)

        elif le_interp not in ['ellipse', 'linear', 'quadratic']:
            raise Exception(f"Unknown interpolation type '{le_interp}'. Available options are 'ellipse', 'linear', 'quadratic'.")
        else:
            raise NotImplementedError
        
        le_parametric = wing_component.project(array_to_project, plot=plot, grid_search_density_parameter=grid_search_density_parameter)
        le = geometry.evaluate(le_parametric).reshape((-1, 3))
    
    else:
        le_parametric =  wing_component.project(np.linspace(le_left, le_right, num_beam_nodes), plot=plot, grid_search_density_parameter=grid_search_density_parameter)
        le = geometry.evaluate(le_parametric).reshape((-1, 3))


    if te_center is not None:
        x = np.array([te_left[0], te_center[0], te_right[0]])
        y = np.array([te_left[1], te_center[1], te_right[1]])
        z = np.array([te_left[2], te_center[2], te_right[2]])
        fz = interp1d(y, z, kind='linear')
        array_to_project = np.zeros((num_beam_nodes, 3))
        interp_y = np.linspace(y[0], y[2], num_beam_nodes)

        if te_interp == 'ellipse':
            # Parameters for ellipse
            h = te_right[0]
            b = h - te_center[0]
            a = te_right[1]

            array_to_project[:, 0] = (b**2 * (1 - interp_y**2/a**2))**0.5 + h
            array_to_project[:, 1] = interp_y
            array_to_project[:, 2] = fz(interp_y)

        elif te_interp == 'linear':
            fx = interp1d(y, x, kind='linear')
            array_to_project[:, 0] = fx(interp_y)
            array_to_project[:, 1] = interp_y
            array_to_project[:, 2] = fz(interp_y)

        elif te_interp == 'quadratic':
            fx = interp1d(y, x, kind='quadratic')
            array_to_project[:, 0] = fx(interp_y)
            array_to_project[:, 1] = interp_y
            array_to_project[:, 2] = fz(interp_y)

        elif le_interp not in ['ellipse', 'linear', 'quadratic']:
            raise Exception(f"Unknown interpolation type '{le_interp}'. Available options are 'ellipse', 'linear', 'quadratic'.")
        else:
            raise NotImplementedError
        
        te_parametric = wing_component.project(array_to_project, plot=plot, grid_search_density_parameter=grid_search_density_parameter)
        te = geometry.evaluate(te_parametric).reshape((-1, 3))
    
    else:
        te_parametric =  wing_component.project(np.linspace(te_left, te_right, num_beam_nodes), plot=plot, grid_search_density_parameter=grid_search_density_parameter)
        te = geometry.evaluate(te_parametric).reshape((-1, 3))

    beam_nodes = m3l.linear_combination(le, te, 1, start_weights=np.ones((num_beam_nodes, ))*(1-node_center), stop_weights=np.ones((num_beam_nodes, ))*node_center).reshape((num_beam_nodes, 3))
    # beam_nodes = m3l.linear_combination(le, te, 1, start_weights=np.ones((num_beam_nodes, ))*(1-node_center), stop_weights=np.ones((num_beam_nodes, ))*node_center)
    # front_spar = m3l.linear_combination(le, te, 1, start_weights=np.ones((num_beam_nodes, ))*(1-front_spar_location), stop_weights=np.ones((num_beam_nodes, ))*front_spar_location).reshape((num_beam_nodes, 3))
    front_spar = m3l.linear_combination(le, te, 1, start_weights=np.ones((num_beam_nodes, ))*(1-front_spar_location), stop_weights=np.ones((num_beam_nodes, ))*front_spar_location)
    # rear_spar = m3l.linear_combination(le, te, 1, start_weights=np.ones((num_beam_nodes, ))*(1-rear_spar_location), stop_weights=np.ones((num_beam_nodes, ))*rear_spar_location).reshape((num_beam_nodes, 3))
    rear_spar = m3l.linear_combination(le, te, 1, start_weights=np.ones((num_beam_nodes, ))*(1-rear_spar_location), stop_weights=np.ones((num_beam_nodes, ))*rear_spar_location)
    # height_location = m3l.linear_combination(le, te, 1, start_weights=np.ones((num_beam_nodes, ))*(1-front_spar_location), stop_weights=np.ones((num_beam_nodes, ))*front_spar_location).reshape((num_beam_nodes, 3))
    height_location = m3l.linear_combination(le, te, 1, start_weights=np.ones((num_beam_nodes, ))*(1-front_spar_location), stop_weights=np.ones((num_beam_nodes, ))*front_spar_location)

    # width = m3l.norm((le - te)*0.5)
    # width = m3l.norm((le - te)*1.0)
    width = m3l.norm((front_spar.reshape((-1, 3)) - rear_spar.reshape((-1, 3))) * 1.5)
    # width = m3l.norm((front_spar.reshape((-1, 3)) - rear_spar.reshape((-1, 3))) * 1.0)
    # width = m3l.norm((front_spar - rear_spar) * 1)
    offset = np.array([0,0,0.5])
    # top_parametric = wing_component.project(beam_nodes.value+offset, direction=np.array([0., 0., -1.]), plot=plot)
    # bot_parametric = wing_component.project(beam_nodes.value-offset, direction=np.array([0., 0., 1.]), plot=plot)

    if horizontal == 'yes':
        # top_parametric = wing_component.project(height_location.value+offset, direction=np.array([0., 0., -1.]), plot=plot)
        # bot_parametric = wing_component.project(height_location.value-offset, direction=np.array([0., 0., 1.]), plot=plot)
        top_parametric = wing_component.project(beam_nodes.value+offset, direction=np.array([0., 0., -1.]), plot=plot)
        bot_parametric = wing_component.project(beam_nodes.value-offset, direction=np.array([0., 0., 1.]), plot=plot)

    if horizontal == 'no':
        # top_parametric = wing_component.project(height_location.value+offset, direction=np.array([0., 1., 0.]), plot=plot)
        # bot_parametric = wing_component.project(height_location.value-offset, direction=np.array([0., -1., 0.]), plot=plot)
        top_parametric = wing_component.project(beam_nodes.value+offset, direction=np.array([0., 1., 0.]), plot=plot)
        bot_parametric = wing_component.project(beam_nodes.value-offset, direction=np.array([0., -1., 0.]), plot=plot)


    # top = geometry.evaluate(top_parametric).reshape((num_beam_nodes, 3))
    # bot = geometry.evaluate(bot_parametric).reshape((num_beam_nodes, 3))

    top = geometry.evaluate(top_parametric).reshape((-1, 3))
    bot = geometry.evaluate(bot_parametric).reshape((-1, 3))

    height = m3l.norm((top - bot)*1.5)
    # height = m3l.norm((top - bot)*1.0)

    if plot:
        # geometry.plot_meshes(meshes=beam_nodes)# , mesh_plot_types=['surface'], mesh_opacity=1., mesh_color='#F5F0E6')
        geometry.plot_meshes(meshes=beam_nodes,mesh_color='#FFD700')

    box_beam_mesh = SimpleBoxBeamMesh(
        height=height,
        width=width,
        beam_nodes=beam_nodes,

        _top_parametric=top_parametric,
        _bot_parametric=bot_parametric,
        _te_parametric=te_parametric,
        _le_parametric=le_parametric,

        _num_beam_nodes=num_beam_nodes,
        _node_center=node_center

    )

    return box_beam_mesh
    


def compute_component_surface_area(
        component_list : List[lg.BSplineSubSet],
        geometry : lg.Geometry, # As long as this commes after FFD geometry changes should be reflected
        parametric_mesh_grid_num : int = 20,
        plot : bool = False,
) -> m3l.Variable:
    """
    Helper function to compute the surface area of a component
    """
     # OML parametric mesh
    surface_area_list = []

    for i in range(len(component_list)):
        component = component_list[i]
        surfaces = component.b_spline_names
        oml_meshes = []
        surface_area = m3l.Variable(value=0, shape=(1, ))
        for name in surfaces:
            oml_para_mesh = []
            for u in np.linspace(0, 1, parametric_mesh_grid_num):
                for v in np.linspace(0, 1, parametric_mesh_grid_num):
                    oml_para_mesh.append((name, np.array([u,v]).reshape((1,2))))
            
            coords_m3l_vec = geometry.evaluate(oml_para_mesh)
            coords_m3l = coords_m3l_vec.reshape((parametric_mesh_grid_num, parametric_mesh_grid_num, -1))

            indices = np.arange(parametric_mesh_grid_num**2 * 3).reshape((parametric_mesh_grid_num, parametric_mesh_grid_num, 3))
            u_end_indices = indices[1:, :, :].flatten()
            u_start_indices = indices[:-1, :, :].flatten()

            v_end_indices = indices[:, 1:, :].flatten()
            v_start_indices = indices[:, :-1, :].flatten()
            
            coords_u_end = coords_m3l_vec[u_end_indices].reshape((parametric_mesh_grid_num-1, parametric_mesh_grid_num, 3))
            coords_u_start = coords_m3l_vec[u_start_indices].reshape((parametric_mesh_grid_num-1, parametric_mesh_grid_num, 3))

            coords_v_end = coords_m3l_vec[v_end_indices].reshape((parametric_mesh_grid_num-1, parametric_mesh_grid_num, 3))
            coords_v_start = coords_m3l_vec[v_start_indices].reshape((parametric_mesh_grid_num-1, parametric_mesh_grid_num, 3))


            indices = np.arange(parametric_mesh_grid_num* (parametric_mesh_grid_num-1)  * 3).reshape((parametric_mesh_grid_num-1, parametric_mesh_grid_num, 3))
            v_start_indices = indices[:, :-1, :].flatten()
            v_end_indices = indices[:, 1:, :].flatten()
            u_vectors = coords_u_end - coords_u_start
            u_vectors_start = u_vectors.reshape((-1, ))
            u_vectors_1 = u_vectors_start[v_start_indices].reshape((parametric_mesh_grid_num-1, parametric_mesh_grid_num-1, 3))
            u_vectors_2 = u_vectors_start[v_end_indices].reshape((parametric_mesh_grid_num-1, parametric_mesh_grid_num-1, 3))


            indices = np.arange(parametric_mesh_grid_num*(parametric_mesh_grid_num-1) * 3).reshape((parametric_mesh_grid_num, parametric_mesh_grid_num-1, 3))
            u_start_indices = indices[:-1, :, :].flatten()
            u_end_indices = indices[1:, :, :].flatten()
            v_vectors = coords_v_end - coords_v_start
            v_vectors_start = v_vectors.reshape((-1, ))
            v_vectors_1 = v_vectors_start[u_start_indices].reshape((parametric_mesh_grid_num-1, parametric_mesh_grid_num-1, 3))
            v_vectors_2 = v_vectors_start[u_end_indices].reshape((parametric_mesh_grid_num-1, parametric_mesh_grid_num-1, 3))

            area_vectors_left_lower = m3l.cross(u_vectors_1, v_vectors_2, axis=2)
            area_vectors_right_upper = m3l.cross(v_vectors_1, u_vectors_2, axis=2)
            area_magnitudes_left_lower = m3l.norm(area_vectors_left_lower, order=2, axes=(-1, ))
            area_magnitudes_right_upper = m3l.norm(area_vectors_right_upper, order=2, axes=(-1, ))
            area_magnitudes = (area_magnitudes_left_lower + area_magnitudes_right_upper)/2
            wireframe_area = m3l.sum(area_magnitudes, axes=(0, 1)).reshape((1, ))
            surface_area =  surface_area + wireframe_area 
            

            # x_vectors = coords_m3l[1:, :, :] - coords_m3l[:-1, :, :]
            # print(x_vectors.value)
            oml_meshes.append(coords_m3l)
        
        
        surface_area_list.append(surface_area)
        # oml_mesh = geometry.evaluate(oml_para_mesh).reshape((-1, 3))
        if plot:
            geometry.plot_meshes(meshes=oml_meshes)

    return surface_area_list
    # print(surface_area)
    # 

