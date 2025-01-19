from python_csdl_backend import Simulator
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['text.usetex'] = False
from lsdo_airfoil import AIRFOIL_COORDINATES_FOLDER
from lsdo_airfoil import CONTROL_POINTS_FOLDER
from lsdo_airfoil.core.sml_foil import SMLAirfoil


plot_airfoil = False
airfoil_name = 'boeing_vertol_vr_12'

# Set flow conditions
mach = 0.3
reynolds = 2e6
aoa = 2
control_points = np.load(CONTROL_POINTS_FOLDER / f'{airfoil_name}.npy')

if plot_airfoil:
    test_airfoil_coord = np.loadtxt(AIRFOIL_COORDINATES_FOLDER / f'{airfoil_name}.txt')
    upper = test_airfoil_coord[0:43, 1]
    lower = test_airfoil_coord[43:, 1]
    x_upper = test_airfoil_coord[0:43, 0]
    x_lower = test_airfoil_coord[43:, 0]

    # Interpolate the airfoil
    x_range = np.linspace(0,1,301)
    i_vec = np.arange(0, len(x_range))
    x_interp = 1 - np.cos(np.pi/(2 * (len(x_range)-1)) * i_vec)
    upper_interp = np.interp(x_interp, x_upper, upper)
    lower_interp = np.interp(x_interp, x_lower, lower)

    # plot airfoil
    plt.figure(1)
    plt.scatter(x_upper, upper, color='r', label='Raw coordinates')
    plt.scatter(x_lower, lower, color='r')

    plt.plot(x_interp, upper_interp, color='b', label='Interpolated coordinates')
    plt.plot(x_interp, lower_interp, color='b')
    plt.axis('equal')
    plt.legend()
    plt.show()


# Create an instance of the subsonic machine learning airfoil model- This is a subclass of m3l.Explicit operation
airfoil_ml = SMLAirfoil()

# Create the corrsesponding csdl model
run_model = airfoil_ml.compute()

# Create the csdl model inputs 
run_model.create_input('mach_number', mach) # mach number 
run_model.create_input('angle_of_attack', aoa) # angle of attack    
run_model.create_input('reynolds_number', reynolds) # reynolds number 
run_model.create_input('control_points', control_points) # B-spline control points (camber/thickness)

# Create a Simulator instance and run the model
sim = Simulator(run_model, analytics=True)
sim.run()

# Access the variables of interest (separate outputs for upper and lower airfoil curve)
# Aerodynamic coefficients
Cl = sim['cl_model.Cl']
Cd = sim['cd_model.Cd']
Cm = sim['cm_model.Cm']

# Pressure coefficient
Cp_upper = sim['cp_model.cp_upper']
Cp_lower = sim['cp_model.cp_lower']

# Displacement thickness
delta_star_upper = sim['dstar_model.dstar_upper']
delta_star_lower = sim['dstar_model.dstar_lower']

# Momentum thickness
theta_upper = sim['theta_model.theta_upper']
theta_lower = sim['theta_model.theta_lower']

# Edge velocity
edge_vel_upper = sim['edge_velocity_model.edge_velocity_upper']
edge_vel_lower = sim['edge_velocity_model.edge_velocity_lower']

# Plot the outputs 
from lsdo_airfoil.utils.plot_distributions import plot_distributions
plot_distributions(sim)