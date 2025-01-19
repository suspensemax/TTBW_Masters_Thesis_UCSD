
from python_csdl_backend import Simulator
from lsdo_airfoil.core.airfoil_model_csdl import AirfoilModelCSDL
from lsdo_airfoil.core.pressure_profile import PressureProfile
import csdl
import numpy as np
from lsdo_airfoil.utils.compute_b_spline_mat import compute_b_spline_mat
from lsdo_airfoil import AIRFOIL_COORDINATES_FOLDER
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['text.usetex'] = False



num_nodes = 1
M = 0.2
AoA = 2 #np.linspace(-8, 15, num_nodes) 
Cl = 0.5
Re =  4e6

pressure_profile = PressureProfile(
    airfoil_name='NASA_langley_ga_1',
    num_nodes=num_nodes,
    use_inverse_cl_map=True,
)

run_model = pressure_profile.compute()
run_model.create_input('mach_number', M * np.ones((num_nodes, )))
run_model.create_input('angle_of_attack', Cl)
run_model.create_input('reynolds_number', Re * np.ones((num_nodes, )))

run_model.add_design_variable('mach_number', lower=0., upper=0.6)
# run_model.add_design_variable('angle_of_attack', scaler=3e-2, lower=2, upper=10)
run_model.add_design_variable('reynolds_number', scaler=0.5e-6, lower=1e5, upper=1e6)

# cp_upper = run_model.declare_variable('CpUpper', shape=(100, ))
# run_model.print_var(cp_upper * 1)


sim = Simulator(run_model, analytics=True)
sim.run()


Cp_upper = sim['CpUpper']
Cp_lower = sim['CpLower']

print(sim['angle_of_attack'])
print(sim['lift_coefficient'])
print(sim['Cl'])
print(sim['Cd'])
print(Cp_upper.shape)

num_pts = 100
x_range = np.linspace(0, 1, num_pts)
i_vec = np.arange(0, len(x_range))
x_interp = 1 - np.cos(np.pi/(2 * len(x_range)) * i_vec)

plt.plot(x_interp, Cp_upper)
plt.plot(x_interp, Cp_lower)
plt.gca().invert_yaxis()


plt.show()

