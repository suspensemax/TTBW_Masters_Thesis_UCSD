
from python_csdl_backend import Simulator
from lsdo_airfoil.core.airfoil_model_csdl import AirfoilModelCSDL
import csdl
import numpy as np
from lsdo_airfoil.utils.compute_b_spline_mat import compute_b_spline_mat
from lsdo_airfoil import AIRFOIL_COORDINATES_FOLDER
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['text.usetex'] = False



num_nodes = 10
M = 0. 
AoA = np.linspace(-8, 15, num_nodes) 
Re =  8e6


run_model = csdl.Model()
run_model.create_input('mach_number', M * np.ones((num_nodes, )))
run_model.create_input('angle_of_attack', AoA)
run_model.create_input('reynolds_number', Re * np.ones((num_nodes, )))

run_model.add_design_variable('mach_number', lower=0., upper=0.6)
run_model.add_design_variable('angle_of_attack', scaler=3e-2, lower=2, upper=10)
run_model.add_design_variable('reynolds_number', scaler=0.5e-6, lower=1e5, upper=1e6)



d_star_upper = run_model.declare_variable('DeltaStarUpper', shape=(num_nodes, 100))
dummy_objective = csdl.sum(d_star_upper)
run_model.register_output('dummy_objective', dummy_objective)
run_model.add_objective('dummy_objective')

airfoil_model = AirfoilModelCSDL(
    num_nodes=num_nodes,
    compute_control_points=False,
    airfoil_name='Clark_y',
    airfoil_raw_shape=(301, )
)
run_model.add(airfoil_model, 'airfoil_model')

sim = Simulator(run_model)
sim.run()

# Displacement Thickness
dstar_upper = sim['DeltaStarUpper'].reshape(num_nodes, 100)
dstar_lower = sim['DeltaStarLower'].reshape(num_nodes, 100)

# Momentum Thickness
theta_upper = sim['ThetaUpper'].reshape(num_nodes, 100)
theta_lower = sim['ThetaLower'].reshape(num_nodes, 100)



num_pts = 100
x_range = np.linspace(0, 1, num_pts)
i_vec = np.arange(0, len(x_range))
x_interp = 1 - np.cos(np.pi/(2 * len(x_range)) * i_vec)


fig, axs = plt.subplots(2, 2, layout='constrained', figsize=(10, 8))
for i in range(num_nodes):
    axs[0, 0].plot(x_interp, dstar_upper[i, :], label=f'aoa = {AoA[i]}')#, label=r'$\delta^{*}_{suction}$')
# axs[0, 0].plot(x_interp, dstar_upper[2, :], label=f'aoa = {AoA[1]}')#, label=r'$\delta^{*}_{suction}$')
# axs[0, 0].plot(x_interp, np.exp(1.*x_interp)-0.99)

# axs[0, 0].plot(x_interp, theta_upper[-1, :], label=r'$\theta_{suction}$')
axs[0, 0].set_xlabel('x')
axs[0, 0].set_ylabel(r'$\delta^{*}_{suction}$')
axs[0, 0].grid()

for i in range(num_nodes):
    axs[0, 1].plot(x_interp, dstar_lower[i, :], label=f'aoa = {round(AoA[i], 2)}')# label=r'$\delta^{*}_{pressure}$')
# # axs[0, 1].plot(x_interp, theta_lower[0, :], label=r'$\theta_{pressure}$')
axs[0, 1].set_xlabel('x')
axs[0, 1].set_ylabel(r'$\delta^{*}_{pressure}$')
axs[0, 1].legend()
axs[0, 1].grid()

for i in range(num_nodes):
    axs[1, 0].plot(x_interp, theta_upper[i, :], label=f'aoa = {AoA[i]}')
axs[1, 0].set_ylabel(r'$\theta_{suction}$')
axs[1, 0].set_xlabel('x')
axs[1, 0].grid()

for i in range(num_nodes):
    axs[1, 1].plot(x_interp, theta_lower[i, :], label=f'aoa = {AoA[i]}')
axs[1, 1].set_xlabel('x')
axs[1, 1].set_ylabel(r'$\theta_{pressure}$')
axs[1, 1].grid()

fig.tight_layout()


plt.show()

