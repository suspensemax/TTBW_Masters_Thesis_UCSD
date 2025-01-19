import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['text.usetex'] = False
import numpy as np


num_pts = 100
x_range = np.linspace(0, 1, num_pts)
i_vec = np.arange(0, len(x_range))
x_interp = 1 - np.cos(np.pi/(2 * len(x_range)) * i_vec)


def plot_distributions(sim):
    
    Cp_upper = sim['cp_model.cp_upper']
    Cp_lower = sim['cp_model.cp_lower']

    delta_star_upper = sim['dstar_model.dstar_upper']
    delta_star_lower = sim['dstar_model.dstar_lower']

    theta_upper = sim['theta_model.theta_upper']
    theta_lower = sim['theta_model.theta_lower']

    edge_vel_upper = sim['edge_velocity_model.edge_velocity_upper']
    edge_vel_lower = sim['edge_velocity_model.edge_velocity_lower']
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    axs[0, 0].plot(x_interp, Cp_upper)
    axs[0, 0].plot(x_interp, Cp_lower)
    axs[0, 0].invert_yaxis()
    axs[0, 0].set_ylabel('Pressure coefficient')


    axs[0, 1].plot(x_interp, delta_star_upper)
    axs[0, 1].plot(x_interp, delta_star_lower)
    axs[0, 1].set_ylabel('Displacement thickness')

    axs[1, 0].plot(x_interp, theta_upper)
    axs[1, 0].plot(x_interp, theta_lower)
    axs[1, 0].set_ylabel('Momentum thickness')

    axs[1, 1].plot(x_interp, edge_vel_upper)
    axs[1, 1].plot(x_interp, edge_vel_lower)
    axs[1, 1].set_ylabel('Edge velocity')

    plt.suptitle(f"Cl = {sim['cl_model.Cl'].flatten()}" + '\n' +f"Cd = {sim['cd_model.Cd'].flatten()}" + '\n' + f"Cm = {sim['cm_model.Cm']}")

    plt.show()
