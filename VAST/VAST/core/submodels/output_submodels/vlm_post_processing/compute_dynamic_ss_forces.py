from turtle import shape
# from csdl_om import Simulator
from csdl import Model
import csdl
from matplotlib.pyplot import axis
import numpy as np

from VAST.core.submodels.output_submodels.vlm_post_processing.compute_effective_aoa_cd_v import AOA_CD

panel_forces = sim['panel_forces'] + sim['panel_forces_dynamic']

u_val = (np.ones(num_nodes) * np.cos(alpha)).reshape((num_nodes,1)) 
w_vel = np.ones((num_nodes,1)) * np.sin(alpha) - h * np.cos(omg*t_vec).reshape((num_nodes,1))
acc = np.gradient(w_vel.flatten())/delta_t


import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
import matplotlib.pyplot as plt

plt.plot(t_vec, w_vel, '.-')
plt.plot(t_vec, acc, '.-')
plt.legend(['w_vel', 'acc'])
plt.plot(t_vec, sim['panel_forces_dynamic'][:,10,2], '.-')
# plt.plot(t_vec, sim['panel_forces'][:,10,2], '.-')
# plt.plot(t_vec, panel_forces[:,10,2], '.-')
plt.legend(['w_vel', 'acc','added_mass'])
# plt.legend(['w_vel', 'acc','added_mass','ss-forces','total'])
plt.show()