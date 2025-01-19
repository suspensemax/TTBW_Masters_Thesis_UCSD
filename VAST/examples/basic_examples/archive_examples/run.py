from ozone.api import ODEProblem

import os
import numpy as np

file_name = '../examples/basic_examples/vnv_meshes/byu_vortex_lattice/x.txt'
full_path = os.path.abspath(file_name)
x = np.loadtxt(full_path)
print('the shape of x is: ', x.shape)