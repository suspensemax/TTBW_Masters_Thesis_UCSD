import python_csdl_backend
from VAST.utils.custom_einsums import EinsumIjKjKi
import csdl
import numpy as np

def test_einsum_IjKjKi():
    k = 4
    i = 2
    j = 3
    in_shape = (i, j, k)
    in_name_1 = 'in_1'
    in_name_2 = 'in_2'
    out_name = 'out'

    in_1_val = np.random.random((i, j))
    in_2_val = np.random.random((k, j))
    model = csdl.Model()
    a = model.declare_variable(in_name_1, val=in_1_val)
    b = model.declare_variable(in_name_2, val=in_2_val)
    product = csdl.custom(a,
                            b,
                            op=EinsumIjKjKi(in_name_1=in_name_1,
                                            in_name_2=in_name_2,
                                            ijk=in_shape,
                                            out_name=out_name))

    model.register_output(out_name, product)
    sim = python_csdl_backend.Simulator(model)

    sim.run()
    sim.check_partials(compact_print=True)