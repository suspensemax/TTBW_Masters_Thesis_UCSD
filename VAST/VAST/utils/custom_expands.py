import csdl
import numpy as np


class ExpandIjkIjlk(csdl.CustomExplicitOperation):
    def initialize(self):
        """
        Declare parameters.
        """
        self.parameters.declare("in_name_1")
        self.parameters.declare("in_shape")
        self.parameters.declare("out_shape")
        self.parameters.declare("out_name")

    def define(self):

        in_name_1 = self.parameters["in_name_1"]
        in_shape = self.parameters["in_shape"]

        out_shape = self.parameters["out_shape"]
        out_name = self.parameters["out_name"]

        self.add_input(in_name_1, shape=in_shape)
        self.add_output(out_name, shape=out_shape)

        rows = np.arange(out_shape[0] * out_shape[1] * out_shape[2] * out_shape[3]).flatten()
        
        cols = (np.outer(np.ones(out_shape[0] * out_shape[1]), np.outer(np.ones(out_shape[2]), np.arange(in_shape[2])).flatten())+\
             np.arange(out_shape[0]*out_shape[1]).reshape(-1,1)*in_shape[2]).flatten()
        

        self.declare_derivatives(out_name, in_name_1, rows=rows, cols=cols)

    def compute(self, inputs, outputs):
        in_name_1 = self.parameters["in_name_1"]

        out_name = self.parameters["out_name"]
        out_shape = self.parameters["out_shape"]

        outputs[out_name] = np.einsum(
            'ijk,l->ijlk', inputs[in_name_1], np.ones(out_shape[2]),
        )

    def compute_derivatives(self, inputs, derivatives):
        in_name_1 = self.parameters["in_name_1"]
        out_name = self.parameters["out_name"]
        out_shape = self.parameters["out_shape"]

        derivatives[out_name, in_name_1] = np.ones(out_shape[0] * out_shape[1] * out_shape[2] * out_shape[3])


class ExpandIjkIljk(csdl.CustomExplicitOperation):
    def initialize(self):
        """
        Declare parameters.
        """
        self.parameters.declare("in_name_1")
        self.parameters.declare("in_shape")
        self.parameters.declare("out_shape")
        self.parameters.declare("out_name")

    def define(self):

        in_name_1 = self.parameters["in_name_1"]
        in_shape = self.parameters["in_shape"]

        out_shape = self.parameters["out_shape"]
        out_name = self.parameters["out_name"]

        self.add_input(in_name_1, shape=in_shape)
        self.add_output(out_name, shape=out_shape)

        rows = np.arange(out_shape[0] * out_shape[1] * out_shape[2] * out_shape[3]).flatten()
        
        cols = (np.outer(
            np.ones(out_shape[0]),
            np.outer(np.ones(out_shape[1]), np.arange(in_shape[1] * in_shape[2])).flatten()
        ) + np.arange(out_shape[0]).reshape(out_shape[0],1)*in_shape[1]*in_shape[2]).flatten()

        self.declare_derivatives(out_name, in_name_1, rows=rows, cols=cols)

    def compute(self, inputs, outputs):
        in_name_1 = self.parameters["in_name_1"]

        out_name = self.parameters["out_name"]
        out_shape = self.parameters["out_shape"]

        outputs[out_name] = np.einsum(
            'ijk,l->iljk', inputs[in_name_1],np.ones(out_shape[1]),
        )

    def compute_derivatives(self, inputs, derivatives):
        in_name_1 = self.parameters["in_name_1"]
        out_name = self.parameters["out_name"]
        out_shape = self.parameters["out_shape"]

        derivatives[out_name, in_name_1] = np.ones(out_shape[0] * out_shape[1] * out_shape[2] * out_shape[3])


# if __name__ == "__main__":
#     import python_csdl_backend
#     i = 2
#     j = 3
#     k = 5
#     l = 4
#     in_shape = (i, j, k)
#     out_shape = (i, j, l, k)
#     in_name_1 = 'in_1'
#     out_name = 'out'

#     in_1_val = np.random.random((i, j, k))
#     model = csdl.Model()
#     a = model.declare_variable(in_name_1, val=in_1_val)
#     product = csdl.custom(a,
#                           op=ExpandIjkIjlk(in_name_1=in_name_1,
#                                             in_shape=in_shape,
#                                             out_name=out_name,
#                                             out_shape=out_shape))

#     model.register_output(out_name, product)
#     sim = python_csdl_backend.Simulator(model)

#     sim.run()
#     sim.check_partials(compact_print=True)

# if __name__ == "__main__":
#     import python_csdl_backend
#     i = 2
#     j = 4
#     k = 5
#     l = 3
#     in_shape = (i, j, k)
#     out_shape = (i, l, j, k)
#     in_name_1 = 'in_1'
#     out_name = 'out'

#     in_1_val = np.random.random((i,j, k))
#     model = csdl.Model()
#     a = model.declare_variable(in_name_1, val=in_1_val)
#     product = csdl.custom(a,
#                           op=ExpandIjkIljk(in_name_1=in_name_1,
#                                             in_shape=in_shape,
#                                             out_name=out_name,
#                                             out_shape=out_shape))

#     model.register_output(out_name, product)
#     sim = python_csdl_backend.Simulator(model)

#     sim.run()
#     sim.check_partials(compact_print=True)