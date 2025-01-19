import csdl
import numpy as np

# EinsumIjKjKi ij,kj->ki
class EinsumIjKjKi(csdl.CustomExplicitOperation):
    """
    helper function to deal with Einsum memory issue in csdl
    """
    def initialize(self):
        """
        Declare parameters.
        """
        self.parameters.declare("in_name_1")
        self.parameters.declare("in_name_2")
        self.parameters.declare("ijk")
        self.parameters.declare("out_name")
        # print('finish initialize---------------')

    def define(self):

        in_name_1 = self.parameters["in_name_1"]
        in_name_2 = self.parameters["in_name_2"]
        in_shape = self.parameters["ijk"]

        out_name = self.parameters["out_name"]

        self.add_output(out_name, shape=(in_shape[2], in_shape[0]))
        self.add_input(in_name_1, shape=(in_shape[0], in_shape[1]))
        self.add_input(in_name_2, shape=(in_shape[2], in_shape[1]))

        rows_pa = np.outer(np.arange(in_shape[0] * in_shape[2]), np.ones(in_shape[1])).flatten()
        cols_pa = np.outer(np.ones(in_shape[2]),np.arange(in_shape[0] * in_shape[1])).flatten()
        rows_pb = rows_pa
        cols_pb = np.repeat(np.arange(in_shape[2]*in_shape[1]).reshape(-1,in_shape[1]), repeats=in_shape[0], axis=0).flatten()
               
        self.declare_derivatives(out_name, in_name_1, rows=rows_pa, cols=cols_pa)
        self.declare_derivatives(out_name, in_name_2, rows=rows_pb, cols=cols_pb)

    def compute(self, inputs, outputs):
        in_name_1 = self.parameters["in_name_1"]
        in_name_2 = self.parameters["in_name_2"]
        out_name = self.parameters["out_name"]

        outputs[out_name] = np.einsum(
            'ij,kj->ki',
            inputs[in_name_1],
            inputs[in_name_2],
        )

    def compute_derivatives(self, inputs, derivatives):
        in_name_1 = self.parameters["in_name_1"]
        in_name_2 = self.parameters["in_name_2"]
        out_name = self.parameters["out_name"]
        in_shape = self.parameters["ijk"]

        derivatives[out_name, in_name_1] = np.repeat(inputs[in_name_2],repeats=in_shape[0],axis=0).flatten()
        
        derivatives[out_name, in_name_2] = np.tile(inputs[in_name_1].flatten(),in_shape[2]).flatten()

# EinsumKijKijKi kij,kij->ki
class EinsumKijKijKi(csdl.CustomExplicitOperation):
    def initialize(self):
        """
        Declare parameters.
        """
        self.parameters.declare("in_name_1")
        self.parameters.declare("in_name_2")
        self.parameters.declare("in_shape")
        self.parameters.declare("out_name")
        # print('finish initialize---------------')

    def define(self):

        in_name_1 = self.parameters["in_name_1"]
        in_name_2 = self.parameters["in_name_2"]
        in_shape = self.parameters["in_shape"]

        out_name = self.parameters["out_name"]

        self.add_output(out_name, shape=(in_shape[0], in_shape[1]))
        self.add_input(in_name_1, shape=in_shape)
        self.add_input(in_name_2, shape=in_shape)

        rows = np.outer(np.arange(in_shape[0] * in_shape[1]),
                        np.ones(in_shape[2])).flatten()
        cols = np.arange(in_shape[0] * in_shape[1] * in_shape[2]).flatten()
        # print('rows\n', rows.shape)
        # print('cols\n', cols.shape)
        self.declare_derivatives(out_name, in_name_1, rows=rows, cols=cols)
        self.declare_derivatives(out_name, in_name_2, rows=rows, cols=cols)
        # print('finish define----------------')

    def compute(self, inputs, outputs):
        in_name_1 = self.parameters["in_name_1"]
        in_name_2 = self.parameters["in_name_2"]
        out_name = self.parameters["out_name"]

        outputs[out_name] = np.einsum(
            'kij,kij->ki',
            inputs[in_name_1],
            inputs[in_name_2],
        )
        # print(outputs[out_name])

    def compute_derivatives(self, inputs, derivatives):
        in_name_1 = self.parameters["in_name_1"]
        in_name_2 = self.parameters["in_name_2"]
        out_name = self.parameters["out_name"]

        derivatives[out_name, in_name_1] = inputs[in_name_2].flatten()
        derivatives[out_name, in_name_2] = inputs[in_name_1].flatten()

# EinsumKijKjKi kij,kj->ki
class EinsumKijKjKi(csdl.CustomExplicitOperation):
    def initialize(self):
        """
        Declare parameters.
        """
        self.parameters.declare("in_name_1")
        self.parameters.declare("in_name_2")
        self.parameters.declare("in_shape")
        self.parameters.declare("out_name")
        # print('finish initialize---------------')

    def define(self):

        in_name_1 = self.parameters["in_name_1"]
        in_name_2 = self.parameters["in_name_2"]
        in_shape = self.parameters["in_shape"]

        out_name = self.parameters["out_name"]

        self.add_output(out_name, shape=(in_shape[0], in_shape[1]))
        self.add_input(in_name_1, shape=in_shape)
        self.add_input(in_name_2, shape=(in_shape[0], in_shape[2]))

        rows_pa = np.outer(np.arange(in_shape[0] * in_shape[1]), np.ones(in_shape[2]), ).flatten()
        cols_pa = np.arange(in_shape[0] * in_shape[1] * in_shape[2]).flatten()
        rows_pb = rows_pa
        cols_pb = np.repeat(np.arange(in_shape[2]*in_shape[0]).reshape(in_shape[0],-1), repeats=in_shape[1], axis=0).flatten()
               
        self.declare_derivatives(out_name, in_name_1, rows=rows_pa, cols=cols_pa)
        self.declare_derivatives(out_name, in_name_2, rows=rows_pb, cols=cols_pb)

    def compute(self, inputs, outputs):
        in_name_1 = self.parameters["in_name_1"]
        in_name_2 = self.parameters["in_name_2"]
        out_name = self.parameters["out_name"]

        outputs[out_name] = np.einsum(
            'kij,kj->ki',
            inputs[in_name_1],
            inputs[in_name_2],
        )
        # print(outputs[out_name])

    def compute_derivatives(self, inputs, derivatives):
        in_name_1 = self.parameters["in_name_1"]
        in_name_2 = self.parameters["in_name_2"]
        out_name = self.parameters["out_name"]
        in_shape = self.parameters["in_shape"]

        derivatives[out_name, in_name_1] = np.repeat(inputs[in_name_2],repeats=in_shape[1],axis=0).flatten()
        
        derivatives[out_name, in_name_2] = inputs[in_name_1].flatten()

# EinsumLijkLikLij lijk,lik->lij
class EinsumLijkLikLij(csdl.CustomExplicitOperation):
    def initialize(self):
        """
        Declare parameters.
        """
        self.parameters.declare("in_name_1")
        self.parameters.declare("in_name_2")
        self.parameters.declare("in_shape")
        self.parameters.declare("out_name")
        # print('finish initialize---------------')

    def define(self):

        in_name_1 = self.parameters["in_name_1"]
        in_name_2 = self.parameters["in_name_2"]
        in_shape = self.parameters["in_shape"]

        out_name = self.parameters["out_name"]

        self.add_output(out_name,
                        shape=(in_shape[0], in_shape[1], in_shape[2]))
        self.add_input(in_name_1, shape=in_shape)
        self.add_input(in_name_2,
                       shape=(in_shape[0], in_shape[1], in_shape[3]))

        rows_1 = np.outer(np.arange(in_shape[0] * in_shape[1] * in_shape[2]),
                          np.ones(in_shape[3])).flatten()
        cols_1 = np.arange(in_shape[0] * in_shape[1] * in_shape[2] *
                           in_shape[3]).flatten()

        rows_2 = np.einsum(
            'ilk,j->ilkj',
            np.arange(in_shape[0] * in_shape[1] * in_shape[2]).reshape(
                in_shape[0], in_shape[1], in_shape[2]),
            np.ones(in_shape[3])).flatten()
        cols_2 = np.einsum(
            'ilk,j->iljk',
            np.arange(in_shape[0] * in_shape[1] * in_shape[3]).reshape(
                in_shape[0], in_shape[1], in_shape[3]),
            np.ones(in_shape[2])).flatten()

        self.declare_derivatives(out_name, in_name_1, rows=rows_1, cols=cols_1)
        self.declare_derivatives(out_name, in_name_2, rows=rows_2, cols=cols_2)

    def compute(self, inputs, outputs):
        in_name_1 = self.parameters["in_name_1"]
        in_name_2 = self.parameters["in_name_2"]
        out_name = self.parameters["out_name"]

        outputs[out_name] = np.einsum(
            'lijk,lik->lij',
            inputs[in_name_1],
            inputs[in_name_2],
        )

    def compute_derivatives(self, inputs, derivatives):
        in_name_1 = self.parameters["in_name_1"]
        in_name_2 = self.parameters["in_name_2"]
        in_shape = self.parameters["in_shape"]
        out_name = self.parameters["out_name"]

        derivatives[out_name,
                    in_name_1] = np.einsum('lik,j->lijk', inputs[in_name_2],
                                           np.ones(in_shape[2])).flatten()
        derivatives[out_name, in_name_2] = inputs[in_name_1].flatten()

# EinsumLijkLjLik lijk,lj->lik
class EinsumLijkLjLik(csdl.CustomExplicitOperation):
    def initialize(self):
        """
        Declare parameters.
        """
        self.parameters.declare("in_name_1")
        self.parameters.declare("in_name_2")
        self.parameters.declare("in_shape")
        self.parameters.declare("out_name")

    def define(self):

        in_name_1 = self.parameters["in_name_1"]
        in_name_2 = self.parameters["in_name_2"]
        in_shape = self.parameters["in_shape"]

        out_name = self.parameters["out_name"]

        self.add_output(out_name,
                        shape=(in_shape[0], in_shape[1], in_shape[3]))
        self.add_input(in_name_1, shape=in_shape)
        self.add_input(in_name_2, shape=(in_shape[0], in_shape[2]))

        rows_1 = np.einsum(
            'ilk,j->iljk',
            np.arange(in_shape[0] * in_shape[1] * in_shape[3]).reshape(
                in_shape[0], in_shape[1], in_shape[3]),
            np.ones(in_shape[2])).flatten()

        cols_1 = np.arange(in_shape[0] * in_shape[1] * in_shape[2] *
                           in_shape[3])

        rows_2 = np.einsum(
            'ik,j->ijk',
            np.arange(in_shape[0] * in_shape[1] * in_shape[3]).reshape(
                in_shape[0], in_shape[1] * in_shape[3]),
            np.ones(in_shape[2])).flatten()
        cols_2 = np.einsum(
            'ik,j->ikj',
            np.arange(in_shape[0] * in_shape[2]).reshape(
                in_shape[0], in_shape[2]),
            np.ones(in_shape[1] * in_shape[3])).flatten()
        self.declare_derivatives(out_name, in_name_1, rows=rows_1, cols=cols_1)
        self.declare_derivatives(out_name, in_name_2, rows=rows_2, cols=cols_2)

    def compute(self, inputs, outputs):
        in_name_1 = self.parameters["in_name_1"]
        in_name_2 = self.parameters["in_name_2"]
        out_name = self.parameters["out_name"]

        outputs[out_name] = np.einsum(
            'lijk,lj->lik',
            inputs[in_name_1],
            inputs[in_name_2],
        )

    def compute_derivatives(self, inputs, derivatives):
        in_name_1 = self.parameters["in_name_1"]
        in_name_2 = self.parameters["in_name_2"]
        in_shape = self.parameters["in_shape"]
        out_name = self.parameters["out_name"]

        derivatives[out_name,
                    in_name_1] = np.einsum('lj,ik->lijk', inputs[in_name_2],
                                           np.ones((in_shape[1],
                                                    in_shape[3]))).flatten()
        derivatives[out_name, in_name_2] = np.moveaxis(inputs[in_name_1], 2, 1).flatten()