import csdl
import numpy as np

class ReplaceZeros(csdl.CustomExplicitOperation):
    '''
    This operation is used to index a matrix with a vector of indices.

    parameters
    ----------
    matrix : csdl 2d array [num_nodes, nn]
        Matrix to be indexed
    ind_array : numpy 0d array [n_ind,]
        array of indices

    Returns
    -------
    indexed_matrix : csdl 2d array [num_nodes, n_ind]
        Indexed matrix    
    '''
    def initialize(self):
        self.parameters.declare('in_name', types=str)
        self.parameters.declare('in_shape', types=tuple)
        self.parameters.declare('out_name', types=str)
    def define(self):
        in_name = self.parameters['in_name']
        out_name = self.parameters['out_name']
        in_shape = self.parameters['in_shape']

        self.add_input(in_name,shape=in_shape)
        self.add_output(out_name,shape=in_shape)

        row_indices = np.arange(np.prod(in_shape))
        col_indices = row_indices
        
        self.declare_derivatives(out_name, in_name, rows=row_indices,cols=col_indices)

    def compute(self, inputs, outputs):
        in_name = self.parameters['in_name']
        out_name = self.parameters['out_name']

        # find the flattened indices of the zero elements
        zero_ind = np.nonzero(inputs[in_name] == 0)
        outputs[out_name] = np.where(inputs[in_name] == 0, 1e-8, inputs[in_name])

    def compute_derivatives(self, inputs, derivatives):
        out_name = self.parameters['out_name']
        in_name = self.parameters['in_name']
        zero_ind = np.flatnonzero(inputs[in_name] != 0)
        derivatives[out_name, in_name][:] = 1

if __name__ == "__main__":
    import python_csdl_backend

    num_nodes = 3
    nn = 4
    in_name = 'in_mat'
    out_name = 'out_mat'
    in_shape = (num_nodes,nn)
    ind_array = np.array([0,2,3,3,4,5,5,6])
    out_name = 'out_mat'
    in_val = np.random.random(in_shape)
    in_val[0,0] = 0
    in_val[1,2] = 0

    model = csdl.Model()
    in_mat = model.create_input(in_name, val=in_val)
    out_mat = csdl.custom(in_mat,
                          op=ReplaceZeros(in_name=in_name,
                                            in_shape=in_shape,
                                            out_name=out_name))

    model.register_output(out_name, out_mat)
    sim = python_csdl_backend.Simulator(model)

    sim.run()
    sim.check_partials(compact_print=True,step=1e-10)
    print('in_mat:',sim['in_mat'].shape,'\n',sim['in_mat'])
    print('out_mat:',sim['out_mat'].shape,'\n',sim['out_mat'])