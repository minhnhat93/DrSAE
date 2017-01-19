from blocks.bricks import Initializable, Brick
from blocks.bricks.base import application
from blocks.bricks.cost import Cost
from blocks.utils import shared_floatx_nans
from blocks.initialization import IsotropicGaussian, Constant
from blocks.roles import add_role, WEIGHT, BIAS, VariableRole
from theano.tensor.nnet import relu
from theano import tensor


class EncodeRole(VariableRole):
    pass

ENCODE = EncodeRole()

def srhinkage():
    pass

class LISTA(Initializable):

    def __init__(self, input_size, dictionary_size, num_of_passes, E_init, S_init, b_init, **kwargs):
        super(LISTA, self).__init__(**kwargs)
        self.input_size = input_size
        self.dictionary_size = dictionary_size
        self.num_of_passes = num_of_passes
        self.E_init = E_init
        self.S_init = S_init
        self.b_init = b_init

    def _allocate(self):
        E = shared_floatx_nans((self.input_size, self.dictionary_size), name='E')
        S = shared_floatx_nans((self.dictionary_size, self.dictionary_size), name='S')
        b = shared_floatx_nans((self.dictionary_size,), name='b')
        add_role(E, ENCODE)
        add_role(S, WEIGHT)
        add_role(b, BIAS)
        self.parameters.append(E)
        self.parameters.append(S)
        self.parameters.append(b)

    def _initialize(self):
        E, S, b = self.parameters
        self.E_init.initialize(E, self.rng)
        self.S_init.initialize(S, self.rng)
        self.b_init.initialize(b, self.rng)

    def get_dim(self, name):
        if name == 'input_size':
            return self.input_size
        if name == 'dictionary_size':
            return self.dictionary_size

    @application(inputs=['input_'], outputs=['output'])
    def apply(self, input_):
        E, S, b = self.parameters
        z = relu(tensor.dot(input_, E) - b)
        for _ in xrange(self.num_of_passes - 1):
            input_to_layer = tensor.dot(input_, E) + tensor.dot(z, S) - b
            z = relu(input_to_layer)
        return z

    def apply_T_iter(self, input_, t):
        E, S, b = self.parameters
        z = relu(tensor.dot(input_, E) - b)
        for _ in xrange(t - 1):
            input_to_layer = tensor.dot(input_, E) + tensor.dot(z, S) - b
            z = relu(input_to_layer)
        return z


class L2NormalizeLayer(Brick):
    @application(inputs=['input_'], outputs=['output'])
    def apply(self, input_):
        return input_ / input_.norm(L=2, axis=1).reshape((input_.shape[0], 1))


class MultinominalLogisticLoss(Cost):
    @application(outputs=['output'])
    def apply(self, y, z):
        index = tensor.arange(y.shape[0]).reshape((-1, 1)) * z.shape[1] + y
        z = z - tensor.log(tensor.exp(z).sum(axis=1)).reshape((-1,1))
        return -z.flatten()[index].sum()
