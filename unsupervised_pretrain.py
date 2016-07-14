from rnn_with_constraints import LISTA, ENCODE, L2NormalizeLayer
from utils import (SGD, L2NormalizeTransform, GlorotUniform, LearningRateMultiplier, DrawFilterData, MeanShift,
                   GlorotNormal, HeUniform, isdebugging, PrintDictionaryInfo)
from blocks.initialization import IsotropicGaussian, Constant, Orthogonal
from blocks.bricks import Linear
from blocks.bricks.cost import SquaredError, AbsoluteError
from blocks.extensions.monitoring import DataStreamMonitoring, TrainingDataMonitoring
from blocks.graph import ComputationGraph
from blocks.filter import VariableFilter
from blocks.roles import WEIGHT
from blocks.main_loop import MainLoop
from blocks.algorithms import (GradientDescent, Scale, CompositeRule, Restrict, VariableClipping,
                               StepClipping, Adam, RMSProp, AdaGrad, AdaDelta)
from blocks.serialization import dump, load, load_parameters
from blocks.extensions import FinishAfter, Printing, Timing
from blocks.extensions.saveload import Checkpoint
from blocks_extras.algorithms import NesterovMomentum
from blocks_extras.extensions.plot import Plot
from fuel.datasets import MNIST
from fuel.streams import DataStream
from fuel.schemes import SequentialScheme, ShuffledScheme
from fuel.transformers import Flatten
from theano import tensor

import numpy
import matplotlib.pyplot as plt
import sys

DEBUG = isdebugging()
sys.setrecursionlimit(2000)
numpy.random.seed(1337)

num_of_passes = 10
LOAD = False
if DEBUG:
    LOAD = True
after_n_epochs = 10000

if LOAD:
    with open('main_loop.tar', 'rb') as src:
        l = load_parameters(src)
        b = Constant(l['/lista.b'])
        S = Constant(l['/lista.S'])
        E = Constant(l['/lista.E'])
        D = Constant(l['/linear.W'])
else:
    b = Constant(0.0)
    S = GlorotUniform(1.0)
    E = GlorotUniform(1.0)
    D = GlorotUniform(1.0)

lista = LISTA(input_size=784, dictionary_size=400, num_of_passes=num_of_passes, E_init=E, S_init=S, b_init=b)
decode = Linear(input_dim=400, output_dim=784, weights_init=D, use_bias=False)

x = tensor.matrix('features')
y = tensor.lmatrix('targets')

z = lista.apply(x)
output = decode.apply(z)

loss_1 = SquaredError().apply(x, output)
loss_1.name = 'loss_squared'
loss_2 = 0.01 * AbsoluteError().apply(z, 0)
loss_2.name = 'loss_sparse'
loss_3 = 0 * y.sum()
loss_3.name = 'loss_classification'
loss = loss_1 + loss_2
loss.name = 'loss_total'

lista.initialize()
decode.initialize()

mnist = MNIST(("train",))
data_stream = Flatten(DataStream.default_stream(
    mnist,
    iteration_scheme=ShuffledScheme(mnist.num_examples, batch_size=1280)))
data_stream = MeanShift(data_stream=data_stream, axis=1, which_sources='features')
data_stream = L2NormalizeTransform(data_stream=data_stream, axis=1, which_sources='features')

mnist_test = MNIST(("test",))
data_stream_test = Flatten(DataStream.default_stream(
    mnist_test,
    iteration_scheme=SequentialScheme(
        mnist_test.num_examples, batch_size=1280)))
data_stream_test = MeanShift(data_stream=data_stream_test, axis=1, which_sources='features')
data_stream_test = L2NormalizeTransform(data_stream=data_stream_test, axis=1, which_sources='features')

monitor = DataStreamMonitoring(
    variables=[loss, loss_1, loss_2], data_stream=data_stream_test, prefix="test")

cg = ComputationGraph(loss)

# E = VariableFilter(roles=[ENCODE])(cg.variables)[0]
# _, D = VariableFilter(roles=[WEIGHT])(cg.variables)
b, S, E, D = cg.parameters
b_scale = 1.0/num_of_passes
S_scale = 1.0/(num_of_passes-1)
E_scale = 1.0/num_of_passes
D_scale = 1.0

from utils import export_image_array
if DEBUG:
    export_image_array(D.eval(), '/home/nhat/dictionary_2/dictionary/','','d')
    export_image_array(E.T.eval(), '/home/nhat/dictionary_2/dictionary/','','e')
#DrawFilterData([D, E], n_rows=4, n_cols=8, n_filters=16, sleep=0, every_n_epochs=1).do(0)

for data in data_stream.get_epoch_iterator():
    Z = z.eval({x: data[0]})
    o = output.eval({x: data[0]})
    z_partial = lista.apply_T_iter(x, 5)
    Z_partial = z_partial.eval({x: data[0]})
    o_partial = decode.apply(z_partial).eval({x: data[0]})
    if DEBUG:
        export_image_array(data[0], '/home/nhat/dictionary_2/reconstruction/', '', 'x')
        export_image_array(o, '/home/nhat/dictionary_2/reconstruction/', '', 'o')
    break

algorithm = GradientDescent(cost=loss, parameters=cg.parameters,
                            step_rule=CompositeRule([
                                Restrict(Scale(learning_rate=b_scale), [b]),
                                Restrict(Scale(learning_rate=S_scale), [S]),
                                Restrict(Scale(learning_rate=E_scale), [E]),
                                Restrict(Scale(learning_rate=D_scale), [D]),
                                Scale(learning_rate=0.1),
                                AdaDelta(),
                                Restrict(VariableClipping(1.25/num_of_passes, axis=1), [E]),
                                Restrict(VariableClipping(1, axis=0), [D]),
                            ]),
                            on_unused_sources='ignore')

main_loop = MainLoop(data_stream=data_stream, algorithm=algorithm, model=cg,
                     extensions=[Timing(),
                                 FinishAfter(after_n_epochs=after_n_epochs),
                                 monitor,
                                 TrainingDataMonitoring([loss, loss_1, loss_2], after_batch=True),
                                 Checkpoint('main_loop.tar', parameters=cg.parameters, save_main_loop=False,
                                            use_cpickle=True, every_n_epochs=20),
                                 Printing(),
                                 DrawFilterData([D, E], n_rows=4, n_cols=8, n_filters=16, sleep=0, every_n_epochs=20,
                                                before_training=True),
                                 PrintDictionaryInfo([D, E, S, b], every_n_epochs=1, before_training=True)
                                 # Plot('Plotting example', channels=['loss', 'loss_1', 'loss_2'], after_batch=True)
                                 ])

main_loop.run()

# with open('model/main_loop.tar', 'wb') as dst:
#     dump(main_loop, dst, use_cpickle=True, parameters=main_loop.model.parameters)

pass
