from blocks.algorithms import StepRule
from blocks.utils import shared_floatx
from blocks.roles import add_role, ALGORITHM_HYPERPARAMETER
from blocks.initialization import NdarrayInitialization
from fuel.transformers import AgnosticSourcewiseTransformer
from sklearn.preprocessing import MinMaxScaler
from skimage.io import imsave
from blocks.extensions import SimpleExtension
from matplotlib import pyplot
from sklearn.preprocessing import normalize
import numpy
import theano
import time
from theano import tensor
import os
import inspect


def export_image_array(image_array, output_folder, prefix='', suffix=''):
    image_array = numpy.asarray(image_array)
    image_array = zero_one_norm(image_array, axis=1)
    fig = pyplot.figure(frameon=False)
    fig.set_size_inches(0.28, 0.28)
    ax = pyplot.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    first_draw = True
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    for _ in range(image_array.shape[0]):
        image = image_array[_].reshape((28,28))
        if first_draw:
            img = ax.imshow(image, aspect='normal', interpolation='None', cmap=pyplot.cm.binary)
        else:
            img.set_data(image)
        fig.savefig(os.path.join(output_folder, '{}{}{}.png'.format(prefix, _, suffix)))
        first_draw = False
    pyplot.close(fig)


class SGD(StepRule):
    def __init__(self, start_init=1.0, increase=0.01, **kwargs):
        super(SGD, self).__init__(**kwargs)
        self.start_init = start_init - 1.0
        self.learning_rate_divide = shared_floatx(start_init, "learning_rate")
        self.divide_increase = shared_floatx(increase, "learning_rate_decrease")
        add_role(self.learning_rate_divide, ALGORITHM_HYPERPARAMETER)

    def compute_step(self, parameter, previous_step):
        self.learning_rate_divide += self.divide_increase
        return previous_step / self.learning_rate_divide, []


class DrawFilterData(SimpleExtension):
    def __init__(self, data, n_rows=4, n_cols=8, n_filters=16, sleep=0.0, **kwargs):
        super(DrawFilterData, self).__init__(**kwargs)
        self.D = data[0]
        self.E = data[1]
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.n_filters = n_filters
        self.sleep = sleep
        self.first_draw = True

    def do(self, which_callback, *args):
        E = numpy.asarray(self.E.eval())
        D = numpy.asarray(self.D.eval())
        pyplot.ion()
        im_list = []
        im_count = 0
        n_rows = self.n_rows
        n_cols = self.n_cols
        n_filters = self.n_filters
        for _ in xrange(n_filters):
            im_count += 1
            pyplot.subplot(n_rows, n_cols, im_count)
            if self.first_draw:
                im_list.append(pyplot.imshow(D[_].reshape((28,28)), interpolation='none', cmap=pyplot.cm.binary))
            else:
                im_list[_].set_data(D[_].reshape(28,28))
            pyplot.axis('off')
        for _ in xrange(n_filters):
            im_count += 1
            pyplot.subplot(n_rows, n_cols, im_count)
            if self.first_draw:
                im_list.append(pyplot.imshow(E[:, _].reshape((28,28)), interpolation='none', cmap=pyplot.cm.binary))
            else:
                im_list[8 + _].set_data(E[:, _].reshape(28,28))
            pyplot.axis('off')
        pyplot.pause(0.0001)
        #time.sleep(self.sleep)


class PrintDictionaryInfo(SimpleExtension):
    def __init__(self, data, **kwargs):
        super(PrintDictionaryInfo, self).__init__(**kwargs)
        self.data = data
        self.D = data[0]
        self.E = data[1]

    def do(self, which_callback, *args):
        for _ in xrange(len(self.data)):
            d = numpy.asarray(self.data[_].eval())
            print('{} {}'.format(d.sum(), d.__abs__().sum()))


class LearningRateMultiplier(StepRule):
    def __init__(self, multiplier=0.5, num_iter=10000, min_learning_rate=0.000001, **kwargs):
        super(LearningRateMultiplier, self).__init__(**kwargs)
        self.num_iter = num_iter
        self.iter_count = 0
        self.min_learning_rate = shared_floatx(min_learning_rate, "min_learning_rate")
        self.learning_rate = shared_floatx(1.0, "learning_rate")
        self.multiplier = shared_floatx(multiplier, "learning_rate_multiplier")
        add_role(self.learning_rate, ALGORITHM_HYPERPARAMETER)

    def compute_step(self, parameter, previous_step):
        self.iter_count += 1
        if self.iter_count % self.num_iter == 0:
            self.learning_rate = self.learning_rate * self.multiplier
        return previous_step * tensor.maximum(self.learning_rate, self.min_learning_rate), []


class L2NormalizeTransform(AgnosticSourcewiseTransformer):
    def __init__(self, data_stream, axis=1, **kwargs):
        super(L2NormalizeTransform, self).__init__(data_stream, data_stream.produces_examples, **kwargs)
        self.axis = axis

    def transform_any_source(self, source_data, _):
        return normalize(numpy.asarray(source_data), axis=self.axis)


class MeanShift(AgnosticSourcewiseTransformer):
    def __init__(self, data_stream, axis=1, **kwargs):
        super(MeanShift, self).__init__(data_stream, data_stream.produces_examples, **kwargs)
        self.axis = axis

    def transform_any_source(self, source_data, source_name):
        src_mean = numpy.asarray(source_data).mean(axis=self.axis)
        return numpy.asarray(source_data) - src_mean.reshape(src_mean.shape[0], 1)


class GlorotUniform(NdarrayInitialization):
    def __init__(self, scale=0.01):
        self.scale = scale

    def generate(self, rng, shape):
        w = numpy.sqrt(2.0 / (shape[0] + shape[1]))
        m = self.scale * rng.uniform(-w, w, size=shape)
        return m.astype(theano.config.floatX)


class GlorotNormal(NdarrayInitialization):
    def __init__(self, scale=0.01):
        self.scale = scale

    def generate(self, rng, shape):
        w = numpy.sqrt(2.0 / (shape[0] + shape[1]))
        m = self.scale * rng.normal(0.0, w, size=shape)
        return m.astype(theano.config.floatX)


class HeUniform(NdarrayInitialization):
    def __init__(self, scale=0.01):
        self.scale = scale

    def generate(self, rng, shape):
        w = numpy.sqrt(2.0 / (shape[0]))
        m = self.scale * rng.uniform(-w, w, size=shape)
        return m.astype(theano.config.floatX)


class HeNormal(NdarrayInitialization):
    def __init__(self, scale=0.01):
        self.scale = scale

    def generate(self, rng, shape):
        w = numpy.sqrt(2.0 / (shape[0]))
        m = self.scale * rng.normal(0.0, w, size=shape)
        return m.astype(theano.config.floatX)


def isdebugging():
    # http://stackoverflow.com/questions/333995/how-to-detect-that-python-code-is-being-executed-through-the-debugger
    for frame in inspect.stack():
        if frame[1].endswith("pydevd.py"):
            return True
    return False


def zero_one_norm(x, axis=0):
    x_min = numpy.min(x, axis, keepdims=True)
    x_max = numpy.max(x, axis, keepdims=True)
    return (x - x_min) / (x_max - x_min)
