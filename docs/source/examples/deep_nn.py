from pydaalcontrib.nn import *
from pydaalcontrib.model.nn import (
	Model, Relu, SigmoidCrossEntropy, Gaussian, Uniform, Xavier, TruncatedGaussian
)
from daal.algorithms.optimization_solver import adagrad
from numpy.testing import assert_equal
import numpy as np

x = conv2d([3,3,6], strides=[2,2], paddings=[2,2])
x = x.with_weights_initializer(Gaussian(0, 1./100))
x = x.with_biases_initializer(Uniform(0, 0))
x = x(max_pool2d([2,2]))(Relu())(dropout(.5))
x = x(conv2d([3,3,12], strides=[2,2], paddings=[2,2]))
x = x.with_weights_initializer(Xavier())
x = x.with_biases_initializer(Xavier())
x = x(max_pool2d([2,2]))(Relu())(dropout(.5))
x = x(conv2d([3,3,24], strides=[2,2], paddings=[2,2]))
x = x.with_weights_initializer(TruncatedGaussian(0, 1./100).with_bounds(-2, 2))
x = x.with_biases_initializer(Uniform(0, 0))
x = x(avg_pool2d([2,2]))(Relu())
x = x(fc(10))(lrn())(fc(1))
x = x(SigmoidCrossEntropy())

model = Model(x)

num_epochs = 10
np.random.seed(0)
labels = np.append(np.zeros(100), np.ones(100))
data = np.random.rand(200, 3, 60, 60)
data[:100] = data[:100]*-1

net = DAALNet().build(model).with_solver(adagrad.Batch())

for i in range(num_epochs):
	perm = np.random.permutation(200)
	net.train(data[perm], labels[perm], batch_size=10)

with net.predict(data) as predictions:
	assert_equal(np.rint(predictions).ravel(), labels)