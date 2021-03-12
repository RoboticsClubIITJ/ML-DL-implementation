from math import sin
from math import pi
from numpy.random import normal
from numpy import vstack
from numpy import argmax
from numpy import asarray
from numpy.random import random
from sklearn.gaussian_process import GaussianProcessRegressor
from MLlib.models import Bayes_Optimization


# objective function
def objective(x, noise=0.1):
    noise = normal(loc=0, scale=noise)
    return (x**2 * sin(5 * pi * x)**6.0) + noise


X = random(100)
y = asarray([objective(x) for x in X])
# reshape into rows and cols
X = X.reshape(len(X), 1)
y = y.reshape(len(y), 1)
# define the model
model = GaussianProcessRegressor()
# fit the model
model.fit(X, y)
# plot before hand
a = Bayes_Optimization()
a.plot(X, y, model)
# perform the optimization process
for i in range(100):
    # select the next point to sample
    x = Bayes_Optimization()
    b = x.opt_acquisition(X, y, model)
    # sample the point
    actual = objective(b)
    # summarize the finding
    est1, _ = x.surrogate(model, [[b]])
    print('>x=%.3f, f()=%3f, actual=%.3f' % (b, est1, actual))
    # add the data to the dataset
    X = vstack((X, [[b]]))
    y = vstack((y, [[actual]]))
    # update the model
    model.fit(X, y)

# plot all samples and the final surrogate function
h = Bayes_Optimization()
h.plot(X, y, model)
# best result
ix = argmax(y)
print('Best Result: x=%.3f, y=%.3f' % (X[ix], y[ix]))
