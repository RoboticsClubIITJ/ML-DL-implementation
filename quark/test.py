from Stackker import Sequential
from layers import FFL, Conv2d, Dropout, Flatten, Pool2d
# from optimizers import Optimizer
# import numpy as np
import pandas as pd
from misc import load_model


# this test works
# x = np.arange(0, 100).reshape(-1, 1)
# x = x / x.max()
# y = x * 2
# mult = Sequential()
# mult.add(FFL(1, 1, activation="sigmoid"))
# mult.add(FFL(neurons=1))
# mult.summary()
# mult.compile_model(lr=0.01, opt="sgd", loss="mse", mr= 0.001)
# mult.train(x, y, 10000,  show_every=100, batch_size = 8, shuffle=True)

# this test also works
# from keras.datasets import mnist
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# x = x_train.reshape(-1, 28 * 28)
# x = (x-x.mean(axis=1).reshape(-1, 1))/x.std(axis=1).reshape(-1, 1)
# y = pd.get_dummies(y_train).to_numpy()
# xt = x_test.reshape(-1, 28 * 28)
# xt = (xt-xt.mean(axis=1).reshape(-1, 1))/xt.std(axis=1).reshape(-1, 1)
# yt = pd.get_dummies(y_test).to_numpy()
# m = Sequential()
# m.add(FFL(784, 10, activation='sigmoid'))
# m.add(FFL(10, 10, activation="softmax"))
# m.compile_model(lr=0.01, opt="adam", loss="cse", mr= 0.001)
# m.summary()
# m.train(x[:], y[:], epochs=10, batch_size=32, val_x=xt, val_y = yt)

# this test works

from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x = x_train.reshape(-1, 28 * 28)
x = (x-x.mean(axis=1).reshape(-1, 1))/x.std(axis=1).reshape(-1, 1)
x = x.reshape(-1, 28, 28, 1)
y = pd.get_dummies(y_train).to_numpy()
xt = x_test.reshape(-1, 28 * 28)
xt = (xt-xt.mean(axis=1).reshape(-1, 1))/xt.std(axis=1).reshape(-1, 1)
xt = xt.reshape(-1, 28, 28, 1)
yt = pd.get_dummies(y_test).to_numpy()

m = Sequential()
m.add(Conv2d(
    input_shape=(28, 28, 1), filters=4, padding=None,
    kernel_size=(3, 3), activation="relu"))
m.add(Conv2d(filters=8, kernel_size=(3, 3), padding=None, activation="relu"))
m.add(Pool2d(kernel_size=(2, 2)))
m.add(Flatten())
m.add(FFL(neurons=64, activation="relu"))
m.add(Dropout(0.1))

m.add(FFL(neurons=10, activation='softmax'))
m.compile_model(lr=0.01, opt="adam", loss="cse")
m.summary()
m.train(x[:30], y[:30], epochs=2, batch_size=30, val_x=xt[:10], val_y=yt[:10])
m.visualize()
m.save_model()
load_model()
m.summary()
print(m.predict(x[10]))
