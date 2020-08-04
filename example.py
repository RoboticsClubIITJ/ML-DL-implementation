from models import LinearRegression
from optimizers import MiniBatchGD
from loss_func import MeanSquaredError
from utils import read_data, printmat


X, Y = read_data('dataset1.txt')

model = LinearRegression()

optimizer = MiniBatchGD(0.0001, MeanSquaredError)

model.fit(X, Y, optimizer=optimizer, epochs=100)

printmat('predictions', model.predict(X))

model.save('test')
