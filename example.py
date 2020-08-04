from models import LinearRegression
from optimizers import SGD
from loss_func import MeanSquaredError
from utils import read_data


X, Y = read_data('dataset1.txt')

model = LinearRegression()

optimizer = SGD(0.0001, MeanSquaredError)

model.fit(X, Y, optimizer=optimizer, epochs=25)

print(model.predict(X))

model.save('test')
