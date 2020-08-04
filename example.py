from models import LinearRegression
from optimizers import GradientDescent
from loss_func import MeanSquaredError
from utils import read_data


X, Y = read_data('dataset1.txt')

model = LinearRegression()

optimizer = GradientDescent(0.001, MeanSquaredError)

model.fit(X, Y, optimizer=optimizer, epochs=25)

print(model.predict(X))

model.save('test')
