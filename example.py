from models import LinearRegression
from optimizers import MomentumGD
from loss_func import MeanSquaredError
from utils import read_data, printmat


X, Y = read_data('dataset2.txt')

model = LinearRegression()

optimizer = MomentumGD(0.00000001, MeanSquaredError)

model.fit(X, Y, optimizer=optimizer, epochs=100)

printmat('predictions', model.predict(X))

model.save('test')
