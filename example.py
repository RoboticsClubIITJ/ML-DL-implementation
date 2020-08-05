from MLlib.models import LinearRegression
from MLlib.optimizers import MomentumGD
from MLlib.loss_func import MeanSquaredError
from MLlib.utils import read_data, printmat


X, Y = read_data('MLlib/datasets/linear_reg_00.txt')

model = LinearRegression()

optimizer = MomentumGD(0.00001, MeanSquaredError)

model.fit(X, Y, optimizer=optimizer, epochs=100)

printmat('predictions', model.predict(X))

model.save('test')
