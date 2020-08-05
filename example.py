from MLlib.models import LinearRegression
from MLlib.optimizers import Adam
from MLlib.loss_func import MeanSquaredError
from MLlib.utils import read_data, printmat


X, Y = read_data('MLlib/datasets/linear_reg_00.txt')

model = LinearRegression()

optimizer = Adam(0.03, MeanSquaredError, beta1=0.5)

model.fit(X, Y, optimizer=optimizer, epochs=250, zeros=True)

printmat('predictions', model.predict(X))

model.save('test')
