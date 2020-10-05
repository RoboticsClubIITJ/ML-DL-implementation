from MLlib.models import LinearRegression
from MLlib.optimizers import Adam
from MLlib.loss_func import MeanSquaredError
from MLlib.utils.misc_utils import read_data, printmat


X, Y = read_data('MLlib/datasets/linear_reg_00.txt')

linear_model = LinearRegression()

optimizer = Adam(0.001, MeanSquaredError)

linear_model.fit(X, Y, optimizer=optimizer, epochs=200, zeros=False)

printmat('predictions', linear_model.predict(X))

linear_model.save('test')
