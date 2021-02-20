from MLlib.models import PolynomialRegression
from MLlib.optimizers import Adam
from MLlib.loss_func import MeanSquaredError
from MLlib.utils.misc_utils import read_data, printmat


X, Y = read_data('datasets/Polynomial_reg.txt')

polynomial_model = PolynomialRegression()

optimizer = Adam(0.01, MeanSquaredError)

polynomial_model.fit(X, Y, optimizer=optimizer, epochs=200, zeros=False)

printmat('predictions', polynomial_model.predict(X))

polynomial_model.save('test')
