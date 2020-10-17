from MLlib.models import LogisticRegression
from MLlib.optimizers import Adam
from MLlib.loss_func import LogarithmicError
from MLlib.utils.misc_utils import read_data, printmat


X, Y = read_data('datasets/logistic_reg_00.txt')

linear_model = LogisticRegression()

optimizer = Adam(0.03, LogarithmicError)

linear_model.fit(X, Y, optimizer=optimizer, epochs=200, zeros=False)

printmat('predictions', linear_model.predict(X))

linear_model.save('test')
