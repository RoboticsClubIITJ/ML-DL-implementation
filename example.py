from MLlib.models import LogisticRegression
from MLlib.optimizers import GradientDescent
from MLlib.loss_func import LogarithmicError
from MLlib.utils import read_data, printmat
from MLlib.visualize import Visualize

X, Y = read_data('MLlib/datasets/logistic_reg_00.txt')

logistic_model = LogisticRegression()

optimizer = GradientDescent(0.05, LogarithmicError)

costs = logistic_model.fit(X, Y, optimizer=optimizer, epochs=200, zeros=False)

printmat('predictions', logistic_model.predict(X))

vis = Visualize(costs)
vis.cost_per_epoch()

logistic_model.save('test')
