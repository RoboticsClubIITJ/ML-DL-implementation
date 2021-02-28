
from MLlib.utils.misc_utils import read_data, printmat
from MLlib.models import Numerical_outliers

x,y=read_data("datasets/numerical_outliers.txt")

Numerical_outliers.get_outliers(y[0])