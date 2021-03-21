from MLlib.utils.preprocessor_utils import Feature_Scaling
import numpy as np

preprocessor = np.genfromtxt('datasets/salaryinp.csv', delimiter=',')
X = preprocessor[1:, 1]
print(X)
Scale = Feature_Scaling(X, 'datasets/salaryinp.csv', 'Salary')
print(Scale.Standard_Scaler())
print(Scale.MaxAbs_Scaler())
print(Scale.Feature_Clipping(100000, 10000))
print(Scale.Mean_Normalization())
print(Scale.MinMax_Normalization(new_min=0, new_max=1))
Scale.Bell_curve('datasets/salaryinp.csv', 'Salary')

