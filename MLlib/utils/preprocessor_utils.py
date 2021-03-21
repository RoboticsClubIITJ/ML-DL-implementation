import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Feature_Scaling():
    """
            Scaling and Normalizing the dataset
            PARAMETERS
            ==========

            X: ndarray(dtype=float,ndim=1)
               1-D Array of Dataset's Input.

            csv_file: input dataset in csv fileformat

            fea1: Feature which is to be normalized

            RETURNS
            =======
            Scaled  and Normalized value of feature.
            """

    def __init__(self, X, csv_file, fea1):
        self.csv_file = csv_file
        self.X = X
        self.fea1 = fea1

    def Bell_curve(self, csv_file, fea1):
        """
                Plotting Density Plot
                PARAMETERS
                ==========

                X: ndarray(dtype=float,ndim=1)
                   1-D Array of Dataset's Input.
                csv_file: input dataset in csv file format

                fea1: Feature ehich is to be normalized

                RETURNS
                =======
                Gaussian bell curve of the feature.
                """

        fea1 = self.fea1
        csv_file = self.csv_file
        df = pd.read_csv(csv_file)
        x = df[fea1]
        x.plot(kind='density', subplots=True, layout=(3, 3))
        plt.show()

    def Standard_Scaler(self, X):
        """
        Data scaling by Standard Scaler.
        PARAMETERS
        ==========

        X: ndarray(dtype=float,ndim=1)
           1-D Array of Dataset's Input.

        RETURNS
        =======
        Scaled value of feature.
        """
        # X_new = (X - mean) / standerd deviation
        X = self.X
        for i in range(len(X)):
            m = np.mean(X)
            X = X - m
        X = np.divide(X, np.std(X))
        return X

    def MaxAbs_Scaler(self):
        """
        Data scaling by Max-Abs Scaler.
        PARAMETERS
        ==========

        X: ndarray(dtype=float,ndim=1)
           1-D Array of Dataset's Input.

        RETURNS
        =======
        Scaled value of feature.
        """
        # x= X/ Absolute(max(x))
        X = self.X
        for i in range(len(X)):
            k = abs(max(X))
        X = np.divide(X, k)
        return X

    def Feature_Clipping(self, max, min):
        """
        Data scaling by Feature Clipping.
        PARAMETERS
        ==========

        X: ndarray(dtype=float,ndim=1)
           1-D Array of Dataset's Input.
        max: random maximum value taken from user
        min: random minimum value taken from user

        RETURNS
        =======
        Scaled value of feature.
        """
        X = self.X
        for i in range(len(X)):
            if X[i] < min:
                X[i] == min
            if X[i] > max:
                X[i] == max
        return X

    def Z_Score_Normalization(self):
        """
        Data scaling by Z-Score Normalization.
        PARAMETERS
        ==========

        X: ndarray(dtype=float,ndim=1)
           1-D Array of Dataset's Input.

        RETURNS
        =======
        Normalized value of feature.
        """
        # X_new =( X - mean )/standard deviation
        X = self.X
        for i in range(len(X)):
            Mean = np.mean(X)
            Std = np.std(X)
            X[i] = (X[i] - Mean)
        X = np.divide(X, Std)
        return X

    def Mean_Normalization(self):
        """
        Data scaling by Mean Normalization.
        PARAMETERS
        ==========

        X: ndarray(dtype=float,ndim=1)
           1-D Array of Dataset's Input.

        RETURNS
        =======
        Normalized value of feature.
        """
        # X_new = (X - avg ) / (max - min)
        X = self.X
        for i in range(len(X)):
            Min = min(X)
            Max = max(X)
            Avg = np.average(X)
            X = (X - Avg)
        X = np.divide(X, (Max - Min))
        return X

    def MinMax_Normalization(self, new_min=0, new_max=1):
        """
        Data scaling by Min-Max Normalization.
        PARAMETERS
        ==========

        X: ndarray(dtype=float,ndim=1)
           1-D Array of Dataset's Input.
        new_max: random maximum value taken from the user
        new_min: random minimum value taken from the user

        RETURNS
        =======
        Normalized value of feature.
      """
        # X_new = (((X - min)/(max - min)) * (new_max - new_min)) + new_min
        X = self.X
        for i in range(len(X)):
            Min = min(X)
            Max = max(X)
            print(Min, Max)
            X = X - Min
        X = np.divide(X, (Max - Min))
        X = (X * (new_max - new_min)) + new_min
        return X
