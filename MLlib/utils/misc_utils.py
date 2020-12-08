import numpy as np
import pickle


def read_data(file):
    """
    Read the training data from a file in the specified directory.

    Parameters
    ==========
    file:
        data type : str
        Name of the file to be read with extension.

    Example
    =======

    If the training data is stored in "dataset.txt" use

    >>> read_data('dataset.txt')

    """
    A = np.genfromtxt(file)

    # Read all the column except the last into X
    # Add an extra column of 1s at the end to act as constant
    # Read the last column into Y
    X = np.hstack((A[:, 1:2], np.ones((A.shape[0], 1))))
    M, N = X.shape

    Y = A[:, -1]
    Y.shape = (1, M)

    return X, Y


def printmat(name, matrix):
    """
    Prints matrix in a easy to read form with
    dimension and label.

    Parameters
    ==========
    name:
        data type : str
        The name displayed in the output.

    matrix:
        data type : numpy array
        The matrix to be displayed.
    """
    print("matrix " + name + ":", matrix.shape)
    print(matrix, "\n")


def generate_weights(rows, cols, zeros=False):
    """
    Generates a Matrix of weights according to the
    specified rows and columns
    """
    if zeros:
        return np.zeros((rows, cols))
    else:
        return np.random.rand(rows, cols)


def load_model(name):
    with open(name, "rb") as robfile:
        model = pickle.load(robfile)

    return model


class OneHotEncoder():
    """
    FUNCTIONS
    (1) FIT(INPUT_X, THRESHOLD)
    (2) CHECK_TRANSFORM(INPUT_X)
    (3) TRANSFORM(INPUT_X)
    (4) FIT_TRANSFORM(INPUT_X, THRESHOLD)

    INPUTS
     X - It is a numpy array of size n x m.
     thresh - It is a threshold value which is calulated as
        THRESH = (NUMBER OF UNIQUE VALUES IN A COLUMN)/(LENNGTH OF COLUMN).
        Column whose threshold value is below the input
        threshold value which be encode otherwise not.

    VARIABLES
     ncols - It is used to store the number of columns in the fit data.
     arr_dic - Is is an array of dictionary, where each dictionary
        is the LabelEncoded value of a particular column.
     arr_nunique - It is an array which  is used to store
        the number of unique values in a particular column.
     encode - It is an array of the size of the number of columns
        in the fit data.
        It has a value of 1 if the columns is to be encoded otherwise 0.
    """

    def fit(self, X, thresh):
        """
        FIT(INPUT_X, THRESHOLD) --- It is used to calculate the
            number of unique values in each column and tell
            whether a particular column should be encoded or not.
        """
        n = X.shape[0]
        m = X.shape[1]

        self.ncols = m
        self.arr_dic = []
        self.arr_nunique = []
        self.encode = []
        for i in range(m):
            ls = np.unique(X[:, i])
            n_unique = len(ls)
            dic = dict()

            if n_unique/n > thresh:
                self.encode.append(0)

            else:
                self.encode.append(1)
                for j, val in enumerate(ls):
                    dic[val] = j

            self.arr_dic.append(dic)
            self.arr_nunique.append(n_unique)

    def check_transform(self, X):
        """
        CHECK_TRANSFORM(INPUT_X) --- It is used to check whether the data which
            is being transformed has same values as the data which was
            used to fit it.
        """
        m = X.shape[1]

        if self.ncols != m:
            print("Number of Columns in input data of fit function and', \
                    'transform function are different")
            return False

        for i in range(m):
            ls = np.unique(X[:, i])
            n_unique = len(ls)

            if self.arr_nunique[i] != n_unique:
                print('Mismatch in the number of unique values in',
                      'the '+str(i)+'th column')
                return False
            for val in ls:
                if val not in self.arr_dic[i].keys():
                    print(str(i)+'th column contain a value which was',
                          'not in the data used to fit data')
                    return False

        return True

    def transform(self, X):
        """
        TRANSFORM(INPUT_X) --- It is used to OneHotEncode the data
            based on the data which was used to fit
        """
        check = self.check_transform(X)
        if not check:
            return None

        n = X.shape[0]
        m = X.shape[1]

        self.out_x = []
        for i in range(m):
            if self.encode[i] == 0:
                self.out_X.append(X[:, i].reshape(n, 1))
                continue
            n_unique = self.arr_nunique[i]
            dic = self.arr_dic[i]
            col = np.zeros((n, n_unique))
            enc_col = [dic[x] for x in X[:, i]]
            col[np.arange(n), enc_col] = 1
            self.out_x.append(col)

        self.out_x = np.concatenate(self.out_x, axis=1)
        return self.out_x

    def fit_transform(self, X, thresh):
        """
        FIT_TRANSFORM(INPUT_X, THRESHOLD) --- This fuction is just a
            combination of the fit and the transform fuction.
        """
        self.fit(X, thresh)
        return self.transform(X)
