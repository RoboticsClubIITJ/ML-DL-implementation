import numpy as np


class matrix_evolution():

    """ Implement Score Metrics and find Confusion Matrix

    Attributes
    ==========
    None

    Methods
    =======

    confusion_matrix(y_true, y_pred)
    Finds true_positive,false_positive,false_negative,true_negative
    and make confusion_matrix

    score_metrics(y_true, y_pred)
    Finds common evaluation metrices like precision,recall,accuracy,F1
    simplicity and FbTheta

    """

    def confusion_matrix(y_true, y_pred):

        """ confusion_matrix Function

        For creating Confusion Matrix

        PARAMETERS
        ==========
        y_true: ndarray(dtype=float,ndim=1)
                1D array of True Values
        y_pred: ndarray(dtype=float,ndim=1)
                1D array of Predicted Values

        RETURNS
        =======
        conf_matrix:ndarray(dtype=float,ndim=2)
                     2D Confusion Matrix

        """
        stack = np.vstack((y_true, y_pred))
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        for i in range(len(stack[0])):
            if stack[0][i] == 1 and stack[1][i] == 1:
                tp += 1
            if stack[0][i] == 0 and stack[1][i] == 0:
                tn += 1
            if stack[0][i] == 0 and stack[1][i] == 1:
                fp += 1
            if stack[0][i] == 1 and stack[1][i] == 0:
                fn += 1
        conf_matrix = np.reshape(np.array([tp, fp, fn, tn]), (-1, 2))
        return conf_matrix

    def score_metrics(y_true, y_pred):

        """ Implements Score Metrics

        PARAMETERS
        ==========

        y_true: ndarray(dtype=float,ndim=1)
                1D array of True Values
        y_pred: ndarray(dtype=float,ndim=1)
                1D array of Predicted Values

        """

        m = matrix_evolution.confusion_matrix(y_true, y_pred)
        tp = m[0][0]
        fp = m[0][1]
        fn = m[1][0]
        tn = m[1][1]
        accuracy = (tp)/(tp + fp + tn + fn)
        precision = (tp)/(tp + fp)
        recall = (tp)/(tp + fn)
        p = precision
        r = recall
        f1 = (2 * p * r)/(p + r)
        specificity = (tn)/(tn + fp)
        b1 = 0.5
        fb1 = ((1 + b1**2)*(p * r))/(p * b1**2 + r)
        b2 = 2
        fb2 = ((1 + b2**2)*(p * r))/(p * b2**2 + r)
        print("Accuracy = ", accuracy)
        print("Precision = ", precision)
        print("Recall = ", recall)
        print("Specificity = ", specificity)
        print("F1 =", f1)
        print("FbTheta for 0.5 = ", fb1)
        print("FbTheta for 2.0 = ", fb2)
