import numpy as np
class Distance_metrics:
    """ 
    Calculate distance between each corresponding points
    of two arrays using different distance metrics
    """
    def Eucledian_Distance(X1,X2):
        """"
        Returns the list of eucledian distance
        between two corresponding points of 
        two arrays

        PARAMETERS
        ==========
        X1:ndarray(dtype=int,axis=1)
        input array with more than 1 dimension

        X2:ndarray(dtype=int,axis=1)
        input array with more than 1 dimension

        RETURNS
        =========

        distance:list
        Returns the list of eucledian distance
        between two corresponding points of 
        two arrays
        """
        distance=[]
        for i in range(len(X1)):
            single=0
            single=np.sum((X1[i]-X2[i])**2)
            distance.append(np.sqrt(single))
        return(distance)

    def Manhattan_Distance(X1,X2):
        """"
        Returns the list of manhattan distance
        between two corresponding points of 
        two arrays

        PARAMETERS
        ==========
        X1:ndarray(dtype=int,axis=1)
        input array with more than 1 dimension

        X2:ndarray(dtype=int,axis=1)
        input array with more than 1 dimension

        RETURNS
        =========

        distance:list
        Returns the list of manhattan distance
        between two corresponding points of 
        two arrays
        """
        distance=[]
        for i in range(len(X1)):
            single=0
            single=np.sum(abs(X1[i]-X2[i]))
            distance.append(single)
        return(distance)

    def Chebyshev_Distance(X1,X2):
        """"
        Returns the list of chebyshev distance
        between two corresponding points of 
        two arrays

        PARAMETERS
        ==========
        X1:ndarray(dtype=int,axis=1)
        input array with more than 1 dimension

        X2:ndarray(dtype=int,axis=1)
        input array with more than 1 dimension

        RETURNS
        =========

        distance:list
        Returns the list of chebyshev distance
        between two corresponding points of 
        two arrays
        """
        distance=[]
        for i in range(len(X1)):
            single=0
            single=np.sum(max(X1[i]-X2[i]))
            distance.append(single)
        return(distance)

    def Minkowski_Distance(X1,X2,p):
        """"
        Returns list of minkowski distance of order 'p'
        between two corresponding vectors of
        two arrays

        PARAMETERS
        ==========
        X1:ndarray(dtype=int,axis=1)
        input array with more than 1 dimension

        X2:ndarray(dtype=int,axis=1)
        input array with more than 1 dimension

        p:float
        input order value between 1 and 2 inclusive

        RETURNS
        =========

        distance:list
        Returns the list of minkowski distance
        between two corresponding vectors of 
        two arrays
        """
        distance=[]
        for i in range(len(X1)):
            single=0
            single=np.sum((abs(X1[i]-X2[i]))**p)
            distance.append((single)**(1/p))
        return(distance)
    
    def WMinkowski_Distance(X1,X2,p,W):
        """"
        Returns list of weighted minkowski distance of order 'p'
        between two corresponding vectors weighted by W of
        two arrays

        PARAMETERS
        ==========
        X1:ndarray(dtype=int,axis=1)
        input array with more than 1 dimension

        X2:ndarray(dtype=int,axis=1)
        input array with more than 1 dimension

        p:float
        input order value between 1 and 2 inclusive

        W:array(dtype=int,axis=1)
        input 1 dimensional array

        RETURNS
        =========

        distance:list
        Returns the list of weighted minkowski distance
        between two corresponding vectors of 
        two arrays
        """
        distance=[]
        for i in range(len(X1)):
            single=0
            single=np.sum((abs(W*(X1[i]-X2[i])))**p)
            distance.append((single)**(1/p))
        return(distance)