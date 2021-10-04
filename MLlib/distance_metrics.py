import numpy as np
class Distance_metrics:
    """ 
    Calculate distance between each corresponding points
    of two arrays using different distance metrics
    """
    def Eucledian_distance(self,X1,X2):
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

        distance:list
        list containing eucledian distance of points

        RETURNS
        =========

        float
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

    def Manhattan_distance(self,X1,X2):
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

        distance:list
        list containing manhattan distance of points

        RETURNS
        =========

        float
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

    def Chebyshev_Distance(self,X1,X2):
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

        distance:list
        list containing chebyshev distance of points

        RETURNS
        =========

        float
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



    

