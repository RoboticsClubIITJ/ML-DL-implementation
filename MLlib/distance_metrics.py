import numpy as np
class Distance_metrics:
    """ 
    Calculate distance between each corresponding points
    of two arrays using different distance metrics
    """
    def Euclidean_Distance(X1,X2):
        """"
        Returns the list of euclidean distance
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
        Returns the list of euclidean distance
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

    def Hamming_Distance(X1,X2):
        """
        Returns the Hamming distance between
        two binary arrays

        PARAMETERS
        ==========
        X1:ndarray(dtype=int, axis=1)
        input array with more than 1 dimension

        X2:ndarray(dtype=int, axis=1)
        input array with more than 1 dimension

        RETURNS
        =======
        distance:float
        Returns the Hamming distance between
        two binary arrays
        """
        s = 0
        for e1,e2 in zip(X1,X2):
            s += abs(e1-e2)
        distance = s/len(X1)
        return distance

    def sEuclidean_distance(X1,X2,V):
        """
        Returns the list of standardized euclidean distance
        between two corresponding points of 
        two arrays

        PARAMETERS
        ==========
        X1:ndarray(dtype=int,axis=1)
        input array with more than 1 dimension

        X2:ndarray(dtype=int,axis=1)
        input array with more than 1 dimension

        V:list
        input array with 1 dimension

        RETURNS
        =========

        distance:list
        Returns the list of standardized euclidean distance
        between two corresponding points of 
        two arrays
        """
        distance=[]
        for i in range(len(X1)):
            single=0
            single=np.sum(((X1[i]-X2[i])/V[i])**2)
            distance.append(np.sqrt(single))
        return(distance)

    def Mahalanobis_Distance(X,d,V=None):
        """
        Returns the mahalanobis distance between 
        points and distribution

        PARAMETERS
        ==========
        X:ndarray(dtype=int,axis=1)
        input array with more than 1 dimension.
        Represents the points

        d:ndarray(dtype=int,axis=1)
        input array with more than 1 dimension.Represent 
        the distribution from which Mahalanobis 
        distance is to be calculated

        V:ndarray(dtype=float64,axis=1)
        input array with more than 1 dimension.Represent 
        the covariance matrix.If None is given,then will
         be computed from the data

        RETURNS
        =========
        distance:list
        Returns the list of mahalanobis distance
        between points and given distribution
        """
        distance=[]
        for i in range(len(X)):
            x_minus_mu = X[i]-np.mean(d,axis=0)
            if V==None:
                V=np.cov(d.T)
            VI = np.linalg.inv(V)
            d = np.sqrt(np.dot(np.dot(x_minus_mu,VI),x_minus_mu.T))
            distance.append(d)
        return(distance)
    
    def __boolean_opr(self,X1,X2):
        """
        Returns result of some bianry operation
        between two arrays.Any non zero value is 
        considered as 1

        PARAMETERS
        ==========
        X1:ndarray(dtype=int,axis=1)
        input array with 1 dimension

        X2:ndarray(dtype=int,axis=1)
        input array with 1 dimension

        RETURNS
        =========
        result:tuple

        result[0]:Number of dimensions
        result[1]:Number of dims in which both values are True
        result[2]:Number of dims in which the first value is True and second is False
        result[3]:Number of dims in which the first value is False and second is True
        result[4]:Number of dims in which both values are False
        """
        if len(X1)!=len(X2):
            raise TypeError("X1 and X2 must have same length")
        result=[]
        for i in range(len(X1)):
            if X1[i]!=0:
                X1[i]=1
            if X2[i]!=0:
                X2[i]=1
        result.append(len(X1))
        result.append(np.sum((X1==1)&(X2==1)))
        result.append(np.sum((X1==1)&(X2==0)))
        result.append(np.sum((X1==0)&(X2==1)))
        result.append(np.sum((X1==0)&(X2==0)))
        return(tuple(result))


    def Jaccard_Distance(self,X1,X2):
        """
        Returns the list of Jaccard distance between
        two corresponding vectors of two binary arrays
        
        PARAMETERS
        ==========
        X1:ndarray(dtype=int,axis=1)
        input array with 1 dimension

        X2:ndarray(dtype=int,axis=1)
        input array with 1 dimension

        RETURNS
        =========
        distance:list
        Returns the list of Jaccard distance
        """
        distance=[]
        for i in range(len(X1)):
            result=self.__boolean_opr(X1[i],X2[i])
            distance.append((result[2]+result[3])/(result[1]+result[2]+result[3]))
        return(distance)        
    
    def Matching_Distance(self,X1,X2):
        """
        Returns the list of Matching distance between
        two corresponding vectors of two binary arrays
        
        PARAMETERS
        ==========
        X1:ndarray(dtype=int,axis=1)
        input array with 1 dimension

        X2:ndarray(dtype=int,axis=1)
        input array with 1 dimension

        RETURNS
        =========
        distance:list
        Returns the list of Matching distance
        """
        distance=[]
        for i in range(len(X1)):
            result=self.__boolean_opr(X1[i],X2[i])
            distance.append((result[2]+result[3])/result[0])
        return(distance)

    def Dice_Distance(self,X1,X2):
        """
        Returns the list of Dice distance between
        two corresponding vectors of two binary arrays
        
        PARAMETERS
        ==========
        X1:ndarray(dtype=int,axis=1)
        input array with 1 dimension

        X2:ndarray(dtype=int,axis=1)
        input array with 1 dimension

        RETURNS
        =========
        distance:list
        Returns the list of Dice distance
        """
        distance=[]
        for i in range(len(X1)):
            result=self.__boolean_opr(X1[i],X2[i])
            distance.append((result[2]+result[3])/(2*result[1]+result[2]+result[3]))
        return(distance)
    
    def Kulsinki_Distance(self,X1,X2):
        """
        Returns the list of Kulsinki distance between
        two corresponding vectors of two binary arrays
        
        PARAMETERS
        ==========
        X1:ndarray(dtype=int,axis=1)
        input array with 1 dimension

        X2:ndarray(dtype=int,axis=1)
        input array with 1 dimension

        RETURNS
        =========
        distance:list
        Returns the list of Kulsinki distance
        """
        distance=[]
        for i in range(len(X1)):
            result=self.__boolean_opr(X1[i],X2[i])
            distance.append(result[2]+result[3]+result[0]-result[1])/(result[2]+result[3]+result[0])
        return(distance)
    
    def Rogers_Tanimoto_Distance(self,X1,X2):
        """
        Returns the list of Rogers-Tanimoto distance between
        two corresponding vectors of two binary arrays
        
        PARAMETERS
        ==========
        X1:ndarray(dtype=int,axis=1)
        input array with 1 dimension

        X2:ndarray(dtype=int,axis=1)
        input array with 1 dimension

        RETURNS
        =========
        distance:list
        Returns the list of Rogers-Tanimoto distance
        """
        distance=[]
        for i in range(len(X1)):
            result=self.__boolean_opr(X1[i],X2[i])
            distance.append(2*(result[2]+result[3])/(result[2]+result[3]+result[0]))
        return(distance)

    def Russell_Rao_Distance(self,X1,X2):
        """
        Returns the list of Russell-Rao distance between
        two corresponding vectors of two binary arrays
        
        PARAMETERS
        ==========
        X1:ndarray(dtype=int,axis=1)
        input array with 1 dimension

        X2:ndarray(dtype=int,axis=1)
        input array with 1 dimension

        RETURNS
        =========
        distance:list
        Returns the list of Russell-Rao distance
        """
        distance=[]
        for i in range(len(X1)):
            result=self.__boolean_opr(X1[i],X2[i])
            distance.append((result[0]-result[1])/result[0])
        return(distance)

    def Sokal_Michener_Distance(self,X1,X2):
        """
        Returns the list of Sokal-Michener distance between
        two corresponding vectors of two binary arrays
        
        PARAMETERS
        ==========
        X1:ndarray(dtype=int,axis=1)
        input array with 1 dimension

        X2:ndarray(dtype=int,axis=1)
        input array with 1 dimension

        RETURNS
        =========
        distance:list
        Returns the list of Sokal-Michener distance
        """
        distance=[]
        for i in range(len(X1)):
            result=self.__boolean_opr(X1[i],X2[i])
            distance.append(2*(result[2]+result[3])/(result[0]+result[2]+result[3]))
        return(distance)

    def Sokal_Sneath_Distance(self,X1,X2):
        """
        Returns the list of Sokal-Sneath distance between
        two corresponding vectors of two binary arrays
        
        PARAMETERS
        ==========
        X1:ndarray(dtype=int,axis=1)
        input array with 1 dimension

        X2:ndarray(dtype=int,axis=1)
        input array with 1 dimension

        RETURNS
        =========
        distance:list
        Returns the list of Sokal-Sneath distance
        """
        distance=[]
        for i in range(len(X1)):
            result=self.__boolean_opr(X1[i],X2[i])
            distance.append((result[2]+result[3])/(0.5*result[1]+result[2]+result[3]))
        return(distance)
