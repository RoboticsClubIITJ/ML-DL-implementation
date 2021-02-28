import numpy as np
from MLlib.utils.misc_utils import read_data, printmat

class Numerical_outliers: 
    def get_percentile(c,percentile_rank): 
        """
           get_percentile Function 

           PARAMETER
           =========
           c:ndarray(dtype=float,ndim=1)
             input dataset
           percentile_rank: float type
           
           RETURNS
           =======
           Data corresponding to percentile rank
           """
        
        
        d=np.sort(c)
        index=int(((len(d)-1)*percentile_rank)//100)
        return d[index]
    
    def get_outliers(x):
        
        """ get_outliers Function
    
           PARAMETER
           =========
    
        x:ndarray(dtype=float,ndim=1)
            input dataset
         """
    
        d=np.sort(x)
        e=np.median(d)
        Q1= Numerical_outliers.get_percentile(x,25)
        Q3= Numerical_outliers.get_percentile(x,75)
        iqr=Q3-Q1
        lowerbound=Q1-1.5*iqr
        upperbound=Q3+1.5*iqr
        for i in range(len(x)):
             if x[i]>upperbound or x[i]<lowerbound:
                print("outlier=",x[i])

x,y=read_data("datasets/numerical_outliers.txt")

Numerical_outliers.get_outliers(y[0])