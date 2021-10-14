from MLlib.models import KMeansClustering
import numpy as np

X = np.genfromtxt('datasets/k_means_clustering.txt')

KMC = KMeansClustering()

na,ca=KMC.work(X, 3, 7,config=True)
KMC.plot(X,3,7 )


        



      
        



        
    
