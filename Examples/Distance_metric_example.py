from MLlib.distance_metrics import Distance_metrics
import numpy as np

data = np.genfromtxt('dataset/salaryinp.csv', delimiter=',')
X1 = np.array(data[:len(data)//2])
X2 = np.array(data[len(data)//2:])


Euc = Distance_metrics.Euclidean_Distance(X1=X1, X2=X2)
print("Euclidean Distance: ", Euc)

Mah = Distance_metrics.Manhattan_Distance(X1=X1, X2=X2)
print("Manhattan Distance: ", Mah)

Che = Distance_metrics.Chebyshev_Distance(X1=X1, X2=X2)
print("Chebyshev Distance: ", Che)

# order will be as per user's requirement
Mink = Distance_metrics.Minkowski_Distance(X1=X1, X2=X2, p=3)
print("Minkowski Distance: ", Mink)

# order and weight will be as per user's requirement
WMink = Distance_metrics.WMinkowski_Distance(X1=X1, X2=X2, p=3, w=0.5)
print("Weighted Minkowski Distance: ", WMink)

# Variance to be provided by user
V = np.random.randint(0, 10, size=(len(X1), 1))
SEuc = Distance_metrics.sEuclidean_distance(X1=X1, X2=X2, V=V)
print("Standardized Euclidean Distance: ", SEuc)

Maha = Distance_metrics.Mahalanobis_distance(X1=X1, X2=X2)
print("Mahalanobis Distance: ", Maha)

# Generating Boolean data
# User can generate boolean data as per requirement
X1 = np.random.randint(0, 2, size=(100, 5))
X2 = np.random.randint(0, 2, size=(100, 5))

Ham = Distance_metrics.Hamming_Distance(X1=X1, X2=X2)
print("Hamming Distance: ", Ham)

Dist_met = Distance_metrics()

Jacc = Dist_met.Jaccard_Distance(X1=X1, X2=X2)
print("Jaccard Distance: ", Jacc)

Match_dis = Dist_met.Matching_Distance(X1=X1, X2=X2)
print("Matching Distance: ", Match_dis)

Dice = Dist_met.Dice_Distance(X1=X1, X2=X2)
print("Dice Distance: ", Dice)

Kuls = Dist_met.Kulsinki_Distance(X1=X1, X2=X2)
print("Kulsinki Distance: ", Kuls)

Rog = Dist_met.Rogers_Tanimoto_Distance(X1=X1, X2=X2)
print("Rogers Tanimoto Distance: ", Rog)

Rus = Dist_met.Russell_Rao_Distance(X1=X1, X2=X2)
print("Russell Rao Distance: ", Rus)

Sok = Dist_met.Sokal_Sneath_Distance(X1=X1, X2=X2)
print("Sokal Sneath Distance: ", Sok)

Sok_M = Dist_met.Sokal_Michener_Distance(X1=X1, X2=X2)
print("Sokal Michener Distance: ", Sok_M)
