import numpy as np
import matplotlib.pyplot as plt 
import math
from time import perf_counter
import logging
logging.basicConfig(filename='MLlib/tests/divisive.log', level=logging.DEBUG, filemode='w', format='\n%(asctime)s\n%(message)s')

class KMeans:
    def runKMeans(self, X: np.ndarray, n_clusters: int, n_iterations: int) -> (list, np.array):
        '''
        The KMeans clustering algorithm.
        Returns:
        clusters: list of np.ndarrays of clusters.
        centroids: np.array of size = n_clusters
        '''
        self.n_clusters = n_clusters
        self.init_centroids(X)
        for i in range(n_iterations):
            self.allocate(X)
            self.update_centroids()
        # logging.debug(f'KMeans results after allocate:\nX:{X}\nclusters:{self.clusters}\nlen:{len(self.clusters)}\nlen1:{len(self.clusters[0])}\nlen2:{len(self.clusters[1])}')
        return self.clusters, self.centroids
    
    def init_centroids(self, X: np.ndarray):
        '''
        Initialize centroids with random examples (or points) from the dataset.
        '''
        #Number of examples
        l = X.shape[0]
        #Initialize centroids array with points from X with random indices chosen from 0 to number of examples
        rng = np.random.default_rng()
        self.centroids = X[rng.choice(l, size=self.n_clusters, replace=False)]
        # self.centroids = X[np.random.randint(0, l, size=self.n_clusters)]
        self.centroids.astype(np.float32)

    
    def allocate(self, X: np.ndarray):
        '''
        This function forms new clusters from the centroids updated in the previous iterations.
        '''
        #Step 1: Fill new clusters with a single point
        #Calculate the differences in the features between X and centroids using broadcast subtract 
        res = X - self.centroids[:, np.newaxis]
        # logging.debug(res.shape)    #(n_clusters, X.shape[0], X.shape[1])
        
        #Find Euclidean distances using the above differences
        euc = np.linalg.norm(res, axis = 2)
        # euc = np.array([[1,1,3],[4,2,3],[6,7,8]], dtype=np.float32)
        m, n = euc.shape
        #Add the closest point to the corresponding centroid to the cluster array.
        #We do this to avoid formation of empty clusters
        # res = np.where(euc == euc.min(axis=1)[:, np.newaxis])  #indices of the first entered points
        first_indices = np.full((m,), -1, dtype=int)

        while True:
            res = np.argmin(euc, axis=1)
            res.astype(int)
            resu = np.unique(res)
            l = len(res)
            lu = len(resu)
            if lu == l:
                first_indices = res
                break
            else:
                arr = []
                for i in range(l): #!FIXME:changed from lu to l
                    # logging.debug(f'arr entries: {np.where(res==i)}, i:{i}')
                    arr.append(np.where(res==i)[0])    #!DOUBTFUL
                # logging.debug(f'euc:{euc}\nres:{res}\narr:{arr}')
                for i in range(l): #!FIXME:changed from n to lu to l
                    if len(arr[i]) == 1:
                        first_indices[arr[i]] = i
                        # logging.debug(f'fi after equal:\nfi:{first_indices}, i:{i}')
                    elif len(arr[i]) > 1:
                        # logging.debug(f'euc entries for arr[i]={arr[i]} and i={i}\neuc[arr[i], i]:{euc[arr[i], i]}')
                        temp = np.argmin(euc[arr[i], i])    #!DOUBTFUL
                        first_indices[temp] = i
            # logging.debug(f'argmin:{np.argmin(euc, axis=1)}\neuc:{euc}\neuc.shape:{euc.shape}\nres(first_indices): {res}\nres2: {res2}\neuc[res]: {euc[res2]}\n')           
            if -1 in first_indices:
                # euc = euc[np.where(first_indices == -1)]
                col_ind = np.nonzero(np.isin(np.arange(n),first_indices))
                euc[:, col_ind[0]] = euc[col_ind[0], :] = np.inf
                m,n = euc.shape
                # logging.debug(f'euc in while:{euc}\nnp.isin(np.arange(n), first_indices):{ np.nonzero(np.isin(np.arange(n),first_indices))}')
            else:
                break
        # logging.debug(f'euc:{euc}\nfirst_indices on completion:{first_indices}')
        cluster_array = X[first_indices]
        cluster_array = list(np.expand_dims(cluster_array, axis=1))

        #Step 2: Allocate the remaining points to the closest clusters
        #Calculate the differences in the features between centroids and X using broadcast subtract 
        res = self.centroids - X[:, np.newaxis]
        # logging.debug(res.shape)    #(X.shape[0], n_clusters, X.shape[1])

        #Find Euclidean distances of each point with all centroids 
        euc = np.linalg.norm(res, axis=2)

        #Find the closest centroid from each point. 
        # Find unique indices of the closest points. Using res again for optimization
        #not unique indices
        res =  np.where(euc == euc.min(axis=1)[:, np.newaxis])  
        #res[0] is used as indices for row-wise indices in res[1]
        min_indices = res[1][np.unique(res[0])]   
        # logging.debug(f'len(min_indices)={len(min_indices)}')
        #Set first indices to -1 to avoid adding them
        min_indices[first_indices] = -1
        # logging.debug(f'len(min_indices)={len(min_indices)}')
        for i, c in enumerate(min_indices):
            if not c == -1:
                cluster_array[c] = np.append(cluster_array[c], [X[i]], axis=0)    #add the point to the corresponding cluster
        # if len(X) == 2 and (cluster_array[0].shape == (2,2) or cluster_array[1].shape == (2,2)):
        # logging.debug(f'first_indices: {first_indices}\nmin_indices: {min_indices}\ncentroids: {self.centroids}')
        #update the fair clusters array 
        self.clusters = cluster_array
    
    def update_centroids(self):
        '''
        This function updates the centroids based on the updated clusters.
        '''
        #Make a rough copy
        centroids = self.centroids
        #Find mean for every cluster
        for i in range(self.n_clusters):
            centroids[i] = np.mean(self.clusters[i], axis=0)
        
        #Update fair copy 
        self.centroids = centroids

def visualize_clusters(global_clusters, global_centroids, iteration):
    '''
    Utility to visualize the changes in clusters at every iteration of the work algorithm in models.py
    '''
    # centroids = np.array(centroids)
    fig, ax = plt.subplots()
    # ax.annotate(f'c{0}', (centroids[0,0], centroids[0,1]))
    # ax.scatter(clusters[0][:,0], clusters[0][:,1], color='blue')
    # ax.annotate(f'c{1}', (centroids[1,0], centroids[1,1]))
    # ax.scatter(clusters[1][:,0], clusters[1][:,1], color='green')
    # ax.scatter(centroids[:,0], centroids[:,1], color = 'red')

    # # ax.pause(2)
    # # ax.set_xlim((0, count))
    # # ax.set_ylim((0, count))
    # ax.set_xlabel('X axis')
    # ax.set_ylabel('Y axis')
    # ax.set_title(f'Iteration {iteration}')
    n_clusters = iteration + 1
    rng = np.random.default_rng()
    colors = rng.random(size=(n_clusters, 4), dtype=np.float32)
    colors[:, 3] = 0.5
    for i in range(n_clusters):
        ax.scatter(global_clusters[i][:,0], global_clusters[i][:,1], marker='.', color=tuple(colors[i]))
    ax.scatter(global_centroids[:,0], global_centroids[:,1], marker = 's', color = '#F008', label='Centroids')
    for i in range(n_clusters):
        ax.annotate(f'c{i}', (global_centroids[i,0], global_centroids[i,1]))
    # ax.set_pause(2)
    # ax.set_xlim((0, datasize))
    # ax.set_ylim((0, datasize))
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_title(f'Iteration {iteration}')
    # ax.set_legend()

def sse(cluster: np.array, centroid: np.array):
    return np.sum((cluster - centroid)**2)

def distcalc(p1, p2):
    """
    Calculates the Euclidean Distance
    between p1 and p2 points.

    PARAMETERS
    ==========

    p1: ndarray(dtype=int,ndim=1,axis=1)
        Point array with its corresponding
        x and y coordinates.

    p2: ndarray(dtype=int,ndim=1,axis=1)
        Point array with its corresponding
        x and y coordinates.

    dist: int
        Sum of sqaurred difference of
        coordinates between p1 and p2
        points.

    distance: float
        Euclidean Distance between p1
        and p2 points.

    RETURNS
    =======

    distance: float
        Euclidean Distance between p1
        and p2 points.

    """
    dist = 0
    for i in range(0, len(p1)-1):
        dist += (p1[i]-p2[i])**2
    distance = dist**0.5
    return distance

def to_adjacency_matrix(global_centroids: np.ndarray, n_clusters) -> np.ndarray:
    '''
    Creates an adjacency matrix of the distances of the result centroids.
    '''
    res = global_centroids - global_centroids[:, np.newaxis]
    centroid_dist_mat = np.linalg.norm(res, axis=2)
    np.fill_diagonal(centroid_dist_mat, np.nan)
    return centroid_dist_mat

def update_mat(centroid_dist_mat, n_clusters):

    #Find the index of the smallest value (other than np.nan) from the centroid distance matrix
    ind = np.unravel_index(np.nanargmin(centroid_dist_mat), (n_clusters, n_clusters))
    dist = centroid_dist_mat[ind]
    #ind is a tuple of x and y indices
    c = max(ind[0],ind[1])  
    o = min(ind[0],ind[1])
    for i in range(n_clusters):
        a = centroid_dist_mat[c, i]
        b = centroid_dist_mat[o, i]
        m = math.sqrt(0.5* (a*a + b*b - dist*dist*0.5))  #Appollonius theorem: used to find median length given the sides of the triangle
        centroid_dist_mat[o,i] = centroid_dist_mat[i,o] = m
    centroid_dist_mat[:,c] = centroid_dist_mat[c,:] = np.nan
    return centroid_dist_mat, ind, dist 

def create_label_map(locations, n_clusters):
    label_map = {}
    # last_free = n_clusters - 2
    xpos_map = np.array([False for i in range(n_clusters+1)])
    # logging.debug(f'xpos_map: {xpos_map}, shape: {xpos_map.shape}')
    last_free = n_clusters
    for tup in locations:
        a, b = tup
        mn = min(a, b)
        mx = max(a,b)
        if mn not in label_map and mx not in label_map: #new pair of clusters
            label_map[mn] = {'label':f'c{mn}'}
            while xpos_map[last_free]:
                last_free -= 1
            label_map[mn]['xpos'] = last_free
            xpos_map[last_free] = True
            label_map[mn]['ypos'] = 0
            last_free -= 1
            # while xpos_map[last_free]:
            #     last_free -= 1
            label_map[mx] = {'label':f'c{mx}'}
            label_map[mx]['xpos'] = last_free
            xpos_map[last_free] = True
            label_map[mx]['ypos'] = 0
        elif mn in label_map and mx not in label_map:   #min is present and max is not
            # logging.debug(f'label_map[mn]:{label_map[mn]}')
            label_map[mx] = {'label':f'c{mx}'}
            x = label_map[mn]['xpos']
            i = x - 1
            while xpos_map[i]:
                i -= 1
            label_map[mx]['xpos'] = i
            label_map[mx]['ypos'] = 0
            xpos_map[i] = True
        elif mx in label_map and mn not in label_map:   #max is present but min is not
            # logging.debug(f'label_map[mx]:{label_map[mx]}')
            label_map[mn] = {'label':f'c{mn}'}
            x = label_map[mx]['xpos']
            i = x - 1
            # logging.debug(f'i={i}, x={x}')
            while xpos_map[i] and i>0:
                i -= 1
            label_map[mn]['xpos'] = i
            label_map[mn]['ypos'] = 0
            xpos_map[i] = True
        # logging.debug(f'xpos_map: {xpos_map}')
    return label_map

def mk_fork(x0,x1,y0,y1,new_level):
    points=[[x0,x0,x1,x1],[y0,new_level,new_level,y1]]
    connector=[(x0+x1)/2.,new_level]
    return (points),connector

def visualize(global_clusters, global_centroids, n_clusters, datasize):
    rng = np.random.default_rng()
    colors = rng.random(size=(n_clusters, 4), dtype=np.float32)
    colors[:, 3] = 0.5
    for i in range(n_clusters):
        plt.scatter(global_clusters[i][:,0], global_clusters[i][:,1], marker='.', color=tuple(colors[i]))
    plt.scatter(global_centroids[:,0], global_centroids[:,1], marker = 's', color = '#F008', label='Centroids')
    for i in range(n_clusters):
        plt.annotate(f'c{i}', (global_centroids[i,0], global_centroids[i,1]))
    # plt.pause(2)
    plt.xlim((0, datasize))
    plt.ylim((0, datasize))
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    '''
    dendrogram using adjacency matrix of the distances of the centroids from each other
    '''
    locations = []
    levels = []
    start = perf_counter()
    centroid_dist_mat = to_adjacency_matrix(global_centroids, n_clusters)
    stop = perf_counter()
    logging.debug(f'Elapsed Time[to_adjacency_matrix]:{stop - start}')
    # logging.debug(f'centroid_dist_mat: \n{centroid_dist_mat}\n')
    start = perf_counter()
    # logging.debug('==========================================================')
    # logging.debug(f'centroid dist matrix initially: \n{centroid_dist_mat}')
        
    for i in range(n_clusters - 1):
        centroid_dist_mat, tup, dist = update_mat(centroid_dist_mat, n_clusters)
        # logging.debug('==========================================================')
        # logging.debug(f'updated matrix at {i}th iteration: \n{centroid_dist_mat}')
        locations.append(tup)
        levels.append(dist)
    stop = perf_counter()
    # logging.debug(f'Elapsed Time[update_mat with loop]:{stop - start}')
    # logging.debug(f'locations: {locations}')
    # logging.debug(f'levels: {levels}')
    label_map = create_label_map(locations, n_clusters)
    # logging.debug(label_map)
    #dendrogram code using locations and levels

    fig,ax=plt.subplots()

    for i,(new_level,(loc0,loc1)) in enumerate(zip(levels,locations)):

        # logging.debug('step {0}:\t connecting ({1},{2}) at level {3}'.format(i, loc0, loc1, new_level ))

        x0,y0=label_map[loc0]['xpos'],label_map[loc0]['ypos']
        x1,y1=label_map[loc1]['xpos'],label_map[loc1]['ypos']

        # logging.debug('\t points are: {0}:({2},{3}) and {1}:({4},{5})'.format(loc0,loc1,x0,y0,x1,y1))

        p,c=mk_fork(x0,x1,y0,y1,new_level)

        ax.plot(*p)
        ax.scatter(*c)

        # logging.debug('\t connector is at:{0}'.format(c))


        label_map[loc0]['xpos']=c[0]
        label_map[loc0]['ypos']=c[1]
        label_map[loc0]['label']='{0}/{1}'.format(label_map[loc0]['label'],label_map[loc1]['label'])
        # logging.debug('\t updating label_map[{0}]:{1}'.format(loc0,label_map[loc0]))

        # ax.text(*c,label_map[loc0]['label'])

    # _xticks=np.arange(0,n_clusters,1)
    # _xticklabels=['BA','NA','RM','FI','MI','TO']

    ax.set_xlim(0, 1.05*(n_clusters))
    ax.set_xlabel('Cluster Number')

    ax.set_ylim(0,1.05*np.max(levels))
    ax.set_ylabel('Distance')
        
    # plt.show()
