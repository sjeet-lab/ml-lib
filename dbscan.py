"""
  ____        _   _____   _____   _____ 
 / ___|      | | | ____| | ____| |_   _|
 \___ \   _  | | |  _|   |  _|     | |  
  ___) | | |_| | | |___  | |___    | |  
 |____/   \___/  |_____| |_____|   |_|  
                                        
"""

class DBSCAN:
    """
    Example:
    [In]: X = [(3,3), (20,20), (21,25), (-5,1), (1,1), (2,2),(10,1) ]
          eps = 6
          min_points = 3
          dbs = DBSCAN(6,3)
          dbs.fit(X)
    [Out]:(1, [0, -1, -1, 0, 0, 0, -1])
    
    [In]: dbs.n_clusters
    [Out]:1
    [In]: dbs.labels
    [Out]:[0, -1, -1, 0, 0, 0, -1]
    """
    def __init__(self,eps = 0.1, min_points = 3):
        self.eps = eps
        self.min_points = min_points
    
    def fit(self,X):
        from scipy.spatial import distance
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import connected_components
        
        dist = []
        self.labels = []
        self.n_clusters = 0
        counter = []
        
        for i in X:
            d = []
            for j in X:
                if distance.euclidean(i,j) <= self.eps:
                    d.append(1)
                else:
                    d.append(0)
            dist.append(d)
        
        graph = csr_matrix(dist)
        init_comps = connected_components(csgraph=graph, directed=False, return_labels=True)
        lbl = init_comps[1]
        
        labels_count = {i:list(lbl).count(i) for i in list(lbl)}

        for i in lbl:
            if labels_count[i] < self.min_points:
                self.labels.append(-1)
            else:
                self.labels.append(i)
        
        for i in self.labels:
            if i != -1 and i not in counter:
                counter.append(i)
        
        self.n_clusters = len(counter)
        return (self.n_clusters,self.labels)
    
    def __str__(self):
        return 'eps = {}, min_points = {}'.format(self.eps,self.min_points)