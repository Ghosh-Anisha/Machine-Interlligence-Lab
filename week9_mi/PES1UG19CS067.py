import numpy as np

class KMeansClustering:
    """
    K-Means Clustering Model

    Args:
        n_clusters: Number of clusters(int)
    """

    def __init__(self, n_clusters, n_init=10, max_iter=1000, delta=0.001):

        self.n_cluster = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.delta = delta

    def init_centroids(self, data):
        idx = np.random.choice(
            data.shape[0], size=self.n_cluster, replace=False)
        self.centroids = np.copy(data[idx, :])

    def fit(self, data):
        """
        Fit the model to the training dataset.
        Args:
            data: M x D Matrix(M data points with D attributes each)(numpy float)
        Returns:
            The object itself
        """
        if data.shape[0] < self.n_cluster:
            raise ValueError(
                'Number of clusters is grater than number of datapoints')

        best_centroids = None
        m_score = float('inf')

        for _ in range(self.n_init):
            self.init_centroids(data)

            for _ in range(self.max_iter):
                cluster_assign = self.e_step(data)
                old_centroid = np.copy(self.centroids)
                self.m_step(data, cluster_assign)

                if np.abs(old_centroid - self.centroids).sum() < self.delta:
                    break

            cur_score = self.evaluate(data)

            if cur_score < m_score:
                m_score = cur_score
                best_centroids = np.copy(self.centroids)

        self.centroids = best_centroids

        return self

    def e_step(self, data):
        """
        Expectation Step.
        Finding the cluster assignments of all the points in the data passed
        based on the current centroids
        Args:
            data: M x D Matrix (M training samples with D attributes each)(numpy float)
        Returns:
            Cluster assignment of all the samples in the training data
            (M) Vector (M number of samples in the train dataset)(numpy int)
        """
        #TODO
        cluster_assignment_067=[]
        row=data.shape[0]
        for i in range(row):
            mindist_067=2147483647
            index_067=0
            for j in range(self.n_cluster):
                dist_067=np.linalg.norm(data[i]-self.centroids[j])
                if(dist_067<mindist_067):
                    mindist_067=dist_067
                    index_067=j
            cluster_assignment_067.append(index_067)
        return cluster_assignment_067

    def zero_cluster(self,k):
        init_cluster=[]
        for i in range(k):
            init_cluster.append(0)
        return init_cluster

    def m_step(self, data, cluster_assgn):
        """
        Maximization Step.
        Compute the centroids
        Args:
            data: M x D Matrix(M training samples with D attributes each)(numpy float)
            cluster_assign: Cluster Assignment
        Change self.centroids
        """
        #TODO
        centroid_assignment_067=[]
        row=data.shape[0]
        col=data.shape[1]
        for i in range(self.n_cluster):
            sum_cen_067=self.zero_cluster(col)
            count=0
            for j in range(row):
                if i==cluster_assgn[j]:
                    sum_cen_067=np.add(data[j],sum_cen_067)
                    count+=1
            sum_cen_067=sum_cen_067/count
            centroid_assignment_067.append(sum_cen_067)
        self.centroids=centroid_assignment_067


    def evaluate(self, data, cluster_assign):
        """
        K-Means Objective
        Args:
            data: Test data (M x D) matrix (numpy float)
            cluster_assign: M vector, Cluster assignment of all the samples in `data`
        Returns:
            metric : (float.)
        """
        #TODO
        row=data.shape[0]
        diff_067=0
        for i in range(self.n_cluster):
            for j in range(row):
                if(i==cluster_assign[j]):
                    temp=np.linalg.norm(data[j]-self.centroids[i])
                    temp_sq=np.square(temp)
                    diff_067=diff_067+temp_sq
            
        return diff_067
