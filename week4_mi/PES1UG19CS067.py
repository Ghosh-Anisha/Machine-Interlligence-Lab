import numpy as np
from collections import defaultdict


class KNN:
    """
    K Nearest Neighbours model
    Args:
        k_neigh: Number of neighbours to take for prediction
        weighted: Boolean flag to indicate if the nieghbours contribution
                  is weighted as an inverse of the distance measure
        p: Parameter of Minkowski distance
    """

    def __init__(self, k_neigh, weighted=False, p=2):

        self.weighted = weighted
        self.k_neigh = k_neigh
        self.p = p


    def fit(self, data, target):
        """
        Fit the model to the training dataset.
        Args:
            data: M x D Matrix( M data points with D attributes each)(float)
            target: Vector of length M (Target class for all the data points as int)
        Returns:
            The object itself
        """

        self.data = data
        self.target = target.astype(np.int64)

        return self

    def find_distance(self, x):
        """
        Find the Minkowski distance to all the points in the train dataset x
        Args:
            x: N x D Matrix (N inputs with D attributes each)(float)
        Returns:
            Distance between each input to every data point in the train dataset
            (N x M) Matrix (N Number of inputs, M number of samples in the train dataset)(float)
        """
        # TODO
        dist=[]
        for v in x:
            min_d=np.abs(v-self.data)**self.p
            min_d=min_d.sum(axis=1)
            min_d=min_d**(1/self.p)
            dist.append(min_d)
        return np.array(dist)
    
    def sort_distance(self,x):
        ans=[]
        dist=self.find_distance(x)
        for i in dist:
            co_ind_067=dict()
            index=0
            for j in i:
                co_ind_067[j]=index
                index+=1
            co_ind_067=sorted(co_ind_067.items())
            ans.append(co_ind_067)
        return ans

    def k_neighbours(self, x):
        """
        Find K nearest neighbours of each point in train dataset x
        Note that the point itself is not to be included in the set of k Nearest Neighbours
        Args:
            x: N x D Matrix( N inputs with D attributes each)(float)
        Returns:
            k nearest neighbours as a list of (neigh_dists_067, idx_of_neigh_067)
            neigh_dists_067 -> N x k Matrix(float) - Dist of all input points to its k closest neighbours.
            idx_of_neigh_067 -> N x k Matrix(int) - The (row index in the dataset) of the k closest neighbours of each input

            Note that each row of both neigh_dists_067 and idx_of_neigh_067 must be SORTED in increasing order of distance
        """
        # TODO
        ans=self.sort_distance(x)
        
        neigh_dists_067=[]
        idx_of_neigh_067=[]
        for i in ans:
            neigh_dist_temp_067=[]
            index=[]
            k=self.k_neigh
            for j in i:
                neigh_dist_temp_067.append(j[0])
                index.append(j[1])
                k=k-1
                if(k==0):
                    break
            neigh_dists_067.append(neigh_dist_temp_067)
            idx_of_neigh_067.append(index)
        
        return (neigh_dists_067,idx_of_neigh_067)

    def predict(self, x):
        """
        Predict the target value of the inputs.
        Args:
            x: N x D Matrix( N inputs with D attributes each)(float)
        Returns:
            pred: Vector of length N (Predicted target value for each input)(int)
        """
        # TODO
        if(self.weighted==False):
            ans=self.k_neighbours(x)[1]
            final_pred_067=[]
            for i in ans:
                pred=defaultdict (lambda: 0.)
                for j in i:
                    pred[self.target[j]]+=1
                max_value_067=max(pred, key=lambda x: pred[x])
                final_pred_067.append(max_value_067)
                pred.clear()
            return np.array(final_pred_067)

        dist,idx=self.k_neighbours(x)
        final_pred_067=[]
        for i in range(len(idx)):
            pred=defaultdict (lambda: 0.)
            for j in range(len(idx[i])):
                pred[self.target[idx[i][j]]]+=(1/(dist[i][j]+0.000000001))
            max_value_067=max(pred,key=lambda x:pred[x])
            final_pred_067.append(max_value_067)
            pred.clear()
        return np.array(final_pred_067)

    def evaluate(self, x, y):
        """
        Evaluate Model on test data using 
            classification: accuracy metric
        Args:
            x: Test data (N x D) matrix(float)
            y: True target of test data(int)
        Returns:
            accuracy_067 : (float.)
        """
        sum_067=0
        for i in range(len(x)):
            if(y[i]==self.predict(x)[i]):
                sum_067+=1
        accuracy_067=sum_067*100/len(x)
        return accuracy_067
        
        
