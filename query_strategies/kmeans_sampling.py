import numpy as np
from active_learning import ActiveLearning
from sklearn.cluster import KMeans


class KMeansSampling(ActiveLearning):
    """
    This class is for K-Means sampling using the embedded features sampling for active learning. This method uses the
    trained model to extract features for the data. These features are then clustered using the K-Means clustering
    method and the centers of the clusters are then selected to be annotated.
    """

    def query(self, n):
        """
        Method for querying the data to be labeled. This method selects data that are centers of clusters specified by
        the K-Means clustering method.
        :param n: Amount of data to query.
        :return: Array of indices to data selected to be labeled.
        """

        unlabeled_indices = np.arange(self.pool_size)[~self.labeled_data]
        embeddings = self.get_embedings(self.train_images[unlabeled_indices], self.train_labels[unlabeled_indices])
        print(len(embeddings[0]))
        cluster_learner = KMeans(n_clusters=n, n_jobs=-1)
        cluster_learner.fit(embeddings[0])
        cluster_indices = cluster_learner.predict(embeddings[0])
        centers = cluster_learner.cluster_centers_[cluster_indices]
        distances = (embeddings[0] - centers) ** 2
        distances = distances.sum(axis=1)
        query_indices = np.array([np.arange(len(embeddings[0]))[cluster_indices == i][distances[cluster_indices == i].argmin()] for i in range(n)])

        return unlabeled_indices[query_indices]
