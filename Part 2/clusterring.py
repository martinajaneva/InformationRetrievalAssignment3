import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

def cluster_embeddings(embeddings, k_clusters):
    """
    Clusterring the document embeddings up to k clusters using KMeans
    Args:
        embeddings (tensor): The document embeddings
        k_clusters (int): The number of clusters to be applied
    Returns:
        return (tuple): The cluster assignments together with their centroids
    """
    # Using KMeans for clusterring with a specified seed for reproducibility
    kmeans = KMeans(n_clusters=k_clusters, random_state=12345)
    kmeans.fit(embeddings.cpu().numpy())
    cluster_assignments = kmeans.labels_
    centroids = kmeans.cluster_centers_
    return cluster_assignments, centroids


def find_relevant_clusters(query_embedding, centroids, top_k=5):
    """
    Using cosine similarity to identify the similarity across the clusters based on their centroids
    and the embedded query
    Args:
        query_embedding (): The embedded query
        centroids (): The centroids of the clusters
        top_k (int): The number of top clusters in terms of similarity
    Returns:
        top_k_clusters (list): Contains the top k most similar clusters
    """
    centroid_similarity = cosine_similarity(query_embedding.cpu().numpy(), centroids)
    top_k_clusters = np.argsort(centroid_similarity[0])[-top_k:][::-1]
    return top_k_clusters
