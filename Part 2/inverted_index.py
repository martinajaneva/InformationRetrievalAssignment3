from tqdm import tqdm

def create_inverted_index(cluster_assignments, filenames):
    """
    Creating the inverted index based on the cluster assignments and the file names. 
    Args:
        cluster_assignments (ndarray): The cluster assignments
        filenames (list): The loaded document names
    Returns:
        inverted_index (dict): A dictionary of the inverted index 
    """
    inverted_index = {}
    for idx, cluster_id in tqdm(enumerate(cluster_assignments)):
        if cluster_id not in inverted_index:
            inverted_index[cluster_id] = []
        inverted_index[cluster_id].append(filenames[idx])
    return inverted_index
