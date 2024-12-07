import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

def load_documents(dataset_path):
    """
    Loads the dataset in parallel.
    Args:
        dataset_path (string): Contains the path to the dataset
    Returns:
        documents (list): The content of the loaded documents
        filenames (list): The file name of the loaded documents
    """

    documents = []
    filenames = []
    
    def load_file(filename):
        """
        Loads the given file in memory
        Args:
            filename (string): Contains the path of the file to be opened
        Returns:
            return (tuple): The content of the file and the file name
        """
        with open(os.path.join(dataset_path, filename), 'r', encoding='utf-8') as file:
            return file.read(), filename
    
    # Load the documents of the dataset and store the content and filenames separately
    with ThreadPoolExecutor() as executor:
        results = list(tqdm(executor.map(load_file, [f for f in os.listdir(dataset_path) if f.endswith('.txt')]), total=len(os.listdir(dataset_path))))
        for content, filename in results:
            documents.append(content)
            filenames.append(filename)
    
    return documents, filenames


def search_documents(query_embedding, embeddings, cluster_assignments, inverted_index, top_k_clusters, top_n_docs=10):
    """
    Searches through the dataset documents for the msot relevant documents to a given query
    Args:
        query_embedding (tensor): The specified embedded query
        embeddings (tensor): Contains the embeddings of the documents
        cluster_assignments (ndarray): Contains the cluster assignments of the document clusters
        inverted_index (dict): Contains the inverted index
        top_k_clusters (int): The number of the top k similar clusters to be considered
        top_n_docs (int): The number of most similar documents to be considered
    Returns:
        top_documents_clean (list): Contains the document id of the most relevant k documents
    """
    candidate_documents = []
    candidate_indices = []
    for cluster_id in top_k_clusters:
        doc_ids = inverted_index.get(cluster_id, [])
        cluster_indices = [i for i, label in enumerate(cluster_assignments) if label == cluster_id]
        candidate_documents.extend(doc_ids)
        candidate_indices.extend(cluster_indices)
    
    # Computing cosine similarity with all candidate documents
    candidate_embeddings = embeddings[candidate_indices]
    similarities = cosine_similarity(query_embedding.cpu().numpy(), candidate_embeddings.cpu().numpy())[0]
    
    # Selecting top-N documents based on similarity
    top_n_indices = np.argsort(similarities)[-top_n_docs:][::-1]
    top_documents = [candidate_documents[i] for i in top_n_indices]
    
    # Cleaninging the predicted document to keep only the document id
    top_documents_clean = []
    for document in top_documents:
        clean_doc = document.replace('output_', '').replace('.txt', '')
        top_documents_clean.append(clean_doc)

    return top_documents_clean