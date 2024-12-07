import pandas as pd

from embeddings import embed_query
from clusterring import find_relevant_clusters
from documents import search_documents
from metrics import find_map_mar


def process_queries(query_docs, model, embeddings, centroids, cluster_assignments, inverted_index, 
                    top_k_clusters, top_n_docs, output, k = None, header_written = False, evaluation = True):
    """
    Process queries to compute and save Mean Average Precision (MAP) and Mean Average Recall (MAR) for each query based on the retrieved results,
    or extract the most relevant documents that closely match the query.

    Evaluation eferences: https://www.evidentlyai.com/ranking-metrics/mean-average-precision-map
    Arguments:
        query_docs (dict): A dictionary where keys are query identifiers and values are dictionaries containing the query details.
        model (SentenseTransformer): Contains the model used in the system.
        embeddings (tensor): Contains the embeddings of the documents
        centroids (ndarray): Contains the centroids of the document clusters
        cluster_assignments (ndarray): Contains the cluster assignments of the document clusters
        inverted_index (dict): Contains the inverted index
        top_k_clusters (int): The number of the top k similar clusters to be considered
        top_n_docs (int): The number of most similar documents to be considered
        output (str): The path to the output CSV file where results will be saved.
        k (list): A list of rank positions (k) for which MAP and MAR are calculated.
        header_written (bool): A flag used to properly store the results into the output file.
        evaluation (bool): Determines whether the system will evaluate the results using MAP and MAR or extract the most relevant results
    Returns:
        None: The function appends the computed MAP and MAR results to the specified CSV file.
    """

    for query, i in query_docs.items():
        query_string = i['query']
        print(f"Processing query - {query_string}")
        query_embedding = embed_query(query_string, model)
        relevant_clusters = find_relevant_clusters(query_embedding, centroids, top_k_clusters)
        top_documents = search_documents(query_embedding, embeddings, cluster_assignments, inverted_index, relevant_clusters, top_n_docs)
    
        if(evaluation == True):
            docs = i['relevant_docs']

            results = pd.DataFrame(data = [{'num_query': query, 'text_query': query_string}])
            
            find_map_mar(k, top_documents, docs, results)
            results.to_csv(output, mode='a', index=False, header=header_written)
            header_written=False

        elif(evaluation == False):
            results = pd.DataFrame({'Query_number': [query] * len(top_documents),
                                    'doc_number': top_documents})

            results.to_csv(output, mode='a', index=False, header=header_written)
            header_written=False