from average_functions import avg_precision_at_k, avg_recall_at_k
import pandas as pd
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity

"""
Calculate Mean Average Precision (MAP) and Mean Average Recall (MAR) at specified ranks (k) for a given set of query results.
References: https://sdsawtelle.github.io/blog/output/mean-average-precision-MAP-for-recommender-systems.html
Arguments:
    k_values (list): A list of rank positions (k) at which MAP and MAR are calculated.
    results (list): A list of documents sorted by highest to lowest similarity.
    docs (set): A set of relevant documents for the query.
    output (DataFrame): A pandas DataFrame to store the MAP and MAR values.
Returns:
    None: The function updates the results DataFrame with MAP and MAR values at specified k.
"""
def find_map_mar(k_values, results, docs, output):
    for k in k_values:
        map_k = avg_precision_at_k(results, docs, k)
        map_col = "map_" + str(k) 
        output.loc[:, map_col] = round(map_k, 3)
        print(f"MAP@{k}: {map_k:.3f}")

        mar_k = avg_recall_at_k(results, docs, k) 
        mar = "mar_" + str(k)
        output.loc[:, mar] = round(mar_k, 3)
        print(f"MAR@{k}: {mar_k:.3f}")


"""
Process queries to calculate MAP and MAR at specified ranks (k) and store the results in a CSV file.
Arguments:
    model (SentenceTransformer): Pre-trained Sentence-BERT model for generating embeddings.
    query_docs (dict): Dictionary containing query IDs, query texts, and relevant documents.
    k_values (list): A list of rank positions (k) for which MAP and MAR are calculated.
    output_path (str): Path to save the output results as a CSV file.
    embeddings (np.ndarray): Array of document embeddings.
    doc_names (list): List of document names corresponding to the embeddings.
Returns:
    None: Results are saved directly to the output CSV file.
"""
def process_map_mar(model, query_docs, k_values, output_path, embeddings, doc_names):

    num_query = 1
    header_written = not os.path.exists(output_path)

    for query_id, query_detail in query_docs.items():
        query_text = query_detail['query']
        docs = set(query_detail['relevant_docs'])
        relevant_docs = {f"output_{doc}.txt" for doc in docs}

        query_embedding = model.encode(query_text).reshape(1, -1)
        similarities = cosine_similarity(query_embedding, embeddings)[0]
        top_k_indices = np.argsort(similarities)[::-1]
        retrieved_docs = [doc_names[i] for i in top_k_indices]
        query_results = pd.DataFrame(data=[{'query_id': query_id, 'query_text': query_text}])

        find_map_mar(k_values, retrieved_docs, relevant_docs, query_results)

        query_results.to_csv(output_path, mode='a', index=False, header=header_written)
        header_written = False

        num_query += 1