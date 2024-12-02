from sentence_transformers import SentenceTransformer
import os
from retrieve_map_mar import process_map_mar
import pandas as pd
import numpy as np
import torch

model = SentenceTransformer("all-MiniLM-L6-v2")

"""
Extracts a batch of queries and their relevant documents from the dataset.
Arguments:
    first_row (int): The starting index for extracting queries from the dataset.
    batch_size (int): The number of queries to extract in the current batch.
    retrieve_docs (bool): Flag indicating whether to retrieve relevant documents for each query.
Returns:
    dict: A dictionary where keys are query IDs and values are dictionaries containing the query text and relevant documents.
"""
def extract_docs_queries(first_row, batch_size, retrieve_docs = False):
    queries_relevant_text = {}
    batch =  queries.iloc[first_row: first_row + batch_size]

    for _, j in batch.iterrows():
        query_id = j['Query number']
        query_text = j['Query']
        if(retrieve_docs == True):
            relevant_docs = queries_results[queries_results['Query_number'] == query_id]['doc_number'].astype(str).tolist()
            queries_relevant_text[query_id] = {
                'query' : query_text, 
                'relevant_docs': relevant_docs }
        else:
            queries_relevant_text[query_id] = {'query' : query_text}
        
    return queries_relevant_text


def embedding():
    path = "full_docs_small"
    docs = {}
    embeddings = {}

    for name in os.listdir(path):
        if name.endswith(".txt"):
            with open(os.path.join(path, name), 'r', encoding='utf-8') as file:
                docs[name] = file.read()

    for name, text in docs.items():
        embeddings[name] = model.encode(text)

    torch.save(embeddings, 'embeddings.pt')


queries = pd.read_excel('queries/dev_small_queries.xlsx')
queries_results = pd.read_csv('queries/dev_query_results_small.csv')

embedding_path = 'embeddings.pt'
if not os.path.exists(embedding_path):
    embedding()

embeddings_dict = torch.load(embedding_path)
doc_names = list(embeddings_dict.keys())
embeddings = np.array(list(embeddings_dict.values()))

query_docs = extract_docs_queries(0, len(queries), retrieve_docs=True)

k_values = [1, 3, 5, 10]
output_file = "query_results.csv"

process_map_mar(model, query_docs, k_values, output_file, embeddings, doc_names)