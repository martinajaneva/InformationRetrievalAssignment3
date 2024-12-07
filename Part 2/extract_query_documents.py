
def extract_docs_queries(queries, first_row, batch_size, queries_results = None, retrieve_docs = False):
    """
    Extracts a batch of queries and their relevant documents from the dataset.
    Arguments:
        queries (dataframe): Contains the queries in a 'Query number' , 'Query' format
        first_row (int): The starting index for extracting queries from the dataset.
        batch_size (int): The number of queries to extract in the current batch.
        queries_results (dataframe): Contains the relevant documents that match each query number.
        retrieve_docs (bool): Flag indicating whether to retrieve relevant documents for each query.
    Returns:
        return (dict): A dictionary containing the query string and the relevant result documents if retrieve_docs is set to True,
        or it contains just the query string if retrieve_docs is set to False
    """

    queries_relevant_text = {}
    batch =  queries.iloc[first_row: first_row + batch_size]

    for _, j in batch.iterrows():
        id = j['Query number']
        query_text = j['Query']
        if(retrieve_docs == True):
            # Extracting the relevant documents of each query number
            relevant_docs = queries_results[queries_results['Query_number'] == id]['doc_number'].astype(str).tolist()
            queries_relevant_text[id] = {
                'query' : query_text, 
                'relevant_docs': relevant_docs }
        else:
            queries_relevant_text[id] = {'query' : query_text}
        
    return queries_relevant_text
