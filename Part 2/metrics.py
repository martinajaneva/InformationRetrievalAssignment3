def avg_precision_at_k(results, relevant_set, k):
    """
    Calculate the Average Precision at a specified rank (AP@K) for a given query
    References: https://sdsawtelle.github.io/blog/output/mean-average-precision-MAP-for-recommender-systems.html
    Args:
        results (list): A list of tuples (document, similarity score), sorted by highest to lowest similarity.
        relevant_set (set): A set of documents relevant to the query.
        k (int): Rank position up to which precision is computed.
    Returns:
        float: Average Precision at the specified K.
    """
    precision_sum = 0.0
    count = 0
    
    for pos, doc in enumerate(results[:k], 1):
        if doc in relevant_set:
            count += 1
            precision_sum += count / pos
     
    return precision_sum / count if count > 0 else 0.0

def avg_recall_at_k(results, relevant_set, k):
    """
    Calculate the Average Recall at a specified rank (AR@K) for a given query.
    References: https://sdsawtelle.github.io/blog/output/mean-average-precision-MAP-for-recommender-systems.html
    Arguments:
        results (list): A list of tuples (document, similarity score), sorted by highest to lowest similarity.
        relevant_set (set): A set of documents relevant to the query.
        k (int): Rank position up to which recall is calculated.
    Returns:
        float: Average Recall at the specified K.
    """
    count = 0
    total_relevant = len(relevant_set)
    
    for doc in results[:k]:
        if doc in relevant_set:
            count += 1
    
    return count / total_relevant if total_relevant > 0 else 0.0


def find_map_mar(k_values, result, docs, results):
    """
    Calculate Mean Average Precision (MAP) and Mean Average Recall (MAR) at specified ranks (k) for a given set of query results.
    References: https://sdsawtelle.github.io/blog/output/mean-average-precision-MAP-for-recommender-systems.html
    Arguments:
        k_values (list): A list of rank positions (k) at which MAP and MAR are calculated.
        result (list): A list of tuples (document, similarity score), sorted by highest to lowest similarity.
        docs (set): A set of relevant documents for the query.
        results (DataFrame): A pandas DataFrame to store the MAP and MAR values.
    Returns:
        None: The function updates the results DataFrame with MAP and MAR values at specified k.
    """
    for k in k_values:
        map_k = avg_precision_at_k(result, docs, k)
        map = "map_" + str(k) 
        results.loc[:, map] = round(map_k, 3)
        print(f"MAP@{k}: {map_k:.3f}")

        mar_k = avg_recall_at_k(result, docs, k) 
        mar = "mar_" + str(k)
        results.loc[:, mar] = round(mar_k, 3)
        print(f"MAR@{k}: {mar_k:.3f}")