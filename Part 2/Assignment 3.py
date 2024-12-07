import os
import pandas as pd

from documents import load_documents
from embeddings import create_document_embeddings, load_saved_embeddings_and_model
from extract_query_documents import extract_docs_queries
from clusterring import cluster_embeddings
from inverted_index import create_inverted_index
from process_queries import process_queries


#-----------------------------------------------------------------------#
####            Option 1: Evaluation on the large dataset            ####
#-----------------------------------------------------------------------#

# Instructions:
# Uncomment the following code to evaluate the system based on the MAP@k and MAR@k
# using different total clusters and top k clusters.

#--------------------------------------------------------------------------------#
#### Uncomment from here: ####

#------------------------------------------------------------------#
####            Step 0: Specifying system parameters            ####
#------------------------------------------------------------------#

dataset_path = "Data/full_docs"
queries_path = "Data/large_queries/dev_queries.tsv"
queries_results_path = "Data/large_queries/dev_query_results.csv"

# Investigating the results of different k and top_k clusters
k_clusters_list = [50, 100, 200, 500, 1000]       # Total embeddings clusters
top_k_clusters_list = [1, 5, 10, 20]              # Top k relevant clusters
top_n_docs = 10                                   # Top n relevant documents
k_values = [1, 3, 5, 10]                          # k documents for MAP, MAR

#--------------------------------------------------------------#
####            Step 1: Load & embed the dataset            ####
#--------------------------------------------------------------#
# Loading the dataset
documents, filenames = load_documents(dataset_path)
print("loaded dataset")

# # Create document embeddings on the loaded dataset
# embeddings, model = create_document_embeddings(documents)     # Warning: Took ~ 5 hours to complete for large dataset!

# Loading the saved embeddings and model
embeddings, model = load_saved_embeddings_and_model()


#----------------------------------------------------------------------------------------#
####            Step 2: Load the queries and extract corresponding results            ####
#----------------------------------------------------------------------------------------#
# Loading queries the queries and  the query results
queries = pd.read_table(queries_path)
queries_results = pd.read_csv(queries_results_path)

# Extract the first 1000 queries and their corresponding results
query_docs = extract_docs_queries(queries=queries, queries_results=queries_results,
                                    first_row=0, batch_size=1000, retrieve_docs=True)

#--------------------------------------------------------------------------------#
####            Step 3: Cluster embeddings & Create inverted index            ####
#--------------------------------------------------------------------------------#
for k_clusters in k_clusters_list:
    print(f"Processing with {k_clusters} clusters") 
    
    # Clustering the document embeddings
    cluster_assignments, centroids = cluster_embeddings(embeddings, k_clusters)

    # Constructing the inverted index
    inverted_index = create_inverted_index(cluster_assignments, filenames)

    for top_k_clusters in top_k_clusters_list:
        print(f"Processing with {top_k_clusters} top clusters") 

        # Preparing the output file to store the results
        output = "results_top_" + str(top_k_clusters) + '_clusters_of' + str(k_clusters) + '.csv'

        # Specifying certain flags to ensure the results are stored correctly
        file_exists = os.path.isfile(output)
        header_written = not file_exists

        print("=========================================================================")

        #-----------------------------------------------------#
        ####            Step 4: Process queries            ####
        #-----------------------------------------------------#

        # Processing the queries
        process_queries(output = output,
                        query_docs = query_docs, 
                        model = model,
                        embeddings = embeddings,
                        centroids = centroids,
                        cluster_assignments = cluster_assignments,
                        inverted_index = inverted_index,
                        top_k_clusters = top_k_clusters,
                        top_n_docs = top_n_docs,
                        k = k_values,
                        header_written = header_written,
                        evaluation = True)

#### Until here ####
#--------------------------------------------------------------------------------#


# ========================================================================================================================================== #


#-------------------------------------------------------------------------#
####            Option 2: Extract top 10 relevant documents            ####
#-------------------------------------------------------------------------#

# Instructions:
# Uncomment the following code to extract the top 10 relevant documents
# based on the test_queries

#--------------------------------------------------------------------------------#
#### Uncomment from here: ####

# #------------------------------------------------------------------#
# ####            Step 0: Specifying system parameters            ####
# #------------------------------------------------------------------#

# dataset_path = "Data/full_docs"
# queries_path = "Data/test_queries/queries.csv"

# # Specifying the k and top_k clusters based on the best results
# # from Option 1.
# k_clusters = 50          # Total embeddings clusters
# top_k_clusters = 20      # Top k relevant clusters
# top_n_docs = 10          # Top n relevant documents

# #--------------------------------------------------------------#
# ####            Step 1: Load & embed the dataset            ####
# #--------------------------------------------------------------#
# # Loading the dataset
# documents, filenames = load_documents(dataset_path)
# print("loaded dataset")

# # # Create document embeddings on the loaded dataset
# # embeddings, model = create_document_embeddings(documents)     # Warning: Took ~ 5 hours to complete for large dataset!

# # Loading the saved embeddings and model
# embeddings, model = load_saved_embeddings_and_model()

# #----------------------------------------------------------------------------------------#
# ####            Step 2: Load the queries and extract corresponding results            ####
# #----------------------------------------------------------------------------------------#
# # Loading the queries
# queries = pd.read_table(queries_path)

# # Extracting the queries
# query_docs = extract_docs_queries(queries=queries,
#                                   first_row=0, batch_size=9999999999, retrieve_docs=False)

# #--------------------------------------------------------------------------------#
# ####            Step 3: Cluster embeddings & Create inverted index            ####
# #--------------------------------------------------------------------------------#

# # Clustering the document embeddings
# cluster_assignments, centroids = cluster_embeddings(embeddings, k_clusters)

# # Constructing the inverted index
# inverted_index = create_inverted_index(cluster_assignments, filenames)

# output = "result.csv"

# # Specifying certain flags to ensure the results are stored correctly
# file_exists = os.path.isfile(output)
# header_written = not file_exists

# #-----------------------------------------------------#
# ####            Step 4: Process queries            ####
# #-----------------------------------------------------#

# # Processing the queries
# process_queries(output = output,
#                 query_docs = query_docs, 
#                 model = model,
#                 embeddings = embeddings,
#                 centroids = centroids,
#                 cluster_assignments = cluster_assignments,
#                 inverted_index = inverted_index,
#                 top_k_clusters = top_k_clusters,
#                 top_n_docs = top_n_docs,
#                 header_written = header_written,
#                 evaluation = False)

#### Until here ####
#--------------------------------------------------------------------------------#