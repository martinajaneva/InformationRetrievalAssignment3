import torch

from sentence_transformers import SentenceTransformer
from tqdm import tqdm

def create_document_embeddings(documents, model_name='all-MiniLM-L6-v2', batch_size=10000):
    """
    Creates the embeddings of the given documents based on a specified pre-trained model
    Args:
        documents (list): Contains the content of all loaded documents
        model_name (string): The name of the model to be used
        batch_size (int): Number of documents to be processed at a time
    Returns:
        return (tuple): The created embeddings of the documents and the model used during the embedding process
    """
    
    model = SentenceTransformer(model_name)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    embeddings = []

    with tqdm(total=len(documents)) as pbar:
        for start_idx in range(0, len(documents), batch_size):
            batch = documents[start_idx:start_idx + batch_size]
            batch_embeddings = model.encode(batch, convert_to_tensor=True, show_progress_bar=True, device=device)
            embeddings.append(batch_embeddings)
            pbar.update(len(batch))
    
    embeddings = torch.cat(embeddings, dim=0)

    # Save embeddings and model to disk
    torch.save(embeddings, 'document_embeddings.pt')
    model.save('sentence_transformer_model')

    return embeddings, model

def load_saved_embeddings_and_model(embeddings_path='document_embeddings.pt', model_path='sentence_transformer_model'):
    """
    Loads the document embeddings and the transformer model from disk
    Args:
        embeddings_path (string): Path to the document embeddings
        model_path (string): Path to the tansformer model
    Returns:
        return (tuple): The loaded embeddings and transformer model
    """
    # Load embeddings from disk
    embeddings = torch.load(embeddings_path, weights_only=True)
    # Load the saved model
    model = SentenceTransformer(model_path)
    return embeddings, model

def embed_query(query, model):
    """
    Create the embedding of the specified query
    Args:
        query (string): The query string
        model (SentenceTransformer): The transformer model to be used for the embedding process
    Returns:
        return (): The embedded query
    """
    query_embedding = model.encode([query], convert_to_tensor=True)
    return query_embedding
