import numpy as np
from rag_components.load_embedding_model import load_embedding_model

def retrieve_context(query, index, documents, top_k=3):
    """Recupera os documentos mais relevantes do índice FAISS para uma dada consulta."""
    model = load_embedding_model()
    query_embedding = model.encode([query])
    
    if index is None or index.ntotal == 0:
        return ""
    
    # Faiss espera np.float32, então convertemos a query embedding
    D, I = index.search(np.array(query_embedding).astype('float32'), top_k)
    
    retrieved_docs = [documents[i] for i in I[0] if i < len(documents)]
    
    return "\n".join(retrieved_docs)