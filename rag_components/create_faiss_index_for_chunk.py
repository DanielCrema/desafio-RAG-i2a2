import streamlit as st
import faiss
import numpy as np
from rag_components.load_embedding_model import load_embedding_model

def create_faiss_index_for_chunk(chunk):
    """Cria/adiciona a um índice FAISS para um chunk específico."""
    model = load_embedding_model()
    
    # 1. Pré-processamento
    docs_chunk = chunk.astype(str).apply(lambda x: ' '.join(x), axis=1).tolist()
    
    # Verifica se há documentos para processar
    if not docs_chunk:
        return True

    # 2. Embedding
    embeddings_chunk = model.encode(docs_chunk, show_progress_bar=False)
    
    dimension = embeddings_chunk.shape[1]
    
    # 3. Criação/Adição ao Índice FAISS
    if st.session_state['faiss_index'] is not None:
        st.session_state['faiss_index'].add(np.array(embeddings_chunk).astype('float32'))
    else:
        index = faiss.IndexFlatL2(dimension)
        index.add(np.array(embeddings_chunk).astype('float32'))
        st.session_state['faiss_index'] = index
        
    # 4. Atualiza Documentos
    if st.session_state['documents'] is None:
        st.session_state['documents'] = []
    st.session_state['documents'].extend(docs_chunk)
    
    return True