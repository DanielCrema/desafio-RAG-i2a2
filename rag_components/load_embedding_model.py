import streamlit as st

@st.cache_resource
def load_embedding_model():
    """Carrega o modelo de embedding uma única vez."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer('paraphrase-MiniLM-L6-v2')