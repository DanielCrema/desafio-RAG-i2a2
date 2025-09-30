import streamlit as st
import os
import pickle
import hashlib
import faiss
import tempfile

def save_progress(file_hash, df, faiss_index, documents, total_lines):
    """Salva o progresso no disco."""
    try:
        if st.session_state.get('selected_file_name'):
            unique_file_hash = hashlib.md5((file_hash + st.session_state['selected_file_name']).encode()).hexdigest()
            
            temp_dir = tempfile.gettempdir()
            
            # Garante que o df não está vazio antes de salvar
            if df is not None and not df.empty:
                with open(os.path.join(temp_dir, f"{unique_file_hash}_df.pkl"), "wb") as f:
                    pickle.dump(df, f)
            
            # Garante que o índice foi criado antes de salvar
            if faiss_index is not None and faiss_index.ntotal > 0:
                faiss.write_index(faiss_index, os.path.join(temp_dir, f"{unique_file_hash}_faiss_index.bin"))
                
            if documents:
                with open(os.path.join(temp_dir, f"{unique_file_hash}_documents.pkl"), "wb") as f:
                    pickle.dump(documents, f)
            
            with open(os.path.join(temp_dir, f"{unique_file_hash}_metadata.txt"), "w") as f:
                f.write(str(total_lines))
                
            return True
        return False
    except Exception as e:
        # st.error(f"Erro ao salvar o progresso: {e}") 
        return False