import hashlib
import os
import pickle
import tempfile
import faiss

def load_progress(file_hash, selected_file_name):
    """Carrega o progresso do disco, se existir."""
    try:
        unique_file_hash = hashlib.md5((file_hash + selected_file_name).encode()).hexdigest()
        temp_dir = tempfile.gettempdir()
        
        df_path = os.path.join(temp_dir, f"{unique_file_hash}_df.pkl")
        faiss_path = os.path.join(temp_dir, f"{unique_file_hash}_faiss_index.bin")
        docs_path = os.path.join(temp_dir, f"{unique_file_hash}_documents.pkl")
        meta_path = os.path.join(temp_dir, f"{unique_file_hash}_metadata.txt")

        # Verifica se todos os arquivos essenciais existem
        if os.path.exists(df_path) and os.path.exists(faiss_path) and os.path.exists(docs_path):
            with open(df_path, "rb") as f:
                df = pickle.load(f)
            faiss_index = faiss.read_index(faiss_path)
            with open(docs_path, "rb") as f:
                documents = pickle.load(f)
            
            total_lines = 0
            if os.path.exists(meta_path):
                 with open(meta_path, "r") as f:
                    total_lines = int(f.read())
            
            # Retorna o total de linhas do DF carregado, que é o número real de linhas processadas
            return df, faiss_index, documents, len(df)
        return None, None, None, 0
    except Exception as e:
        # print(f"Erro ao carregar o progresso: {e}")
        return None, None, None, 0