import os
import io
import zipfile
import pandas as pd
import streamlit as st
import google.generativeai as genai
from helpers.normalize_text import normalize_text

def agente1_identifica_arquivos(zip_bytes):
    """
    Identifica todos os arquivos CSV, XLSX e TXT no ZIP e tenta obter cabeçalhos.
    """
    files_info = []
    
    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as z:
        for name in z.namelist():
            if name.startswith('__MACOSX/') or name.endswith('/'):
                continue
                
            ext = os.path.splitext(name)[1].lower()
            
            if ext in ['.csv', '.xlsx', '.txt']:
                try:
                    with z.open(name, 'r') as file_in_zip:
                        data_in_memory = io.BytesIO(file_in_zip.read())
                        
                        header = []
                        if ext == '.csv':
                            temp_df = pd.read_csv(data_in_memory, nrows=0, encoding='utf-8', on_bad_lines='skip', low_memory=False)
                            header = temp_df.columns.tolist()
                        
                        elif ext == '.xlsx':
                            temp_df = pd.read_excel(data_in_memory, nrows=0)
                            header = temp_df.columns.tolist()
                        
                        elif ext == '.txt':
                            data_in_memory.seek(0)
                            # Tentativa de ler a primeira linha para inferir o separador
                            first_line = io.TextIOWrapper(data_in_memory, encoding='utf-8').readline().strip()
                            data_in_memory.seek(0)
                            
                            separator = '\s+'
                            if ',' in first_line:
                                separator = ','
                            elif ';' in first_line:
                                separator = ';'

                            temp_df = pd.read_csv(data_in_memory, nrows=1, encoding='utf-8', sep=separator, engine='python', on_bad_lines='skip', header=None)
                            
                            if temp_df.shape[1] > 0:
                                header = [f"COL_{i+1}" for i in range(temp_df.shape[1])]
                            else:
                                header = ["COL_1"] 
                                
                        file_info = {
                            "name": name,
                            "extension": ext,
                            "header": header,
                            "schema_text": ", ".join(header),
                            "num_cols": len(header)
                        }
                        files_info.append(file_info)
                except Exception as e:
                    pass
            
    return files_info

def agente1_interpreta_contexto_arquivo(api_key, file_info_list):
    """
    Usa o Gemini para descrever o que cada arquivo representa com base no nome e cabeçalho.
    """
    if not api_key:
        return {info["name"]: "API Key não configurada para gerar contexto." for info in file_info_list}

    contextos = {}
    try:
        genai.configure(api_key=api_key)
        # MODELO ATUALIZADO PARA gemini-2.5-flash
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        prompt_parts = ["# PERSONA: Você é um Analista de Dados Sênior. Sua única função é INFERIR o CONTEÚDO e CONTEXTO de um arquivo de dados baseado no NOME e CABEÇALHO. DÊ UMA DESCRIÇÃO DE UMA ÚNICA FRASE CURTA. \n\n# ARQUIVOS PARA ANÁLISE:\n"]
        
        for info in file_info_list:
            prompt_parts.append(f"- ARQUIVO: {info['name']} (Colunas: {info['schema_text']})\n")
        
        prompt_parts.append("\n# INFERÊNCIA:\nResponda APENAS com uma lista numerada, onde cada item é uma descrição concisa (uma frase) para o respectivo arquivo, focando no que ele representa. Ex: 'O arquivo representa dados de transações de cartão de crédito e a coluna CLASS indica fraude.'\n")
        
        response = model.generate_content("".join(prompt_parts))
        
        descricoes = [line.strip() for line in response.text.split('\n') if line.strip().startswith(('1.', '2.', '3.', '-', '*')) or (len(line.strip()) > 5 and i > 0)]
        
        for i, info in enumerate(file_info_list):
            if i < len(descricoes):
                text = descricoes[i]
                if text and text[0].isdigit() and '.' in text[:3]:
                    text = text.split('.', 1)[-1].strip()
                contextos[info["name"]] = text
            else:
                contextos[info["name"]] = f"Erro na inferência automática. Cabeçalho: {info['schema_text']}"
                
    except Exception as e:
        for info in file_info_list:
             contextos[info["name"]] = f"Erro na inferência automática. Cabeçalho: {info['schema_text']}"
            
    return contextos

def agente1_processa_arquivo_chunk(zip_bytes, selected_file_name, start_row, nrows, df_columns, expected_num_cols):
    """
    Processa um chunk do arquivo selecionado (CSV, XLSX, TXT) dentro do ZIP.
    """
    ext = os.path.splitext(selected_file_name)[1].lower()
    
    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as z:
            with z.open(selected_file_name, 'r') as file_in_zip:
                file_bytes_in_memory = io.BytesIO(file_in_zip.read())
                
                chunk = pd.DataFrame()
                
                # Configurações de leitura
                skiprows = range(1, start_row + 1) if start_row > 0 else 0
                header = None if start_row > 0 else 'infer'

                # Leitura de CSV
                if ext == '.csv':
                    chunk = pd.read_csv(
                        file_bytes_in_memory,
                        skiprows=skiprows,
                        nrows=nrows,
                        low_memory=False,
                        header=header,
                        encoding='utf-8',
                        on_bad_lines='skip'
                    )
                
                # Leitura de XLSX
                elif ext == '.xlsx':
                    chunk = pd.read_excel(file_bytes_in_memory, header=header, skiprows=skiprows, nrows=nrows)
                        
                # Leitura de TXT
                elif ext == '.txt':
                    # Tenta inferir o separador para leitura do chunk
                    file_bytes_in_memory.seek(0)
                    first_line = io.TextIOWrapper(file_bytes_in_memory, encoding='utf-8', errors='ignore').readline().strip()
                    file_bytes_in_memory.seek(0)
                    
                    separator = '\s+'
                    if ',' in first_line:
                        separator = ','
                    elif ';' in first_line:
                        separator = ';'
                        
                    chunk = pd.read_csv(
                        file_bytes_in_memory,
                        skiprows=skiprows,
                        nrows=nrows,
                        low_memory=False,
                        header=header,
                        encoding='utf-8',
                        sep=separator,
                        engine='python',
                        on_bad_lines='skip'
                    )

                if chunk.empty:
                    return None, "Processamento de todos os lotes concluído."

                # --- TRATAMENTO DE COLUNAS/ESQUEMA ---
                if start_row == 0:
                    # Captura o cabeçalho original (antes da normalização)
                    if df_columns is None:
                        st.session_state['df_columns'] = chunk.columns
                    expected_num_cols = len(st.session_state['df_columns'])
                    
                
                if df_columns is not None:
                    # Correção de Length Mismatch para chunks subsequentes
                    current_cols = chunk.shape[1]
                    
                    if current_cols < expected_num_cols:
                        for i in range(current_cols, expected_num_cols):
                            chunk[f'TEMP_FILL_{i}'] = pd.NA
                        chunk = chunk.iloc[:, :expected_num_cols]
                    
                    elif current_cols > expected_num_cols:
                        chunk = chunk.iloc[:, :expected_num_cols]
                        
                    # Atribui os nomes de coluna originais (normalizados)
                    chunk.columns = [normalize_text(col.strip().upper()) for col in st.session_state['df_columns']]
                else:
                    # Normalização das colunas
                    chunk.columns = [normalize_text(col.strip().upper()) for col in chunk.columns]

                return chunk, "Dados carregados e prontos para análise!"
            
    except Exception as e:
        return None, f"Erro ao processar o arquivo: header must be integer or list of integers {e}"