import pandas as pd
import streamlit as st

def agente_limpeza_dados(df):
    """
    Identifica e converte colunas para tipos numéricos e categóricos.
    Aplica a limpeza 'in-place' no DF.
    """
    if df is None:
        return None

    # Iterar sobre uma cópia da lista de colunas para evitar problemas de modificação durante o loop
    for col in list(df.columns):
        if col not in df.columns: # Proteção caso a coluna seja excluída ou renomeada
             continue

        temp_series = pd.to_numeric(df[col], errors='coerce')
        
        # 1. Numérico
        if temp_series.notna().sum() / len(temp_series) > 0.8:
            df[col] = temp_series
            st.session_state['cleaned_status'][col] = 'Numeric'
        
        # 2. Categórico
        elif df[col].nunique() < 50 and len(df[col].unique()) < len(df) / 2:
            df[col] = df[col].astype('category')
            st.session_state['cleaned_status'][col] = 'Categorical'
        
        # 3. Texto/Objeto
        else:
            st.session_state['cleaned_status'][col] = 'Object'
    
    return df