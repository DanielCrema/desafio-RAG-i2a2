import streamlit as st

def init_session_state():
    if 'gemini_api_key' not in st.session_state:
        st.session_state['gemini_api_key'] = ''
    if 'zip_bytes' not in st.session_state:
        st.session_state['zip_bytes'] = None
    if 'zip_hash' not in st.session_state:
        st.session_state['zip_hash'] = None
        
    # CORREÇÃO DO KeyError: "available_files"
    if 'available_files' not in st.session_state:
        st.session_state['available_files'] = [] 
        
    if 'file_options_map' not in st.session_state:
        st.session_state['file_options_map'] = {}
    if 'selected_file_name' not in st.session_state:
        st.session_state['selected_file_name'] = None
    if 'df' not in st.session_state:
        st.session_state['df'] = None
    if 'df_columns' not in st.session_state:
        st.session_state['df_columns'] = None
    if 'faiss_index' not in st.session_state:
        st.session_state['faiss_index'] = None
    if 'documents' not in st.session_state:
        st.session_state['documents'] = []
    if 'total_lines' not in st.session_state:
        st.session_state['total_lines'] = 0
    if 'processed_percentage' not in st.session_state:
        st.session_state['processed_percentage'] = 0
    if 'current_chunk_start' not in st.session_state:
        st.session_state['current_chunk_start'] = 0
    if 'cleaned_status' not in st.session_state:
        st.session_state['cleaned_status'] = {}
    if 'file_name_context' not in st.session_state:
        st.session_state['file_name_context'] = ""
    if 'conclusoes_historico' not in st.session_state:
        st.session_state['conclusoes_historico'] = ""
    if 'codigo_gerado' not in st.session_state:
        st.session_state['codigo_gerado'] = None
    if 'resultado_texto' not in st.session_state:
        st.session_state['resultado_texto'] = None
    if 'resultado_df' not in st.session_state:
        st.session_state['resultado_df'] = None
    if 'erro_execucao' not in st.session_state:
        st.session_state['erro_execucao'] = None
    if 'img_bytes' not in st.session_state:
        st.session_state['img_bytes'] = None
    if 'consultar_ia' not in st.session_state:
        st.session_state['consultar_ia'] = False
    if 'exibir_codigo' not in st.session_state:
        st.session_state['exibir_codigo'] = False
    if 'habilitar_grafico' not in st.session_state:
        st.session_state['habilitar_grafico'] = False
    if 'gerar_pdf' not in st.session_state:
        st.session_state['gerar_pdf'] = False
    if 'user_query_input_widget' not in st.session_state:
        st.session_state['user_query_input_widget'] = ""
    if 'current_query_text' not in st.session_state:
        st.session_state['current_query_text'] = ""