import streamlit as st
import pandas as pd
import zipfile
import os
import io
import hashlib

# --- Helpers, Modules and Sandboxing ---
from helpers.normalize_text import normalize_text
from modules.init_session_state import init_session_state
from sandboxing.executa_codigo_seguro import executa_codigo_seguro

# ------- Agents -------
from agents.agente_limpeza_dados import agente_limpeza_dados
from agents.agente0 import agente0_clarifica_pergunta
from agents.agente1 import (
    agente1_identifica_arquivos,
    agente1_interpreta_contexto_arquivo,
    agente1_processa_arquivo_chunk
)
from agents.agente2 import agente2_gera_codigo_pandas_eda
from agents.agente3 import agente3_formatar_apresentacao

# --- RAG Components ---
from rag_components.create_faiss_index_for_chunk import create_faiss_index_for_chunk
from rag_components.retrieve_context import retrieve_context
from rag_components.save_progress import save_progress
from rag_components.load_progress import load_progress

# Importação da SentenceTransformer será feita via st.cache_resource

# --- Configurações Iniciais e Variáveis ---
# Streamlit Page Configuration
st.set_page_config(
    page_title="Análise Exploratória de Dados (EDA) com Gemini e RAG",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constantes
CHUNK_SIZE = 1000

# --- Inicialização de Session State ---
init_session_state()

# --- Streamlit UI ---
st.title("Análise Exploratória de Dados (EDA) com Gemini e RAG")
st.markdown("---")

# --- Configuração da Sidebar ---
with st.sidebar:
    st.header("Configurações")
    with st.form("api_key_form"):
        api_key_input = st.text_input("Cole sua API Key do Gemini aqui", value=st.session_state['gemini_api_key'], type="password", key="gemini_api_key_input_form")
        submitted = st.form_submit_button("Salvar API Key")
        if submitted:
            st.session_state['gemini_api_key'] = api_key_input
            st.success("API Key salva com sucesso!")

    st.markdown("---")
    st.info("1. Faça o upload do arquivo **ZIP**.")
    st.info("2. Clique em **'Listar Arquivos'** para identificação.")
    st.info("3. Escolha o arquivo e clique em **'Analisar'**.")
    st.info("4. Faça sua pergunta de EDA.")

# --- Seção 1: Upload e Seleção de Dados ---
st.header("1. Upload e Seleção de Dados")
zipfile_input = st.file_uploader("Selecione o arquivo ZIP com o(s) arquivo(s) de dados (CSV, XLSX, TXT)", type=["zip"])

if st.button("Listar Arquivos no ZIP"):
    if zipfile_input is not None:
        st.session_state['zip_bytes'] = zipfile_input.getvalue()
        st.session_state['zip_hash'] = hashlib.md5(st.session_state['zip_bytes']).hexdigest()
        
        # Reseta estados importantes
        st.session_state['selected_file_name'] = None 
        st.session_state['df'] = None 
        st.session_state['conclusoes_historico'] = "" 
        st.session_state['processed_percentage'] = 0
        st.session_state['available_files'] = []
        st.session_state['file_options_map'] = {}
        st.session_state['faiss_index'] = None
        st.session_state['documents'] = []


        with st.spinner("Analisando arquivos e gerando contexto com Gemini..."):
            file_info_list = agente1_identifica_arquivos(st.session_state['zip_bytes'])
            
            if not file_info_list:
                st.error("O ZIP não contém arquivos CSV, XLSX ou TXT válidos.")
            else:
                st.session_state['available_files'] = file_info_list
                
                file_context_map = agente1_interpreta_contexto_arquivo(st.session_state.get('gemini_api_key'), file_info_list)
                
                options = {}
                for info in file_info_list:
                    context = file_context_map.get(info['name'], "Contexto não gerado.")
                    options[info['name']] = f"**{info['name']}** - {context}"
                
                st.session_state['file_options_map'] = options
                st.success(f"Encontrados {len(file_info_list)} arquivos de dados. Escolha qual analisar.")

if st.session_state['available_files']:
    
    st.markdown("---")
    st.subheader("Selecione o Arquivo para Análise:")
    
    options_list = list(st.session_state['file_options_map'].keys())
    
    if not options_list:
        st.error("Nenhum arquivo elegível encontrado após a identificação.")
    else:
        display_options = [st.session_state['file_options_map'][key] for key in options_list]
        
        default_index = 0
        try:
            if st.session_state['selected_file_name'] in options_list:
                default_index = options_list.index(st.session_state['selected_file_name'])
        except:
             default_index = 0

        selected_display = st.radio(
            "Arquivos Encontrados (Nome - Contexto Inferido):",
            options=display_options,
            index=default_index,
            key="file_selection_radio"
        )
        
        selected_file_name = options_list[display_options.index(selected_display)]
        st.session_state['selected_file_name'] = selected_file_name
        
        selected_file_info = next((info for info in st.session_state['available_files'] if info["name"] == selected_file_name), None)
        
        if st.button(f"Analisar Arquivo: {selected_file_name}") and selected_file_info:
            
            expected_num_cols = selected_file_info['num_cols']
            
            # --- INÍCIO DO PROCESSO DE CARGA/CHUNKED (RAG) ---
            
            st.info(f"Tentando carregar progresso anterior para **{selected_file_name}**...")
            
            # Tenta carregar o progresso anterior
            df_loaded, index_loaded, docs_loaded, lines_loaded_processed = load_progress(st.session_state['zip_hash'], selected_file_name)
            
            st.session_state['df'] = df_loaded
            st.session_state['faiss_index'] = index_loaded
            st.session_state['documents'] = docs_loaded
            st.session_state['current_chunk_start'] = lines_loaded_processed # Onde deve continuar o chunking
            
            # Tenta obter o total de linhas real do arquivo
            total_lines_file = 0
            try:
                ext = selected_file_info['extension']
                if ext == '.csv' or ext == '.txt':
                    with zipfile.ZipFile(io.BytesIO(st.session_state['zip_bytes']), "r") as z:
                        with z.open(selected_file_name, 'r') as file_in_zip:
                            # Subtrai 1 para o cabeçalho
                            total_lines_file = sum(1 for line in io.TextIOWrapper(file_in_zip, encoding='utf-8', errors='ignore')) - 1 
                else:
                    # Para XLSX, apenas usamos uma estimativa inicial alta
                    total_lines_file = CHUNK_SIZE * 50 
                    
            except Exception as e:
                total_lines_file = CHUNK_SIZE * 10 
            
            # Define o total de linhas real do arquivo (ou o que foi processado se for maior que a estimativa)
            st.session_state['total_lines'] = max(total_lines_file, lines_loaded_processed)
            
            st.session_state['file_name_context'] = normalize_text(os.path.splitext(selected_file_name)[0].upper().replace('_', ' ').replace('-', ' '))

            # Verifica se o carregamento foi completo ou se precisa continuar
            if lines_loaded_processed > 0 and lines_loaded_processed >= total_lines_file:
                st.session_state['df'] = agente_limpeza_dados(st.session_state['df'])
                st.session_state['processed_percentage'] = 100
                st.success(f"Processamento de **{selected_file_name}** concluído (total de linhas: {len(st.session_state['df'])}).")
                progress_bar = st.progress(1.0, text="Processamento finalizado. A ferramenta está pronta para uso!")
                st.rerun() 
            
            # Se o carregamento parcial ocorreu, precisamos continuar
            elif lines_loaded_processed > 0 and lines_loaded_processed < total_lines_file:
                st.info(f"Progresso parcial encontrado ({lines_loaded_processed} linhas). Continuaremos o processamento para as {total_lines_file - lines_loaded_processed} linhas restantes.")
                st.session_state['df_columns'] = st.session_state['df'].columns # Garante que as colunas sejam mantidas
                st.session_state['df'] = agente_limpeza_dados(st.session_state['df']) # Limpa a parte já carregada
            
            # --- INÍCIO DO NOVO PROCESSAMENTO (Se o carregamento falhou ou é a primeira vez) ---
            else:
                st.info(f"Iniciando novo processamento para **{selected_file_name}** ({st.session_state['total_lines']} linhas estimadas)...")
                st.session_state['df'] = None
                st.session_state['faiss_index'] = None
                st.session_state['documents'] = []
                st.session_state['conclusoes_historico'] = ""
                st.session_state['df_columns'] = None
                st.session_state['processed_percentage'] = 0
                st.session_state['cleaned_status'] = {}
                st.session_state['current_chunk_start'] = 0
                lines_loaded_processed = 0


            # Loop de processamento de chunks
            progress_bar = st.progress(lines_loaded_processed / st.session_state['total_lines'], 
                                       text=f"Criando embeddings e índice RAG... {lines_loaded_processed}/{st.session_state['total_lines']} linhas...")
            
            start_row = lines_loaded_processed
            
            while start_row < st.session_state['total_lines'] or start_row == 0:
                
                chunk_processed, msg = agente1_processa_arquivo_chunk(
                    st.session_state['zip_bytes'], 
                    selected_file_name, 
                    start_row, 
                    CHUNK_SIZE, 
                    st.session_state['df_columns'],
                    expected_num_cols
                )
                
                if chunk_processed is not None:
                    
                    # 1. Aplica limpeza e concatena
                    chunk_processed = agente_limpeza_dados(chunk_processed)
                    
                    if st.session_state['df'] is None:
                        st.session_state['df'] = chunk_processed
                        st.session_state['df_columns'] = chunk_processed.columns
                        # Re-calcula o número total de colunas esperado
                        expected_num_cols = len(st.session_state['df_columns'])
                    else:
                        # Garante que as colunas do chunk coincidam com o DF principal
                        if len(chunk_processed.columns) == len(st.session_state['df_columns']):
                            chunk_processed.columns = st.session_state['df_columns']
                        st.session_state['df'] = pd.concat([st.session_state['df'], chunk_processed], ignore_index=True)
                    
                    # 2. Cria índice RAG para o chunk
                    create_faiss_index_for_chunk(chunk_processed)
                    
                    # 3. Atualiza progresso
                    start_row += len(chunk_processed)
                    st.session_state['current_chunk_start'] = start_row
                    
                    # Recalibra o total de linhas se necessário
                    if len(chunk_processed) < CHUNK_SIZE and start_row < st.session_state['total_lines']:
                         st.session_state['total_lines'] = start_row
                         
                    progress_value = min(start_row / st.session_state['total_lines'], 1.0) if st.session_state['total_lines'] > 0 else 1.0
                    st.session_state['processed_percentage'] = progress_value * 100
                    
                    progress_bar.progress(progress_value, 
                                          text=f"Criando embeddings e índice RAG... {start_row}/{st.session_state['total_lines']} linhas - {st.session_state['processed_percentage']:.1f}%")
                    
                    save_progress(st.session_state['zip_hash'], st.session_state['df'], st.session_state['faiss_index'], st.session_state['documents'], st.session_state['total_lines'])
                    
                    # Condição de parada (processou o último chunk)
                    if len(chunk_processed) < CHUNK_SIZE:
                        st.session_state['total_lines'] = start_row # Fixa o total de linhas
                        break
                        
                else:
                    if "todos os lotes concluído" in msg:
                        st.session_state['total_lines'] = start_row # Fixa o total de linhas
                        break
                    st.error(msg)
                    break
            
            if st.session_state['df'] is not None and len(st.session_state['df']) > 0:
                st.session_state['total_lines'] = len(st.session_state['df'])
                st.success(f"Processamento de **{selected_file_name}** concluído! Total de linhas carregadas: {len(st.session_state['df'])}")
                progress_bar.progress(1.0, text="Processamento finalizado. A ferramenta está pronta para uso!")
                st.session_state['processed_percentage'] = 100
                st.rerun() 
            else:
                progress_bar.empty()
                st.error("Falha ao carregar o arquivo. Verifique se o formato está correto.")

st.markdown("---")

# --- Seção 2: Consulta à IA ---
st.header("2. Consultar a IA")

if st.session_state['df'] is None or st.session_state['processed_percentage'] < 5.0:
    st.info("Aguardando o processamento do arquivo para habilitar a consulta.")
else:
    if st.session_state['file_name_context'] and st.session_state['selected_file_name']:
        st.info(f"Analisando: **{st.session_state['selected_file_name']}**. Contexto: **{st.session_state['file_name_context']}**.")

    pergunta = st.text_area(
        "Pergunte em português sobre os dados:",
        value=st.session_state.get('user_query_input_widget', ""),
        placeholder="Ex: Qual o tipo de cada coluna? Me dê as estatísticas descritivas. Há correlação entre V1 e V2? Gere um boxplot para outliers.",
        height=100,
        key="user_query_input_widget"
    )
    
    # Botões de Ação
    col1_icons, col2_icons, col3_icons, col4_icons, col5_icons = st.columns([1, 1, 1, 1, 4])
    
    if col1_icons.button("Consultar (🔎)"):
        st.session_state['consultar_ia'] = True

    # Botões de display para ativar após a consulta (só aparecem após a primeira consulta)
    if st.session_state.get('codigo_gerado'):
        with col2_icons:
            if st.button("Gráfico (📊)"):
                st.session_state['habilitar_grafico'] = True
        with col3_icons:
            if st.button("Código (✍️)"):
                st.session_state['exibir_codigo'] = True
        with col4_icons:
            if st.button("PDF (📄)"):
                st.session_state['gerar_pdf'] = True

    st.markdown("---")
    
    # --- Lógica de Execução da Consulta (BLOBO ATUALIZADO) ---
    if 'consultar_ia' in st.session_state and st.session_state['consultar_ia']:
        st.session_state['consultar_ia'] = False
        
        if not st.session_state.get('gemini_api_key'):
            st.error("Por favor, insira e salve sua API Key do Gemini na barra lateral.")
        elif st.session_state['faiss_index'] is None or st.session_state['faiss_index'].ntotal == 0:
            st.warning("O índice RAG não foi criado. Por favor, processe o arquivo (clique em 'Analisar Arquivo' e aguarde o progresso).")
        else:
            pergunta_original = pergunta # Captura a pergunta original do widget
            
            # --- NOVA ETAPA DE CLARIFICAÇÃO ---
            with st.spinner("Clarificando sua pergunta e corrigindo possíveis erros de digitação..."):
                pergunta_clarificada = agente0_clarifica_pergunta(pergunta_original, st.session_state['gemini_api_key'])
            
            pergunta_para_ia = pergunta_clarificada # A partir daqui, usa a versão limpa
            
            # Exibe a correção se ela ocorreu
            if pergunta_para_ia != pergunta_original:
                 st.warning(f"Sua consulta foi clarificada para: **{pergunta_para_ia}**")
            # --- FIM DA NOVA ETAPA ---
            
            st.info(f"Análise realizada sobre **{st.session_state['processed_percentage']:.1f}%** dos dados já processados (total de **{len(st.session_state['df'])}** linhas).")
            
            with st.spinner("Gerando código e analisando dados..."):
                df_to_use = st.session_state['df']
                faiss_index = st.session_state['faiss_index']
                documents = st.session_state['documents']
                api_key = st.session_state['gemini_api_key']

                # 1. Recupera o Contexto (RAG) - USANDO A PERGUNTA CLARIFICADA
                retrieved_context = retrieve_context(pergunta_para_ia, faiss_index, documents)
                
                # 2. Gera Código e Conclusão - USANDO A PERGUNTA CLARIFICADA
                codigo_gerado, conclusoes = agente2_gera_codigo_pandas_eda(
                    pergunta_para_ia, 
                    api_key, 
                    df_to_use, 
                    retrieved_context, 
                    st.session_state['conclusoes_historico'],
                    st.session_state['file_name_context'] 
                )
                
                if conclusoes:
                    # Adiciona a nova conclusão ao histórico
                    st.session_state['conclusoes_historico'] += f"\n- {conclusoes}"
                
                st.session_state['codigo_gerado'] = codigo_gerado
                
                # 3. Executa o Código
                if codigo_gerado.startswith("Erro:"):
                    st.error(codigo_gerado)
                else:
                    resultado_texto, resultado_df, erro_execucao, img_bytes = executa_codigo_seguro(codigo_gerado, df_to_use)
                    
                    st.session_state['resultado_texto'] = resultado_texto
                    st.session_state['resultado_df'] = resultado_df
                    st.session_state['erro_execucao'] = erro_execucao
                    st.session_state['img_bytes'] = img_bytes
                    
                    if erro_execucao:
                        st.error(erro_execucao)
                    else:
                        st.subheader("Resultado da Análise:")
                        
                        # Exibe Tabela (Corrigido para evitar truncamento em colunas longas)
                        if resultado_df is not None and not resultado_df.empty:
                            
                            # Hack para colunas longas (Texto completo)
                            if 'INFORMAÇÃO' in resultado_df.columns:
                                st.markdown("##### Informação Detalhada (Texto Completo):")
                                
                                # Cria a tabela Markdown (Título e Corpo)
                                table_markdown = "| | INFORMAÇÃO |\n"
                                table_markdown += "| :--- | :--- |\n"
                                
                                # Adiciona as linhas do DataFrame
                                for index, row in resultado_df.iterrows():
                                    # Usa o índice como a primeira coluna e o texto como a segunda
                                    index_display = index if resultado_df.index.name is None else row.name
                                    table_markdown += f"| **{index_display}** | {row['INFORMAÇÃO']} |\n"
                                
                                st.markdown(table_markdown)
                                
                            else:
                                # Usa o st.dataframe normal para colunas numéricas/curtas
                                column_config = {col: st.column_config.Column(
                                    width="large",
                                    help="Descrição"
                                ) for col in resultado_df.columns}

                                if resultado_df.index.name:
                                    column_config[resultado_df.index.name] = st.column_config.TextColumn(
                                        width="small",
                                        help="Tipo/Índice"
                                    )
                                st.dataframe(resultado_df, use_container_width=True, column_config=column_config)
                        
                        # Exibe Gráfico (se houver)
                        if img_bytes and 'habilitar_grafico' not in st.session_state:
                            st.subheader("Gráfico Gerado:")
                            st.image(img_bytes, caption="Gráfico da Análise", use_container_width=True)
    
    # --- Lógica de Exibição dos Botões Secundários ---
    if 'exibir_codigo' in st.session_state and st.session_state['exibir_codigo']:
        st.session_state['exibir_codigo'] = False
        if st.session_state.get('codigo_gerado'):
            st.subheader("Cógido Python Gerado:")
            st.code(st.session_state['codigo_gerado'], language='python')
        else:
            st.warning("Nenhum código gerado. Faça uma consulta primeiro.")

    if 'habilitar_grafico' in st.session_state and st.session_state['habilitar_grafico']:
        st.session_state['habilitar_grafico'] = False
        if st.session_state.get('img_bytes'):
            st.subheader("Gráfico Gerado:")
            st.image(st.session_state['img_bytes'], caption="Gráfico da Análise", use_container_width=True)
        else:
            st.warning("Nenhum gráfico gerado na última consulta.")

    if 'gerar_pdf' in st.session_state and st.session_state['gerar_pdf']:
        st.session_state['gerar_pdf'] = False
        if st.session_state.get('codigo_gerado'):
            with st.spinner("Gerando PDF..."):
                _, _, pdf_bytes = agente3_formatar_apresentacao(st.session_state['resultado_texto'], st.session_state.get('resultado_df'), pergunta, st.session_state.get('img_bytes'))
                
                if pdf_bytes:
                    st.subheader("Download do Relatório:")
                    st.download_button(
                        label="Baixar Relatório em PDF",
                        data=pdf_bytes,
                        file_name="relatorio_eda.pdf",
                        mime="application/pdf"
                    )
                else:
                    st.warning("Não foi possível gerar o PDF. Verifique se as bibliotecas (ReportLab) estão instaladas.")
        else:
            st.warning("Por favor, execute uma consulta e analise os dados primeiro para gerar o PDF.")

st.markdown("---")

# --- Seção 3: Conclusões do Agente ---
st.header("3. Conclusões do Agente")
if st.session_state['conclusoes_historico']:
    st.markdown(st.session_state['conclusoes_historico'])
else:
    st.info("As conclusões da análise aparecerão aqui após a primeira consulta.")

st.markdown("---")
st.markdown("Adaptado por Daniel, com apoio de Rafael - grupo TENKAI")