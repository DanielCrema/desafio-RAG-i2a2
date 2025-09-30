import pandas as pd
import google.generativeai as genai
from helpers.normalize_text import normalize_text

def agente2_gera_codigo_pandas_eda(pergunta, api_key, df, retrieved_context=None, historico_conclusoes=None, file_context=None):
    """Gera código Pandas para EDA e a conclusão em linguagem natural."""
    if df is None:
        return "Erro: DataFrame não carregado. Faça o upload do arquivo primeiro.", None

    if not api_key:
        return "Erro: Chave da API do Gemini não fornecida.", None

    try:
        genai.configure(api_key=api_key)
        # MODELO ATUALIZADO PARA gemini-2.5-flash
        model = genai.GenerativeModel('gemini-2.5-flash')

        schema = '\n'.join([f"- {c} (dtype: {df[c].dtype})" for c in df.columns])

        pergunta_limpa = normalize_text(pergunta).upper()
        
        # 1. PERGUNTAS SOBRE TIPOS/ESQUEMA DE COLUNAS (Gera um DataFrame tabular VERTICAL)
        if any(keyword in pergunta_limpa for keyword in ["QUE TIPO DE DADOS", "QUAIS OS TIPOS DE COLUNAS", "DTYPE COLUNAS", "TIPOS DE DADOS NAS COLUNAS", "TIPOS DAS COLUNAS"]):
            
            codigo_gerado = f"""
# Cria um DataFrame vertical com duas colunas
schema_df = pd.DataFrame(df.dtypes).reset_index()
schema_df.columns = ['NOME_DA_COLUNA', 'TIPO_DE_DADO']

# Atribui para visualização tabular no Streamlit
resultado_df = schema_df
print(resultado_df.to_string(index=False)) # Imprime sem o índice para limpeza
"""
            conclusoes = "A análise revela o tipo de dado de cada coluna no conjunto de dados, auxiliando na verificação de consistência e na preparação para modelagem."
            return codigo_gerado, conclusoes
            
        # 2. PERGUNTAS SOBRE CONTEXTO GERAL (Gera apenas texto que será convertido em tabela 1x1)
        if any(keyword in pergunta_limpa for keyword in ["QUE SE TRATA O ARQUIVO", "CONTEUDO DO ARQUIVO", "REPRESENTA O ARQUIVO", "O QUE E ESSE DATASET"]):
            prompt_interpretacao = f"""
# PERSONA E OBJETIVO PRINCIPAL
Você é um Analista de Dados Sênior. Sua única função é INTERPRETAR e DESCREVER o conteúdo de um DataFrame. Sua saída DEVE ser APENAS uma descrição textual detalhada (qualitativa), sem código ou comentários.

# CONTEXTO DO DATAFRAME `df`
Esquema do DataFrame:
{schema}
CONTEXTO DO NOME DO ARQUIVO: '{file_context}'.

# PERGUNTA DO USUÁRIO
{pergunta}

# DESCRIÇÃO FINAL DO CONTEÚDO
"""
            response = model.generate_content(prompt_interpretacao)
            texto_limpo = response.text.replace("'", "\\'").replace('"', '\\"').replace('\n', ' ').strip()
            # O executor de código irá criar um resultado_df a partir deste print para garantir a tabela.
            codigo_gerado = f"print('{texto_limpo}')"
            
            # Conclusão customizada para contextualização (sem contagem errada de linhas)
            conclusoes_contexto = f"O arquivo '{file_context}' foi contextualizado. Ele contém {len(df)} registros."
            
            return codigo_gerado, conclusoes_contexto
            
        # 3. CORREÇÃO ESPECÍFICA PARA BOXPLOT e HISTOGRAMA (Evita múltiplas figuras e layout fixo)
        if any(keyword in pergunta_limpa for keyword in ["OUTLIER", "BOXPLOT", "DISPERSAO", "HISTOGRAMA", "DISTRIBUICAO"]):
             
            plot_type = 'boxplot' if any(k in pergunta_limpa for k in ["OUTLIER", "BOXPLOT"]) else 'hist'
            plot_func = 'df.boxplot(column=col, ax=axes[i], grid=False)' if plot_type == 'boxplot' else 'axes[i].hist(df[col].dropna(), bins=20, edgecolor="black")'
            plot_title = 'Análise de Outliers - Boxplots para Colunas Numéricas' if plot_type == 'boxplot' else 'Distribuição de Dados - Histogramas para Colunas Numéricas'
            
            codigo_gerado = f"""
import numpy as np

# 1. Identifica colunas numéricas
numerical_cols = df.select_dtypes(include=np.number).columns
num_plots = len(numerical_cols)

if num_plots == 0:
    print("Não há colunas numéricas para plotar.")
else:
    # 2. Calcula o layout dinâmico (max 4 colunas)
    n_cols = min(4, num_plots)
    n_rows = int(np.ceil(num_plots / n_cols))

    # 3. Cria a única figura principal com o layout dinâmico
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(4 * n_cols, 3 * n_rows))
    axes = axes.flatten() # Achata para iteração fácil

    # 4. Itera e plota
    for i, col in enumerate(numerical_cols):
        # Usa o eixo (axis) do subplot correto
        {plot_func}
        axes[i].set_title(col, fontsize=10)
        axes[i].tick_params(axis='x', rotation=45)

    # 5. Remove eixos vazios (se existirem)
    for j in range(num_plots, n_rows * n_cols):
        fig.delaxes(axes[j])

    plt.suptitle("{plot_title}", y=1.02, fontsize=14)
    plt.tight_layout()
"""
            conclusoes = f"Os gráficos de {plot_type} foram gerados para visualizar a dispersão dos dados e identificar potenciais problemas de distribuição ou outliers em cada coluna numérica do dataset."
            
            return codigo_gerado, conclusoes

        # --- LÓGICA: GERAÇÃO DE CÓDIGO (RAG - ANÁLISE GERAL) ---
        
        rag_context_str = f"\n\nCONTEXTO ADICIONAL DOS DADOS (RAG):\n{retrieved_context}" if retrieved_context else ""
        historico_conclusoes_str = f"\n\nHISTÓRICO DE ANÁLISE E CONCLUSÕES ANTERIORES:\n{historico_conclusoes}" if historico_conclusoes else ""
        file_context_str = f"\n\nCONTEXTO DO NOME DO ARQUIVO: '{file_context}'."
        
        prompt = f"""
# PERSONA E OBJETIVO PRINCIPAL
Você é um assistente especialista em Análise Exploratória de Dados (E.D.A.) com Pandas.
Sua única função é traduzir uma pergunta em linguagem natural para um código Python.
Você DEVE gerar apenas o código Python.

# CONTEXTO DO DATAFRAME `df`
Esquema do DataFrame:
{schema}
{file_context_str}
{rag_context_str}
{historico_conclusoes_str}

# REGRAS DE GERAÇÃO DE CÓDIGO (MUITO IMPORTANTE)
1.  **Sempre use `df` como o nome do DataFrame.**
2.  **NUNCA gere código para carregar (`pd.read_csv`, `pd.read_excel`, etc.) ou salvar o DataFrame `df`. Ele já está carregado e pronto para uso.**
3.  **Se o resultado for uma tabela de dados (DataFrame), SEMPRE atribua-o a `resultado_df` e imprima `resultado_df` (ex: `print(resultado_df.to_string())`).**
4.  **Para gráficos, use `matplotlib.pyplot` (importado como `plt`). Para gráficos com múltiplos subplots, use `plt.subplots()` com layout dinâmico (`numpy.ceil`).**
5.  **A saída final deve ser APENAS o código Python, sem explicações ou comentários, e JAMAIS inclua qualquer pergunta.**
6.  **EVITE usar zero à esquerda em números decimais inteiros (ex: use '8' em vez de '08') para evitar erro de sintaxe 'octal integers'.**

# PERGUNTA DO USUÁRIO
{pergunta}

# CÓDIGO PYTHON (PANDAS/MATPLOTLIB)
"""
        response = model.generate_content(prompt)
        codigo_gerado = response.text.replace("```python", "").replace("```", "").strip()
        
        # Agente 4: GERA AS CONCLUSÕES APÓS A ANÁLISE
        conclusoes_prompt = f"""
# PERSONA
Você é um analista de dados sênior e seu único trabalho é sintetizar os resultados de uma análise e fornecer conclusões ou insights claros e objetivos para o usuário.

# CONTEXTO
A pergunta do usuário foi: "{pergunta}"
O resultado da análise (Código Python) foi:
{codigo_gerado}

# TAREFA
Com base na pergunta do usuário e nos resultados, forneça uma ou duas frases de conclusão sobre o que foi descoberto. Não mencione o código. Apenas a conclusão.
"""
        conclusoes_response = model.generate_content(conclusoes_prompt)
        conclusoes = conclusoes_response.text

        return codigo_gerado, conclusoes
    except Exception as e:
        return f"Erro ao chamar a API do Gemini: {e}", None