import io
import contextlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from helpers.normalize_text import normalize_text

def executa_codigo_seguro(codigo, df):
    """Executa o código Pandas/Matplotlib gerado em um ambiente isolado."""
    if codigo.startswith("Erro:"):
        return codigo, None, None, None

    output_stream = io.StringIO()
    local_vars = {'df': df, 'pd': pd, 'plt': plt, 'normalize_text': normalize_text, 'np': np} # Adiciona np
    img_bytes = None

    try:
        with contextlib.redirect_stdout(output_stream):
            # Adiciona o df de forma segura para o exec
            local_vars['df'] = df.copy() 
            exec(codigo, {"__builtins__": __builtins__}, local_vars)
        
        # --- Lógica Aprimorada de Captura de Gráfico ---
        # Captura apenas a primeira figura (esperamos que seja a figura com todos os subplots)
        if len(plt.get_fignums()) > 0:
            for fig_num in plt.get_fignums():
                plt.figure(fig_num)
                buf = io.BytesIO()
                
                # Garante que layout está ajustado para subplots
                try:
                    plt.tight_layout()
                except Exception:
                    pass
                    
                plt.savefig(buf, format="png")
                img_bytes = buf.getvalue()
                buf.close()
                plt.close(fig_num)
                break
        
        resultado_texto = output_stream.getvalue().strip()
        resultado_df = local_vars.get('resultado_df')

        # Normaliza Series para DataFrame
        if isinstance(resultado_df, pd.Series):
            resultado_df = resultado_df.reset_index()
            if len(resultado_df.columns) == 2 and 'index' in resultado_df.columns:
                resultado_df.columns = ['Categoria', 'Valor']
        
        # BLOCO CRÍTICO: GARANTE QUE TODO TEXTO SEJA CONVERTIDO EM DATAFRAME (TABELA)
        if resultado_df is None and resultado_texto and not img_bytes:
            # Cria um DataFrame de 1x1 com a resposta textual
            resultado_df = pd.DataFrame({'INFORMAÇÃO': [resultado_texto]})
            # Limpa o resultado_texto para que o Streamlit priorize resultado_df
            resultado_texto = ""

        return resultado_texto, resultado_df, None, img_bytes

    except Exception as e:
        error_message = f"Erro ao executar o código gerado pela IA:\n\n{e}\n\nCódigo que falhou:\n```python\n{codigo}\n```"
        return error_message, None, error_message, None