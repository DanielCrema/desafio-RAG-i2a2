import google.generativeai as genai

def agente0_clarifica_pergunta(pergunta_original, api_key):
    """Usa o Gemini para corrigir erros de digitação e clarificar a intenção."""
    if not api_key:
        return pergunta_original

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        prompt = f"""
# INSTRUÇÕES:
Você é um Clarificador de Consultas. Sua única função é corrigir erros de digitação e tornar a consulta do usuário o mais clara e objetiva possível, SEM alterar o significado original. Sua saída DEVE ser APENAS a consulta corrigida/clarificada.

# EXEMPLOS DE CORREÇÃO:
USUÁRIO: "Qual o tipu de cada colna?"
SAÍDA: "Qual o tipo de cada coluna?"

USUÁRIO: "ttata o arquiu?"
SAÍDA: "Do que se trata o arquivo?"
        
# CONSULTA ORIGINAL DO USUÁRIO:
{pergunta_original}

# CONSULTA CLARIFICADA:
"""
        response = model.generate_content(prompt)
        # Limita para garantir que seja apenas uma frase
        return response.text.strip().split('\n')[0]
    
    except Exception:
        # Em caso de erro, retorna a pergunta original para não bloquear o fluxo
        return pergunta_original