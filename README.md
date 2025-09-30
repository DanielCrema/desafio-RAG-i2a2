# Desafio I2A2 - Analisador de Dados com RAG
Analisador de Dados de Fraude em Cartão de Crédito desenvolvido como parte do desafio extra do curso de agentes inteligentes de IA Generativa do [Instituto I2A2](https://www.i2a2.academy/).
- Recebe planilhas em formato .csv ou .xlsx compactados em .zip
- Tokeniza os dados localmente, para RAG (Retrieval Augmented Generation)
- Por fim se pode questionar um `agente inteligente` para obter informações sobre os dados.

### Requisito:
Possuir GPU Nvidia CUDA robusta o suficiente para realizar tokenização.

## 🚀 Começando

As instruções abaixo vão te guiar na configuração do ambiente de desenvolvimento utilizando o [`uv`](https://github.com/astral-sh/uv), um gerenciador de pacotes e ambientes virtuais rápido para Python.

### 📦 Pré-Requisitos

- Python 3.10 ou superior instalado
- `uv` instalado:  
  Você pode instalar o `uv` com pip ou pipx:  

  ```bash
  pip install uv
  ```

## 🛠️ Configurando o Ambiente

#### 1. Crie e ative o ambiente virtual:

#### Linux
```bash
uv venv
source .venv/bin/activate
```

#### Windows
```bash
uv venv
source .venv\Scripts\activate
```

#### 2. Instale as dependências: 

```bash
uv pip install -r requirements.txt
```

## ▶️ Executando o Aplicativo
Após configurar o ambiente, execute o app com o comando:

```bash
streamlit run main.py
```

## 🗂️ Estrutura do Projeto
```bash
.
│
├── agents/                  # Agentes inteligentes
│
├── data/                    # Repositório de dados: Contém apenas um set para teste
│
├── helpers/                 # Utilitários
│
├── modules/                 # Módulos
│
├── rag_components/          # Tokenização, embedding, load, save e contextualização
│
├── sandboxing/              # Sandbox para rodar código gerado com segurança
│
└── main.py
        # Código principal da aplicação

```

## 🎓 Créditos

- Grupo TenkAI
- [Instituto I2A2](https://www.i2a2.academy/)