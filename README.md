# API de Multas com IA + Neo4j

Este projeto é uma API em Python desenvolvida com **FastAPI**, que utiliza **rede neural (PyTorch)**, **processamento de linguagem natural (spaCy)** e **Neo4j** para responder perguntas sobre multas de trânsito.

A API recebe perguntas em linguagem natural (ex: *"Qual o valor da multa por dirigir sem cinto?"*), processa, encontra a query correta no banco de dados e retorna uma resposta estruturada.  
Além disso, conta com uma camada de **reescrita via LLM**, que transforma a resposta bruta em um texto mais humano e fácil de entender.

O projeto foi pensado para ser **incremental**: você pode adicionar novas perguntas e queries ao modelo, e ele aprende continuamente.

---

## 📌 Funcionalidades

- Consulta de multas em linguagem natural, com respostas mais humanas.  
- Rede neural simples (**PyTorch**) para previsão da query correta.  
- Aprendizado incremental — é possível treinar a API com novas perguntas e queries.  
- Integração com **Neo4j** para armazenar e consultar os dados das multas.  
- **CORS habilitado** para fácil integração com front-ends ou outras APIs (ex: consumo via C#).  
- Reescrita de respostas via **LLM** para deixar a experiência mais natural.  

---

## ⚙️ Tecnologias utilizadas

- Python 3.10+  
- FastAPI  
- PyTorch  
- spaCy (`pt_core_news_sm`)  
- scikit-learn  
- Neo4j  
- Requests  
- Uvicorn  

---

## 📂 Estrutura do projeto
```txt
├── treinoJson/
│ └── treino.json # Base de treino (perguntas + queries)
├── main.py # Código principal da API
├── requirements.txt # Dependências do projeto
└── README.md # Este arquivo
```



---

## Como rodar o projeto

### 1. Clonar o repositório
```bash
git clone https://github.com/seu-usuario/sua-api-multas.git
cd sua-api-multas
```

### 2. Criar e ativar ambiente virtual
```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
```

### 3. Instalar dependências
```bash
pip install -r requirements.txt
python -m spacy download pt_core_news_sm
```

### 4. Configurar variáveis
Neo4j: ajuste URI e AUTH no código com as credenciais do seu banco.
LLM: configure a API Key para a reescrita (já há suporte no código).

### 5. Rodar a API
```bash
uvicorn main:app --reload
```

A API ficará disponível em:
👉 http://localhost:8000
👉 Documentação automática: http://localhost:8000/docs

## 📊 Endpoints principais
### 🔹 Treinar modelo
Adiciona uma nova pergunta e sua query correspondente.

### POST /treinar/
Exemplo de body (JSON):
```json
{
  "pergunta": "Qual o valor da multa por dirigir sem cinto?",
  "query": "MATCH (m:Multa {tipo:'sem_cinto'}) RETURN m.tipo AS tipo, m.valor AS valor, m.artigo AS artigo, m.pontos AS pontos"
}
```
Resposta:
```json
{
  "mensagem": "Treinamento completo realizado com sucesso."
}
```

## 🧩 Fluxo de funcionamento

Usuário faz uma pergunta → ex: "Qual o valor da multa por excesso de velocidade?"
Processamento NLP → extração de palavras-chave (verbos, substantivos, adjetivos).
Modelo neural (PyTorch) → prevê qual query do Neo4j deve ser executada.
Consulta ao Neo4j → busca as informações no banco.
Resposta gerada → a resposta bruta é reescrita por LLM para ficar mais fluida.
API retorna → JSON com a resposta humanizada.

## Observações importantes

O modelo só entende perguntas que estejam relacionadas às perguntas de treino cadastradas.
Para melhorar a precisão, alimente o treino.json com diversas variações de perguntas para a mesma query.
O LLM só reescreve respostas: ele não inventa novas informações.
Se não houver dados de treino, a API retornará erro.
