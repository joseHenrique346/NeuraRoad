# from neo4j import GraphDatabase
# from fastapi import FastAPI
# from pydantic import BaseModel
# from neo4j import GraphDatabase
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.preprocessing import Normalizer, LabelEncoder
# from sklearn.metrics.pairwise import cosine_similarity
# import numpy as np
# import json
# import os
# import spacy
# import requests
# import torch
# import torch.nn as nn
# import torch.optim as optim

import os
import json
import torch
import numpy as np
import spacy
import pickle
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import Normalizer, LabelEncoder
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from neo4j import GraphDatabase
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Request

nlp = spacy.load("pt_core_news_sm")

# Caminho do arquivo JSON
file_path = 'treinoJson/treino.json'

# https://console.groq.com/home

def reescrever_com_llm(resposta_ia, pergunta):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer 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",
        "Content-Type": "application/json"
    }

    prompt = (
        "Você é um assistente que reescreve respostas de forma mais humana e explicativa. "
        "IMPORTANTE: Você deve usar SOMENTE a informação fornecida abaixo. "
        "NÃO adicione exemplos, leis, artigos ou dados que não foram fornecidos. "
        "Responda de um jeito simples e sem fugir do assunto. "
        "Não invente coisas. "
        "Finja que você não recebe alguma resposta, responda como se fosse você falando. "
        "Seu papel é apenas reformular o texto, deixando-o mais fluido e compreensível.\n\n"
        f"Informação fornecida:\n{resposta_ia}"
    )

    data = {
        "model": "meta-llama/llama-4-scout-17b-16e-instruct",
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"Pergunta: {pergunta}"},
            {"role": "assistant", "content": f"Resposta da IA: {resposta_ia}"}
        ]
    }

    response = requests.post(url, headers=headers, json=data)

    if response.status_code == 200:
        resposta = response.json()
        return resposta['choices'][0]['message']['content']
    else:
        return f"Erro na API: {response.status_code}, {response.text}"

# Modelo de rede neural simples
class SimpleNN(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, 128)
        self.fc2 = torch.nn.Linear(128, 64)
        self.fc3 = torch.nn.Linear(64, output_size)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Utilitários de dados
def carregar_dados():
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            dados = json.load(f)
            dados.setdefault("dados_treino", [])
            dados.setdefault("novos_dados", [])
            return dados
    return {"dados_treino": [], "novos_dados": []}

def salvar_dados(dados):
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(dados, f, ensure_ascii=False, indent=4)

def entender_pergunta(pergunta):
    doc = nlp(pergunta.lower())
    return [token.lemma_ for token in doc if token.pos_ in ["VERB", "NOUN", "ADJ"] and not token.is_stop]

# FastAPI + modelos
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ou especifique o domínio exato do front, ex: ["http://localhost:5500"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TreinoData(BaseModel):
    pergunta: str
    query: str

class Pergunta(BaseModel):
    pergunta: str

# Neo4j
URI = "neo4j+s://4f900673.databases.neo4j.io"
AUTH = ("neo4j", "o_oB00gE9NkoBAOGcvp_Xs2ymTN3c44HVgu85qIEJVk")

def executar_query(session, query):
    try:
        result = session.run(query)
        return [dict(record) for record in result]
    except Exception as e:
        return {"erro": f"Erro ao executar a consulta: {e}"}

def gerar_resposta(query, resultado):
    if not resultado:
        return "Nenhuma informação encontrada para essa consulta."

    if isinstance(resultado, dict) and "erro" in resultado:
        return resultado["erro"]

    chaves_padrao = {"tipo", "gravidade", "valor", "artigo", "pontos"}
    
    resposta = "Multas encontradas:\n"
    for idx, item in enumerate(resultado, 1):
        partes = []
        if isinstance(item, dict):
            if "tipo" in item:
                partes.append(f"{item.get('tipo')}")
            if "artigo" in item:
                partes.append(f"Art. {item.get('artigo')}")
            if "gravidade" in item:
                partes.append(f"{item.get('gravidade')}")
            if "valor" in item:
                partes.append(f"R$ {item.get('valor'):.2f}")
            if "pontos" in item:
                partes.append(f"{item.get('pontos')} pontos")
            campos_extras = [f"{k.capitalize()}: {v}" for k, v in item.items() if k not in chaves_padrao]
            partes.extend(campos_extras)
        else:
            # Se item não for dict, trate como string simples
            partes.append(str(item))

        resposta += f"- {' | '.join(partes)}\n"

    return resposta.strip()


def treinar_modelo_completo(dados):
    perguntas = [item["pergunta"] for item in dados["dados_treino"]]
    queries = [item["query"] for item in dados["dados_treino"]]

    termos_proc = [" ".join(entender_pergunta(p)) for p in perguntas]

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(termos_proc)

    normalizer = Normalizer()
    X_norm = normalizer.fit_transform(X.toarray())

    encoder = LabelEncoder()
    y = encoder.fit_transform(queries)

    modelo = SimpleNN(X.shape[1], len(encoder.classes_))
    modelo.train()

    optimizer = torch.optim.Adam(modelo.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    entrada_tensor = torch.from_numpy(X_norm).float()
    target_tensor = torch.tensor(y, dtype=torch.long)

    epochs = 50
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = modelo(entrada_tensor)
        loss = criterion(output, target_tensor)
        loss.backward()
        optimizer.step()


    return modelo, encoder, vectorizer, normalizer

modelo_mem = None
encoder_mem = None
vectorizer_mem = None
normalizer_mem = None
dados_mem = {"dados_treino": []}

@app.on_event("startup")
def carregar_memoria_e_treinar():
    global dados_mem, modelo_mem, encoder_mem, vectorizer_mem, normalizer_mem

    # Carrega os dados do arquivo
    dados_mem = carregar_dados()

    if not dados_mem["dados_treino"]:
        print("Nenhum dado de treino encontrado no treino.json.")
        return

    # Re-treina o modelo com base nos dados do JSON
    modelo, encoder, vectorizer, normalizer = treinar_modelo_completo(dados_mem)

    modelo_mem = modelo
    encoder_mem = encoder
    vectorizer_mem = vectorizer
    normalizer_mem = normalizer

    print("Modelo carregado em memória com sucesso a partir do treino.json.")

# Endpoint para treinamento incremental
@app.post("/treinar/")
def treinar(data: TreinoData):
    # Adiciona o novo dado ao conjunto em memória
    dados_mem["dados_treino"].append({"pergunta": data.pergunta, "query": data.query})

    # Também salva no treino.json para persistência
    salvar_dados(dados_mem)

    # Re-treina o modelo com todos os dados
    perguntas = [item["pergunta"] for item in dados_mem["dados_treino"]]
    queries = [item["query"] for item in dados_mem["dados_treino"]]

    # Pré-processamento
    termos_proc = [" ".join(entender_pergunta(p)) for p in perguntas]

    # Vetorização
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(termos_proc)

    normalizer = Normalizer()
    X_norm = normalizer.fit_transform(X.toarray())

    # Codificação das queries
    encoder = LabelEncoder()
    y = encoder.fit_transform(queries)

    # Treinamento do modelo
    modelo = SimpleNN(X_norm.shape[1], len(encoder.classes_))
    modelo.train()
    optimizer = torch.optim.Adam(modelo.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    entrada_tensor = torch.from_numpy(X_norm).float()
    target_tensor = torch.tensor(y, dtype=torch.long)

    epochs = 50
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = modelo(entrada_tensor)
        loss = criterion(output, target_tensor)
        loss.backward()
        optimizer.step()


    # Atualiza variáveis globais
    global modelo_mem, encoder_mem, vectorizer_mem, normalizer_mem
    modelo_mem = modelo
    encoder_mem = encoder
    vectorizer_mem = vectorizer
    normalizer_mem = normalizer

    return {"mensagem": "Treinamento completo realizado com sucesso."}


@app.post("/consultar_multa/")
def consultar_multa(data: Pergunta):
    if not dados_mem["dados_treino"]:
        return {"erro": "Nenhuma pergunta de treino ainda foi cadastrada."}

    if modelo_mem is None:
        return {"erro": "Modelo ainda não treinado."}

    # Pré-processa a pergunta
    termos = entender_pergunta(data.pergunta)
    texto_proc = " ".join(termos)
    X = vectorizer_mem.transform([texto_proc])
    X_array = normalizer_mem.transform(X.toarray())

    # Previsão
    modelo_mem.eval()
    with torch.no_grad():
        entrada_tensor = torch.tensor(X_array, dtype=torch.float32)
        output = modelo_mem(entrada_tensor)
        predicted_idx = torch.argmax(output, dim=1).item()
        query_prevista = encoder_mem.inverse_transform([predicted_idx])[0]

    # Executa a query prevista no Neo4j
    try:
        with GraphDatabase.driver(URI, auth=AUTH) as driver:
            with driver.session() as session:
                resultado = executar_query(session, query_prevista)
    except Exception as e:
        return {"erro": f"Erro ao executar query no banco de dados: {str(e)}"}

    resposta = gerar_resposta(query_prevista, resultado)

    return {
        "query_prevista": query_prevista,
        "Response": resposta + "\n\n(IA) Resposta gerada com base nos dados de treino."
    }


#neo4j+s://4f900673.databases.neo4j.io neo4j+s://4f900673.databases.neo4j.io
#o_oB00gE9NkoBAOGcvp_Xs2ymTN3c44HVgu85qIEJVk o_oB00gE9NkoBAOGcvp_Xs2ymTN3c44HVgu85qIEJVk

#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