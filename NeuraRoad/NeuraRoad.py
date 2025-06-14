﻿from neo4j import GraphDatabase

URI = "neo4j+s://4f900673.databases.neo4j.io"
AUTH = ("neo4j", "o_oB00gE9NkoBAOGcvp_Xs2ymTN3c44HVgu85qIEJVk")
try:
    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        driver.verify_connectivity()
        print("✅ Conexão segura estabelecida com sucesso!")

except Exception as e:
    print(f"❌ Falha na conexão: {e}")





import spacy

nlp = spacy.load("pt_core_news_sm")

def entender_pergunta(pergunta):
    doc = nlp(pergunta.lower())
    termos_chave = [
        token.lemma_ for token in doc
        if token.pos_ in ["VERB", "NOUN", "ADJ"]
        and not token.is_stop
    ]
    return termos_chave

print(entender_pergunta("Quais são as multas mais caras?"))




import requests
import json

#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

#https://console.groq.com/home

def reescrever_com_llm(resposta_ia, pergunta, api_key):
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





from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import Normalizer, LabelEncoder
import numpy as np
import json
import os
import spacy
import torch
import torch.nn as nn
import torch.optim as optim

# Carrega modelo do spaCy para português
nlp = spacy.load("pt_core_news_sm")

# 1. Dados iniciais padrão
DADOS_INICIAIS = {
    "dados_treino": [
        ("Quais são as multas mais comuns?", "MATCH (m:Multa) RETURN m.tipo AS tipo, m.gravidade AS gravidade, m.valor AS valor, m.artigo AS artigo, m.pontos AS pontos LIMIT 3"),
        ("Quais infrações são mais frequentes?", "MATCH (m:Multa) RETURN m.tipo AS tipo, m.gravidade AS gravidade, m.valor AS valor, m.artigo AS artigo, m.pontos AS pontos LIMIT 3"),
        ("Qual a multa mais cara?", "MATCH (m:Multa) RETURN m.tipo AS tipo, m.gravidade AS gravidade, m.valor AS valor, m.artigo AS artigo, m.pontos AS pontos ORDER BY m.valor DESC LIMIT 1"),
        ("Qual é a multa de maior valor?", "MATCH (m:Multa) RETURN m.tipo AS tipo, m.gravidade AS gravidade, m.valor AS valor, m.artigo AS artigo, m.pontos AS pontos ORDER BY m.valor DESC LIMIT 1"),
        ("Mostre multas por excesso de velocidade", "MATCH (m:Multa) WHERE toLower(m.tipo) CONTAINS 'excesso' RETURN m.tipo AS tipo, m.gravidade AS gravidade, m.valor AS valor, m.artigo AS artigo, m.pontos AS pontos"),
        ("Quais multas estão relacionadas a excesso de velocidade?", "MATCH (m:Multa) WHERE toLower(m.tipo) CONTAINS 'excesso' RETURN m.tipo AS tipo, m.gravidade AS gravidade, m.valor AS valor, m.artigo AS artigo, m.pontos AS pontos"),
        ("Liste multas gravíssimas", "MATCH (m:Multa) WHERE toLower(m.gravidade) = 'gravíssima' RETURN m.tipo AS tipo, m.gravidade AS gravidade, m.valor AS valor, m.artigo AS artigo, m.pontos AS pontos"),
        ("Quais infrações são consideradas gravíssimas?", "MATCH (m:Multa) WHERE toLower(m.gravidade) = 'gravíssima' RETURN m.tipo AS tipo, m.gravidade AS gravidade, m.valor AS valor, m.artigo AS artigo, m.pontos AS pontos")
    ],
    "novos_dados": []
}

# 2. Funções de persistência
def salvar_dados():
    with open("treino.json", "w") as f:
        json.dump({
            "dados_treino": dados_treino,
            "novos_dados": novos_dados
        }, f, indent=2)

def carregar_dados():
    try:
        if os.path.exists("treino.json"):
            with open("treino.json", "r") as f:
                dados = json.load(f)
                if "dados_treino" in dados and "novos_dados" in dados:
                    print(f"[✔] Dados carregados de treino.json (Total: {len(dados['dados_treino'])} exemplos)")
                    return dados["dados_treino"], dados["novos_dados"]
                else:
                    print("[!] treino.json incompleto, carregando dados padrão.")
        else:
            print("[!] treino.json não encontrado, carregando dados padrão.")
    except Exception as e:
        print(f"[✖] Erro ao carregar treino.json: {e}")
    return DADOS_INICIAIS["dados_treino"], DADOS_INICIAIS["novos_dados"]

# 3. Função de entendimento de pergunta
def entender_pergunta(pergunta):
    doc = nlp(pergunta.lower())
    return [token.lemma_ for token in doc if token.pos_ in ["VERB", "NOUN", "ADJ"] and not token.is_stop]

# 4. Modelo de rede neural com PyTorch
class SimpleNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

# 5. Inicialização do sistema
dados_treino, novos_dados = carregar_dados()
perguntas_treino = [p for p, _ in dados_treino]
queries_treino = [q for _, q in dados_treino]
perguntas_processadas = [" ".join(entender_pergunta(p)) for p in perguntas_treino]

vectorizer = TfidfVectorizer(min_df=1, token_pattern=r'(?u)\b\w\w+\b')
normalizer = Normalizer()
encoder = LabelEncoder()

X_train = vectorizer.fit_transform(perguntas_processadas)
X_train = normalizer.fit_transform(X_train.toarray())
y_train_encoded = encoder.fit_transform(queries_treino)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_encoded, dtype=torch.long)

modelo_pytorch = SimpleNN(X_train.shape[1], len(encoder.classes_))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(modelo_pytorch.parameters(), lr=0.001)

for epoch in range(100):
    modelo_pytorch.train()
    optimizer.zero_grad()
    output = modelo_pytorch(X_train_tensor)
    loss = criterion(output, y_train_tensor)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

# 6. Funções auxiliares
def executar_query(driver, query):
    try:
        with driver.session() as session:
            result = session.run(query)
            return [dict(record) for record in result]
    except Exception as e:
        print(f"Erro ao executar query: {e}")
        return []

def gerar_resposta(query, resultado):
    if not resultado:
        return "Nenhuma informação encontrada para essa consulta."

    if "ORDER BY m.valor DESC" in query or "ORDER BY m.valor ASC" in query:
        m = resultado[0]
        return f"A multa é: {m.get('tipo')} (Art. {m.get('artigo')})\nGravidade: {m.get('gravidade')} | Valor: R$ {m.get('valor'):.2f} | Pontos: {m.get('pontos')}"

    if "RETURN m.tipo AS tipo" in query and "LIMIT" not in query:
        resposta = "Multas encontradas:\n"
        for m in resultado:
            resposta += f"- {m.get('tipo')} | Art. {m.get('artigo')} | {m.get('gravidade')} | R$ {m.get('valor'):.2f} | {m.get('pontos')} pontos\n"
        return resposta.strip()

    return str(resultado)

def aprender(pergunta, query_correta):
    global X_train, y_train_tensor, perguntas_treino, queries_treino

    novos_dados.append({"pergunta": pergunta, "query": query_correta})
    dados_treino.append((pergunta, query_correta))
    perguntas_treino.append(pergunta)
    queries_treino.append(query_correta)

    pergunta_proc = " ".join(entender_pergunta(pergunta))
    X_novo = vectorizer.transform([pergunta_proc])
    X_novo = normalizer.transform(X_novo.toarray())
    X_train = np.vstack([X_train, X_novo])

    y_train_encoded = encoder.fit_transform(queries_treino)
    y_train_tensor = torch.tensor(y_train_encoded, dtype=torch.long)
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)

    modelo = SimpleNN(X_train.shape[1], len(encoder.classes_))
    optimizer = optim.Adam(modelo.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(100):
        modelo.train()
        optimizer.zero_grad()
        output = modelo(X_train_tensor)
        loss = criterion(output, y_train_tensor)
        loss.backward()
        optimizer.step()

    salvar_dados()
    return f"Exemplo adicionado: {pergunta}"

def consultar_multas(driver, pergunta, api_key):
    termos = entender_pergunta(pergunta)
    texto_proc = " ".join(termos)
    X = vectorizer.transform([texto_proc])
    X = normalizer.transform(X.toarray())

    modelo_pytorch.eval()
    with torch.no_grad():
        entrada_tensor = torch.tensor(X, dtype=torch.float32)
        output = modelo_pytorch(entrada_tensor)
        predicted_idx = torch.argmax(output, dim=1).item()
        query_prevista = encoder.inverse_transform([predicted_idx])[0]

    resultado = executar_query(driver, query_prevista)
    resposta = gerar_resposta(query_prevista, resultado)

    return resposta + "\n\n[IA] Resposta gerada com base nos dados de treino."





from neo4j import GraphDatabase
# import torch
# import torch.nn as nn
# import torch.optim as optim
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import Normalizer
import numpy as np
import json
import os

# Verificar as classes únicas de y_train_encoded
print(f"Classes únicas em y_train_encoded: {np.unique(y_train_encoded)}")

# Modelo simples de rede neural
class SimpleNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)  # Aqui, o output_size precisa ser igual ao número de classes
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)  # Saída com logits para cada classe
        return x

# Função de persistência de dados
def salvar_dados(dados_treino, novos_dados):
    with open("treino.json", "w") as f:
        json.dump({
            "dados_treino": dados_treino,
            "novos_dados": novos_dados
        }, f, indent=2)

def carregar_dados():
    try:
        if os.path.exists("treino.json"):
            with open("treino.json", "r") as f:
                dados = json.load(f)
                if "dados_treino" in dados and "novos_dados" in dados:
                    return dados["dados_treino"], dados["novos_dados"]
                else:
                    print("[!] treino.json incompleto, carregando dados padrão.")
        else:
            print("[!] treino.json não encontrado, carregando dados padrão.")
    except Exception as e:
        print(f"[✖] Erro ao carregar treino.json: {e}")
    return DADOS_INICIAIS["dados_treino"], DADOS_INICIAIS["novos_dados"]

# Inicialização de variáveis globais
modelo_pytorch = None
vectorizer = TfidfVectorizer()
normalizer = Normalizer()

# 1. Inicialização do modelo
def inicializar_modelo():
    global modelo_pytorch, perguntas_treino, queries_treino, novos_dados, X_train, vectorizer, normalizer

    try:
        dados_treino, novos_dados = carregar_dados()

        perguntas_treino = [item[0] for item in dados_treino]
        queries_treino = [item[1] for item in dados_treino]

        perguntas_processadas = [" ".join(entender_pergunta(p)) for p in perguntas_treino]

        X_train = vectorizer.fit_transform(perguntas_processadas)
        X_train = normalizer.fit_transform(X_train.toarray())

        X_tensor = torch.tensor(X_train, dtype=torch.float32)

        # Verificar que y_train_encoded tem o mesmo tamanho de X_train
        y_tensor = torch.tensor(y_train_encoded[:len(X_train)], dtype=torch.long) # Ajuste para garantir que tenha o mesmo tamanho de X_train

        # Ajustar o número de classes para a camada final do modelo
        output_size = len(np.unique(y_tensor))  # número de classes, que deve ser 4
        modelo_pytorch = SimpleNN(X_train.shape[1], output_size)

        # Função de perda adequada para classificação (CrossEntropyLoss)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(modelo_pytorch.parameters(), lr=0.001)

        # Treinamento do modelo
        for epoch in range(100):
            modelo_pytorch.train()
            optimizer.zero_grad()
            output = modelo_pytorch(X_tensor)
            loss = criterion(output, y_tensor)
            loss.backward()
            optimizer.step()

        print(f"[✔] Modelo treinado com {len(perguntas_treino)} exemplos.")
        return perguntas_treino, queries_treino  # Retornar apenas dois valores

    except Exception as e:
        print(f"[✖] Falha ao inicializar modelo: {e}")
        return DADOS_INICIAIS["dados_treino"], DADOS_INICIAIS["novos_dados"]

# 2. Função para entender a pergunta
def entender_pergunta(pergunta):
    doc = nlp(pergunta.lower())
    termos_chave = [
        token.lemma_ for token in doc
        if token.pos_ in ["VERB", "NOUN", "ADJ"]
        and not token.is_stop
    ]
    return termos_chave

# 3. Função para executar consulta no Neo4j
def executar_query(driver, query):
    try:
        with driver.session() as session:
            result = session.run(query)
            return [dict(record) for record in result]
    except Exception as e:
        print(f"Erro ao executar query: {e}")
        return []

# 4. Gerar resposta formatada
def gerar_resposta(query, resultado):
    if not resultado:
        return "Nenhuma informação encontrada para essa consulta."

    # Exemplo 1: Multa mais cara/barata
    if "ORDER BY m.valor DESC" in query or "ORDER BY m.valor ASC" in query:
        m = resultado[0]  # só uma multa retornada
        return (
            f"A multa é: {m.get('tipo')} (Art. {m.get('artigo')})\n"
            f"Gravidade: {m.get('gravidade')} | Valor: R$ {m.get('valor'):.2f} | Pontos: {m.get('pontos')}"
        )

    # Exemplo 2: Listagem de multas (ex: gravíssimas)
    if "RETURN m.tipo AS tipo" in query and "LIMIT" not in query:
        resposta = "Multas encontradas:\n"
        for m in resultado:
            resposta += (
                f"- {m.get('tipo')} | Art. {m.get('artigo')} | "
                f"{m.get('gravidade')} | R$ {m.get('valor'):.2f} | {m.get('pontos')} pontos\n"
            )
        return resposta.strip()

    # Fallback genérico
    return str(resultado)

# 5. Função de aprendizado incremental
def aprender(pergunta, query_correta):
    global X_train, queries_treino

    novos_dados.append({"pergunta": pergunta, "query": query_correta})
    dados_treino.append((pergunta, query_correta))
    perguntas_treino.append(pergunta)
    queries_treino.append(query_correta)

    pergunta_processada = " ".join(entender_pergunta(pergunta))
    X_novo = vectorizer.transform([pergunta_processada])
    X_novo = normalizer.transform(X_novo.toarray())
    X_train = np.vstack([X_train, X_novo])

    modelo.fit(X_train, queries_treino)
    salvar_dados(dados_treino, novos_dados)
    return f"Exemplo adicionado: {pergunta}"

# 6. Função de consulta com confirmação
def consultar_com_confirmacao(driver, pergunta, api_key):
    global X_train

    termos_chave = entender_pergunta(pergunta)
    texto_processado = " ".join(termos_chave)

    print(f"[ℹ] Processada como: {texto_processado}")

    if pergunta in perguntas_treino:
        idx = perguntas_treino.index(pergunta)
        query_prevista = queries_treino[idx]
        print(f"[ℹ] Pergunta já conhecida. Query associada:\n{query_prevista}")
    else:
        modelo_pytorch.eval()
        X = vectorizer.transform([texto_processado])
        X = normalizer.transform(X.toarray())
        X_tensor = torch.tensor(X, dtype=torch.float32)

        with torch.no_grad():
            output = modelo_pytorch(X_tensor)
            idx_previsto = torch.argmax(output).item()
            query_prevista = queries_treino[idx_previsto]
        print(f"[ℹ] Pergunta nova. Query prevista pelo modelo:\n{query_prevista}")

    # Executar a query prevista pelo modelo tradicional
    if query_prevista:
        resultado = executar_query(driver, query_prevista)
        resposta_gerada = gerar_resposta(query_prevista, resultado)
        print(f"\n🤖 Resposta da IA (modelo neural):\n{resposta_gerada}")
    else:
        print("[✖] Nenhuma query prevista pelo modelo.")
        resposta_gerada = ""

    # Chamar o LLM para mostrar a interpretação dele
    print("\n💡 Aguardando sugestão do LLM...")
    resposta_llm = reescrever_com_llm(resposta_gerada, pergunta, api_key)
    print(f"\n🧠 Sugestão do LLM:\n{resposta_llm}")

    # Confirmação
    while True:
        confirmar = input("\nA resposta da IA (modelo neural) está correta? (s/n): ").strip().lower()
        if confirmar in ["s", "n"]:
            break
        print("Digite apenas 's' para sim ou 'n' para não.")

    if confirmar == "s":
        if pergunta not in perguntas_treino:
            print(aprender(pergunta, query_prevista))
        else:
            print("[✔] Nada novo aprendido. A pergunta já está registrada corretamente.")
        return

    query_correta = input("Digite a query correta para esta pergunta:\n").strip()
    if query_correta:
        if pergunta in perguntas_treino:
            idx = perguntas_treino.index(pergunta)
            queries_treino[idx] = query_correta
            print("[✏️] Query atualizada para pergunta existente.")
        else:
            aprender(pergunta, query_correta)
            print("[➕] Nova pergunta aprendida com a query fornecida.")

        inicializar_modelo()
        salvar_dados()
    else:
        print("Consulta encerrada. Nenhuma alteração foi feita.")

# Execução principal
api_key = "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"

with GraphDatabase.driver(URI, auth=AUTH) as driver:
    try:
        driver.verify_connectivity()
        print("[✔] Conexão com Neo4j estabelecida com sucesso.")

        perguntas_treino, queries_treino = inicializar_modelo()

        while True:
            pergunta_teste = input("Digite sua pergunta (ou 'sair'): ").strip()
            if pergunta_teste.lower() in ["sair", "exit"]:
                print("Encerrando...")
                break
            resposta = consultar_com_confirmacao(driver, pergunta_teste, api_key)
            print(resposta)

    except Exception as erro:
        print(f"[✖] Erro na execução principal: {erro}")
