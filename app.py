from graphiti_core import Graphiti
from graphiti_core.llm_client.gemini_client import GeminiClient, LLMConfig
from graphiti_core.embedder.gemini import GeminiEmbedder, GeminiEmbedderConfig
from graphiti_core.cross_encoder.gemini_reranker_client import GeminiRerankerClient
import os

from fastapi import FastAPI
from fastapi.responses import JSONResponse

# Получаем настройки из переменных окружения
api_key = os.getenv("GOOGLE_API_KEY")
neo4j_uri = os.getenv("NEO4J_URI", "bolt://neo4j:7687")
neo4j_user = os.getenv("NEO4J_USER", "neo4j")
neo4j_pass = os.getenv("NEO4J_PASS", "password")

graphiti = Graphiti(
    neo4j_uri,
    neo4j_user,
    neo4j_pass,
    llm_client=GeminiClient(
        config=LLMConfig(
            api_key=api_key,
            model="gemini-2.0-flash"
        )
    ),
    embedder=GeminiEmbedder(
        config=GeminiEmbedderConfig(
            api_key=api_key,
            embedding_model="embedding-001"
        )
    ),
    cross_encoder=GeminiRerankerClient(
        config=LLMConfig(
            api_key=api_key,
            model="gemini-2.5-flash-lite-preview-06-17"
        )
    )
)

app = FastAPI()

@app.get("/")
def root():
    return JSONResponse(content={"status": "Graphiti с Gemini успешно инициализирован!"})

# Пример запроса к LLM через Graphiti
@app.get("/ask")
def ask(question: str):
    try:
        # Пример использования llm_client напрямую
        response = graphiti.llm_client.chat(question)
        return JSONResponse(content={"question": question, "answer": response})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
