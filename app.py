from graphiti_core import Graphiti
from graphiti_core.llm_client.gemini_client import GeminiClient, LLMConfig
from graphiti_core.embedder.gemini import GeminiEmbedder, GeminiEmbedderConfig
from graphiti_core.cross_encoder.gemini_reranker_client import GeminiRerankerClient
import os

# Получаем настройки из переменных окружения
api_key = os.getenv("GOOGLE_API_KEY")
eo4j_uri = os.getenv("NEO4J_URI", "bolt://neo4j:7687")
eo4j_user = os.getenv("NEO4J_USER", "neo4j")
eo4j_pass = os.getenv("NEO4J_PASS", "password")

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

# Ваш основной код здесь
print("Graphiti с Gemini успешно инициализирован!")
