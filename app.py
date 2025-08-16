from graphiti_core import Graphiti
from graphiti_core.llm_client.gemini_client import GeminiClient, LLMConfig
from graphiti_core.embedder.gemini import GeminiEmbedder, GeminiEmbedderConfig
from graphiti_core.cross_encoder.gemini_reranker_client import GeminiRerankerClient
import os

from fastapi import FastAPI, Query, Body
from fastapi.responses import JSONResponse

api_key = os.getenv("GOOGLE_API_KEY")
neo4j_uri = os.getenv("NEO4J_URI", "bolt://neo4j:7687")
neo4j_user = os.getenv("NEO4J_USER", "neo4j")
neo4j_pass = os.getenv("NEO4J_PASS", "password")

graphiti = Graphiti(
    neo4j_uri,
    neo4j_user,
    neo4j_pass,
    llm_client=GeminiClient(
        config=LLMConfig(api_key=api_key, model="gemini-2.0-flash")
    ),
    embedder=GeminiEmbedder(
        config=GeminiEmbedderConfig(api_key=api_key, embedding_model="embedding-001")
    ),
    cross_encoder=GeminiRerankerClient(
        config=LLMConfig(api_key=api_key, model="gemini-2.5-flash-lite-preview-06-17")
    )
)

app = FastAPI()

@app.get("/")
def root():
    return JSONResponse(content={"status": "Graphiti с Gemini успешно инициализирован!"})

@app.get("/ask")
def ask(question: str = Query(..., description="Вопрос для LLM")):
    try:
        response = graphiti.llm_client.chat(question)
        return JSONResponse(content={"question": question, "answer": response})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/embed")
def embed(text: str = Query(..., description="Текст для эмбеддинга")):
    try:
        embedding = graphiti.embedder.embed([text])
        return JSONResponse(content={"text": text, "embedding": embedding[0]})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/rerank")
def rerank(query: str = Query(..., description="Запрос"), documents: str = Query(..., description="Документы через ||")):
    try:
        docs = documents.split("||")
        reranked = graphiti.cross_encoder.rerank(query, docs)
        return JSONResponse(content={"query": query, "documents": docs, "reranked": reranked})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/neo4j/cypher")
def cypher(query: str = Query(..., description="Cypher-запрос для Neo4j")):
    try:
        result = graphiti._db.run(query)
        data = [record.data() for record in result]
        return JSONResponse(content={"cypher": query, "result": data})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# CRUD для узлов
@app.post("/neo4j/node")
def add_node(label: str = Query(..., description="Label для узла"), properties: dict = Body(..., description="Свойства узла")):
    try:
        prop_str = ", ".join([f"{k}: ${k}" for k in properties.keys()])
        cypher = f"CREATE (n:{label} {{{prop_str}}}) RETURN n"
        result = graphiti._db.run(cypher, properties)
        nodes = [record["n"] for record in result]
        return JSONResponse(content={"created": True, "nodes": [dict(node) for node in nodes]})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/neo4j/node")
def get_nodes(label: str = Query(..., description="Label узлов"), limit: int = Query(10, description="Лимит")):
    try:
        cypher = f"MATCH (n:{label}) RETURN n LIMIT $limit"
        result = graphiti._db.run(cypher, {"limit": limit})
        nodes = [record["n"] for record in result]
        return JSONResponse(content={"nodes": [dict(node) for node in nodes]})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.delete("/neo4j/node")
def delete_node(label: str = Query(..., description="Label"), property_key: str = Query(...), property_value: str = Query(...)):
    try:
        cypher = f"MATCH (n:{label} {{{property_key}: $property_value}}) DETACH DELETE n RETURN COUNT(n) as deleted"
        result = graphiti._db.run(cypher, {"property_value": property_value})
        deleted = [record["deleted"] for record in result]
        return JSONResponse(content={"deleted": deleted})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.put("/neo4j/node")
def update_node(label: str = Query(...), property_key: str = Query(...), property_value: str = Query(...), update: dict = Body(...)):
    try:
        set_str = ", ".join([f"n.{k} = ${k}" for k in update.keys()])
        cypher = f"MATCH (n:{label} {{{property_key}: $property_value}}) SET {set_str} RETURN n"
        params = {"property_value": property_value}
        params.update(update)
        result = graphiti._db.run(cypher, params)
        nodes = [record["n"] for record in result]
        return JSONResponse(content={"updated": [dict(node) for node in nodes]})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# CRUD для ребер
@app.post("/neo4j/edge")
def add_edge(from_label: str = Query(...), from_key: str = Query(...), from_value: str = Query(...),
             to_label: str = Query(...), to_key: str = Query(...), to_value: str = Query(...),
             rel_type: str = Query(...), properties: dict = Body({}, description="Свойства ребра")):
    try:
        prop_str = ", ".join([f"{k}: ${k}" for k in properties.keys()])
        cypher = (
            f"MATCH (a:{from_label} {{{from_key}: $from_value}}), (b:{to_label} {{{to_key}: $to_value}}) "
            f"CREATE (a)-[r:{rel_type} {{{prop_str}}}]->(b) RETURN r"
        )
        params = {"from_value": from_value, "to_value": to_value}
        params.update(properties)
        result = graphiti._db.run(cypher, params)
        rels = [record["r"] for record in result]
        return JSONResponse(content={"created": True, "edges": [dict(rel) for rel in rels]})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/neo4j/edge")
def get_edges(rel_type: str = Query(...), limit: int = Query(10)):
    try:
        cypher = f"MATCH ()-[r:{rel_type}]->() RETURN r LIMIT $limit"
        result = graphiti._db.run(cypher, {"limit": limit})
        rels = [record["r"] for record in result]
        return JSONResponse(content={"edges": [dict(rel) for rel in rels]})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.delete("/neo4j/edge")
def delete_edge(rel_type: str = Query(...), property_key: str = Query(...), property_value: str = Query(...)):
    try:
        cypher = f"MATCH ()-[r:{rel_type} {{{property_key}: $property_value}}]->() DELETE r RETURN COUNT(r) as deleted"
        result = graphiti._db.run(cypher, {"property_value": property_value})
        deleted = [record["deleted"] for record in result]
        return JSONResponse(content={"deleted": deleted})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.put("/neo4j/edge")
def update_edge(rel_type: str = Query(...), property_key: str = Query(...), property_value: str = Query(...), update: dict = Body(...)):
    try:
        set_str = ", ".join([f"r.{k} = ${k}" for k in update.keys()])
        cypher = f"MATCH ()-[r:{rel_type} {{{property_key}: $property_value}}]->() SET {set_str} RETURN r"
        params = {"property_value": property_value}
        params.update(update)
        result = graphiti._db.run(cypher, params)
        rels = [record["r"] for record in result]
        return JSONResponse(content={"updated": [dict(rel) for rel in rels]})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# Получить схему графа
@app.get("/neo4j/schema")
def get_schema():
    try:
        cypher = "CALL db.schema.visualization()"
        result = graphiti._db.run(cypher)
        data = [record.data() for record in result]
        return JSONResponse(content={"schema": data})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
