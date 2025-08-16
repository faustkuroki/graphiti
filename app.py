import os, json, time
from datetime import datetime, timezone
from typing import Optional, Union, List

from fastapi import FastAPI, HTTPException, Query, Body
from pydantic import BaseModel
from neo4j import GraphDatabase

from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType

# КЛИЕНТЫ ИЗ ТВОЕГО ФОРКА
from graphiti_core.llm_client.gemini_client import GeminiClient
from graphiti_core.embedder.gemini import GeminiEmbedder
from graphiti_core.cross_encoder.gemini_reranker_client import GeminiRerankerClient

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://neo4j:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
if not GOOGLE_API_KEY:
    raise RuntimeError("Set GOOGLE_API_KEY (или GEMINI_API_KEY)")

LLM_MODEL    = os.getenv("GEMINI_LLM_MODEL", "gemini-2.0-flash")
EMB_MODEL    = os.getenv("GEMINI_EMB_MODEL", "text-embedding-004")
RERANK_MODEL = os.getenv("GEMINI_RERANK_MODEL", "gemini-2.0-flash")

# Инициализируем Graphiti с нашими клиентами
graphiti = Graphiti(
    NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD,
    llm_client=GeminiClient(api_key=GOOGLE_API_KEY, model=LLM_MODEL),
    embedder=GeminiEmbedder(api_key=GOOGLE_API_KEY, embedding_model=EMB_MODEL),
    cross_encoder=GeminiRerankerClient(api_key=GOOGLE_API_KEY, model=RERANK_MODEL),
)

app = FastAPI(title="Graphiti Sidecar (Gemini via fork)")

def neo4j_driver():
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

# ---- health & bootstrap ----
@app.on_event("startup")
async def init_indices():
    for i in range(30):
        try:
            with neo4j_driver().session() as s:
                s.run("RETURN 1").single()
            break
        except Exception as e:
            print(f"[startup] bolt not ready ({i+1}/30): {e}")
            time.sleep(2)
    else:
        raise RuntimeError("Neo4j not reachable via bolt")
    await graphiti.build_indices_and_constraints()

@app.get("/healthz")
async def health():
    try:
        with neo4j_driver().session() as s:
            s.run("RETURN 1").single()
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(500, f"neo4j error: {e}")

# ---- модели ----
class EpisodeIn(BaseModel):
    user_id: Optional[str] = None
    content: Union[dict, str]
    type: str = "text"
    description: str = "user_event"
    reference_time: Optional[datetime] = None

class SearchIn(BaseModel):
    query: str
    user_id: Optional[str] = None
    center_node_uuid: Optional[str] = None
    limit: int = 8

# ---- CRUD-примеры (Cypher через свой драйвер) ----
@app.get("/neo4j/cypher")
def cypher(query: str = Query(...)):
    try:
        with neo4j_driver().session() as s:
            data = [r.data() for r in s.run(query)]
        return {"cypher": query, "result": data}
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/neo4j/node")
def add_node(label: str = Query(...), properties: dict = Body(...)):
    try:
        prop_str = ", ".join([f"{k}: ${k}" for k in properties.keys()])
        cypher = f"CREATE (n:{label} {{{prop_str}}}) RETURN n"
        with neo4j_driver().session() as s:
            res = s.run(cypher, **properties)
            nodes = [dict(record["n"]) for record in res]
        return {"created": True, "nodes": nodes}
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/neo4j/node")
def get_nodes(label: str = Query(...), limit: int = Query(10)):
    try:
        with neo4j_driver().session() as s:
            res = s.run(f"MATCH (n:{label}) RETURN n LIMIT $limit", limit=limit)
            nodes = [dict(record["n"]) for record in res]
        return {"nodes": nodes}
    except Exception as e:
        raise HTTPException(500, str(e))

# ---- Episodes / Search (Graphiti API) ----
@app.post("/episodes")
async def add_episode(ep: EpisodeIn):
    now = datetime.now(timezone.utc)
    src = EpisodeType.text if ep.type == "text" else EpisodeType.json
    body = ep.content if isinstance(ep.content, str) else json.dumps(ep.content, ensure_ascii=False)
    ts_before = now
    await graphiti.add_episode(
        name="n8n-episode",
        episode_body=body,
        source=src,
        source_description=ep.description,
        reference_time=ep.reference_time or now,
    )
    ts_after = datetime.now(timezone.utc)
    if ep.user_id:
        cypher = """
        MERGE (u:User {id:$uid})
        WITH u
        MATCH (n)
        WHERE n.ingested_at >= $ts_from AND n.ingested_at < $ts_to
        SET n.tenantId = $uid
        MERGE (u)-[:OWNS]->(n)
        """
        with neo4j_driver().session() as sess:
            sess.run(cypher, uid=ep.user_id, ts_from=ts_before, ts_to=ts_after)
    return {"status":"ok"}

@app.post("/search")
async def search(inp: SearchIn):
    center_uuid = inp.center_node_uuid
    if inp.user_id and not center_uuid:
        with neo4j_driver().session() as s:
            rec = s.run("MERGE (u:User {id:$uid}) RETURN u.uuid as uuid", uid=inp.user_id).single()
            center_uuid = rec["uuid"] if rec else None
    res = await graphiti.search(inp.query, center_node_uuid=center_uuid)
    return {"facts": [r.model_dump() for r in res[: max(1, int(inp.limit))]]}

# ---- простые LLM/Embed/Rerank демо-ручки (с учетом наших интерфейсов) ----
@app.get("/ask")
async def ask(question: str = Query(...)):
    try:
        answer = await graphiti.llm_client.generate(question)   # async!
        return {"question": question, "answer": answer}
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/embed")
def embed(text: str = Query(...)):
    try:
        vecs = graphiti.embedder.embed([text])
        return {"text": text, "embedding": vecs[0]}
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/rerank")
def rerank(query: str = Query(...), documents: str = Query(..., description="Разделяй ||")):
    try:
        docs = documents.split("||")
        scores = graphiti.cross_encoder.score(query, docs)
        return {"query": query, "documents": docs, "scores": scores}
    except Exception as e:
        raise HTTPException(500, str(e))
