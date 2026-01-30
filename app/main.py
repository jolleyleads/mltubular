from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import numpy as np

app = FastAPI(title='MLTubular – RAG-ready ML API')

# ---- Minimal Embedding Stub ----
def embed(text: str) -> np.ndarray:
    return np.random.rand(384)

# ---- Toy Vector Store ----
VECTOR_DB = []

def add_doc(text: str):
    VECTOR_DB.append((embed(text), text))

def retrieve(query: str, k: int = 3):
    q = embed(query)
    sims = [(np.dot(q, v), t) for v, t in VECTOR_DB]
    return [t for _, t in sorted(sims, reverse=True)[:k]]

add_doc('This system demonstrates Retrieval-Augmented Generation.')
add_doc('Designed for ML and LLM engineering interviews.')

class AskRequest(BaseModel):
    query: str

@app.post('/ask')
def ask(req: AskRequest):
    context = retrieve(req.query)
    return {
        'query': req.query,
        'retrieved_context': context,
        'llm_answer': 'Stubbed response using retrieved context'
    }
