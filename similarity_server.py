import faiss
import json
import numpy as np
from fastapi import FastAPI, Request
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import uvicorn

# === 初始化 ===
app = FastAPI()
model = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index("qa_index.faiss")

with open("qa_mapping.json", "r", encoding="utf-8") as f:
    qa_data = json.load(f)

# === 请求模型 ===
class Question(BaseModel):
    question: str
    top_k: int = 3  # 可选参数，默认返回 top 3

@app.post("/match")
def match_question(data: Question):
    user_question = data.question
    top_k = data.top_k

    embedding = model.encode([user_question], normalize_embeddings=True)
    scores, indices = index.search(np.array(embedding), top_k)

    results = []
    for idx in indices[0]:
        qa = qa_data[idx]
        results.append({
            "matched_question": qa["qa_text"],
            "matched_index": int(idx)
        })

    return {
        "input": user_question,
        "matches": results
    }

# 运行：uvicorn similarity_server:app --reload --port 8000