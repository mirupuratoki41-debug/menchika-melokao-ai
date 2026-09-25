import os
import json
import random
import numpy as np
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="メン地下めろ顔AI診断 API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if os.path.exists("images"):
    app.mount("/images", StaticFiles(directory="images"), name="images")

@app.get("/")
def read_root():
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    return {"message": "index.html が見つかりません。"}

# --- AIが解析したデータの読み込み ---
items_db = []

def load_db():
    global items_db
    if os.path.exists("embeddings.json"):
        with open("embeddings.json", "r", encoding="utf-8") as f:
            items_db = json.load(f)
        print(f"AI顔特徴データ ({len(items_db)}件) を読み込みました！")
    else:
        print("警告: embeddings.json がありません。先に generate_embeddings.py を実行してください。")

load_db()

# --- リクエストデータの型定義 ---
class NextPairRequest(BaseModel):
    user_vector: Optional[List[float]] = None
    history_ids: Optional[List[str]] = []

class UpdateVectorRequest(BaseModel):
    user_vector: Optional[List[float]] = None
    selected_id: str

class Top9Request(BaseModel):
    user_vector: Optional[List[float]] = None


# --- API エンドポイント ---

@app.post("/api/next-pair")
def get_next_pair(req: NextPairRequest):
    history = set(req.history_ids or [])
    available_items = [item for item in items_db if item["id"] not in history]

    if len(available_items) < 2:
        available_items = items_db

    # 初回はランダム表示
    if req.user_vector is None or len(available_items) < 5:
        selected = random.sample(available_items, 2)
    else:
        # 2回目以降：好みの顔立ちに近い候補同士を接戦させて精度を上げる
        u_vec = np.array(req.user_vector)
        candidates = []
        for item in available_items:
            i_vec = np.array(item["vector"])
            sim = float(np.dot(u_vec, i_vec))
            candidates.append((sim, item))
        
        candidates.sort(key=lambda x: x[0], reverse=True)
        top_pool = [c[1] for c in candidates[:10]]
        selected = random.sample(top_pool, 2)

    return {
        "item1": selected[0],
        "item2": selected[1]
    }

@app.post("/api/update-vector")
def update_vector(req: UpdateVectorRequest):
    selected_item = next((item for item in items_db if item["id"] == req.selected_id), None)
    if not selected_item:
        raise HTTPException(status_code=404, detail="選択された画像が見つかりません")

    selected_vec = np.array(selected_item["vector"])

    if req.user_vector is None:
        new_vec = selected_vec
    else:
        current_vec = np.array(req.user_vector)
        # ユーザーの好みのベクトルを徐々に学習
        new_vec = current_vec * 0.65 + selected_vec * 0.35

    norm = np.linalg.norm(new_vec)
    if norm > 0:
        new_vec = new_vec / norm

    return {"user_vector": new_vec.tolist()}

@app.post("/api/top9")
def get_top9(req: Top9Request):
    if req.user_vector is None:
        results = random.sample(items_db, min(9, len(items_db)))
        for item in results:
            item["score"] = random.randint(70, 95)
        return {"top9": results}

    u_vec = np.array(req.user_vector)
    scores = []

    for item in items_db:
        i_vec = np.array(item["vector"])
        similarity = float(np.dot(u_vec, i_vec))
        # 類似度を % 表記に変換
        match_percentage = int(max(50, min(99, (similarity + 0.2) * 80)))
        
        item_copy = dict(item)
        item_copy["score"] = match_percentage
        scores.append(item_copy)

    scores.sort(key=lambda x: x["score"], reverse=True)
    return {"top9": scores[:9]}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)