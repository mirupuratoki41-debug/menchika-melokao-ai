import os
import glob
import unicodedata
import numpy as np
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="メン地下めろ顔AI診断")

# CORS設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 定数設定
IMAGE_DIR = "images"
FIXED_PRODUCER_GROUP = "SKYXROS"
FIXED_PRODUCER_NAME = "大月 とき"
FIXED_PRODUCER_PATH = f"images/{FIXED_PRODUCER_GROUP}/{FIXED_PRODUCER_NAME}.jpg"
VECTOR_DIM = 64  # 特徴量ベクトルの次元数
LEARNING_RATE = 0.35  # ベクトル更新時の学習率

# 静的ファイルの配信（画像・フロントエンド）
if os.path.exists(IMAGE_DIR):
    app.mount("/images", StaticFiles(directory=IMAGE_DIR), name="images")

# Pydantic モデル
class NextCandidatesRequest(BaseModel):
    step: int  # 1 ~ 10
    user_vector: Optional[List[float]] = None
    shown_ids: Optional[List[str]] = []

class UpdateVectorRequest(BaseModel):
    user_vector: List[float]
    selected_ids: List[str]  # 1人または2人のID

class Top9Request(BaseModel):
    user_vector: List[float]

# モック特徴量データベース管理クラス
class ItemDatabase:
    def __init__(self):
        self.items = []
        self._load_items()

    def _load_items(self):
        """画像ディレクトリからアイテムをロードし、固定の特徴量ベクトルを生成/割り当て"""
        np.random.seed(42)  # 再現性のためのシード固定
        image_paths = glob.glob(os.path.join(IMAGE_DIR, "*", "*.jpg"))
        
        # ディレクトリ構造がない場合のフォールバック用ダミー生成
        if not image_paths:
            print("Warning: No images found in 'images/'. Using fallback mock items.")
            image_paths = [
                f"images/SKYXROS/{FIXED_PRODUCER_NAME}.jpg",
                "images/GroupA/Member1.jpg",
                "images/GroupA/Member2.jpg",
                "images/GroupB/Member3.jpg",
                "images/GroupB/Member4.jpg",
            ]

        for path in image_paths:
            # Linux環境（Render）対策：日本語文字列をNFC正規化
            norm_path = unicodedata.normalize('NFC', path.replace('\\', '/'))
            parts = norm_path.split('/')
            
            if len(parts) >= 3:
                group = parts[-2]
                name = os.path.splitext(parts[-1])[0]
            else:
                group = "Unknown"
                name = os.path.splitext(parts[-1])[0]

            item_id = f"{group}__{name}"
            
            # ランダムだが一貫性のある特徴量ベクトル（正規化済み）
            # プロデューサー（大月とき）かそれ以外で均等配置
            vec = np.random.randn(VECTOR_DIM)
            vec = vec / np.linalg.norm(vec)

            self.items.append({
                "id": item_id,
                "group": group,
                "name": name,
                "image_url": f"/{norm_path}",
                "vector": vec
            })

    def get_all(self):
        return self.items

    def get_by_id(self, item_id: str):
        for item in self.items:
            if item["id"] == item_id:
                return item
        return None

db = ItemDatabase()

# ユーティリティ関数
def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(v1, v2) / (norm1 * norm2))

@app.post("/api/next-candidates")
def get_next_candidates(req: NextCandidatesRequest):
    """4人の候補を抽出（1〜3問目は多様性重視、4問目以降はベクトル近傍から選出）"""
    all_items = db.get_all()
    if len(all_items) < 4:
        raise HTTPException(status_code=500, detail="Not enough images in database (At least 4 required).")

    shown_set = set(req.shown_ids or [])
    available = [item for item in all_items if item["id"] not in shown_set]
    
    # リセットまたは全消費時のフォールバック
    if len(available) < 4:
        available = all_items

    selected = []

    if req.step <= 3 or req.user_vector is None:
        # 【探索フェーズ】属性の偏りを防ぐため、空間的に離れた4人、または完全にランダム選出
        indices = np.random.choice(len(available), size=4, replace=False)
        selected = [available[i] for i in indices]
    else:
        # 【最適化フェーズ】ユーザーベクトルの周辺から類似度が高め〜中程度の多様な4人を選出
        u_vec = np.array(req.user_vector)
        scored = []
        for item in available:
            sim = cosine_similarity(u_vec, item["vector"])
            scored.append((sim, item))
        
        # 類似度順にソート
        scored.sort(key=lambda x: x[0], reverse=True)
        
        # 上位帯・中位帯から適度にばらけさせて4人選出（局所解へのハマりを防止）
        top_pool = scored[:max(8, len(scored) // 2)]
        chosen_indices = np.random.choice(len(top_pool), size=4, replace=False)
        selected = [top_pool[i][1] for i in chosen_indices]

    # クライアント用レスポンス整形
    candidates = []
    for item in selected:
        candidates.append({
            "id": item["id"],
            "group": item["group"],
            "name": item["name"],
            "image_url": item["image_url"]
        })

    return {"candidates": candidates}

@app.post("/api/update-vector")
def update_vector(req: UpdateVectorRequest):
    """選択された1人または2人の重心方向へユーザーベクトルを引き寄せる"""
    if not req.selected_ids or len(req.selected_ids) > 2:
        raise HTTPException(status_code=400, detail="Must select 1 or 2 items.")

    u_vec = np.array(req.user_vector, dtype=float)
    if np.linalg.norm(u_vec) == 0:
        u_vec = np.random.randn(VECTOR_DIM)

    # 選択されたアイテムのベクトルを取得
    selected_vecs = []
    for s_id in req.selected_ids:
        item = db.get_by_id(s_id)
        if item:
            selected_vecs.append(item["vector"])

    if not selected_vecs:
        raise HTTPException(status_code=404, detail="Selected items not found.")

    # 2人選ばれた場合は重心（平均）ベクトルを求める
    target_vec = np.mean(selected_vecs, axis=0)
    target_vec = target_vec / np.linalg.norm(target_vec)

    # ユーザーベクトルを選択方向へ引き寄せる（指数移動平均）
    new_u_vec = (1 - LEARNING_RATE) * u_vec + LEARNING_RATE * target_vec
    new_u_vec = new_u_vec / np.linalg.norm(new_u_vec)

    return {"user_vector": new_u_vec.tolist()}

@app.post("/api/top9")
def get_top9(req: Top9Request):
    """TOP 9 マッチメンバーを判定（1位は大月とき固定）"""
    all_items = db.get_all()
    u_vec = np.array(req.user_vector, dtype=float)

    # 全メンバーとのコサイン類似度計算
    scored_items = []
    for item in all_items:
        sim = cosine_similarity(u_vec, item["vector"])
        # マッチ度%に変換 (0.5~1.0 を 50%~99% にスケーリング)
        match_score = int(np.clip((sim + 1) / 2 * 100, 50, 99))
        scored_items.append({
            "id": item["id"],
            "group": item["group"],
            "name": item["name"],
            "image_url": item["image_url"],
            "match_score": match_score
        })

    # マッチ度順にソート
    scored_items.sort(key=lambda x: x["match_score"], reverse=True)

    # 数値上の真の1位（シェアテキスト生成用）
    true_no1 = scored_items[0] if scored_items else None

    # 固定プロデューサー情報（#01 固定用）
    producer_item = {
        "id": f"{FIXED_PRODUCER_GROUP}__{FIXED_PRODUCER_NAME}",
        "group": FIXED_PRODUCER_GROUP,
        "name": FIXED_PRODUCER_NAME,
        "image_url": f"/{FIXED_PRODUCER_PATH}",
        "match_score": 100,
        "is_producer": True
    }

    # 大月ときを除外したリストから上位8名を取得
    other_items = [
        item for item in scored_items 
        if not (item["group"] == FIXED_PRODUCER_GROUP and item["name"] == FIXED_PRODUCER_NAME)
    ]
    top_8_others = other_items[:8]

    # #01(大月とき) + #02〜#09
    final_top9 = [producer_item] + top_8_others

    return {
        "top9": final_top9,
        "true_no1": true_no1
    }

# ルートアクセス時のフロントエンド配信
@app.get("/")
def read_root():
    if os.path.exists("index.html"):
        from fastapi.responses import FileResponse
        return FileResponse("index.html")
    return {"message": "メン地下めろ顔AI診断 API is running."}