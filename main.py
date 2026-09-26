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

IMAGE_DIR = "images"
FIXED_PRODUCER_GROUP = "SKYXROS"
FIXED_PRODUCER_NAME = "大月 とき"
FIXED_PRODUCER_PATH = f"images/{FIXED_PRODUCER_GROUP}/{FIXED_PRODUCER_NAME}.jpg"
VECTOR_DIM = 64
LEARNING_RATE = 0.35

if os.path.exists(IMAGE_DIR):
    app.mount("/images", StaticFiles(directory=IMAGE_DIR), name="images")

class NextCandidatesRequest(BaseModel):
    step: int
    user_vector: Optional[List[float]] = None
    shown_ids: Optional[List[str]] = []

class UpdateVectorRequest(BaseModel):
    user_vector: List[float]
    selected_ids: List[str]

class Top9Request(BaseModel):
    user_vector: List[float]

# 属性軸の基準ベクトル定義
np.random.seed(100)
ATTR_VECTORS = {
    "cute": np.random.randn(VECTOR_DIM),       # 子犬・あざと度
    "handsome": np.random.randn(VECTOR_DIM),   # 王道ハンサム度
    "pure": np.random.randn(VECTOR_DIM),       # 透明感・儚さ
    "dark": np.random.randn(VECTOR_DIM),       # ダーク・色気度
    "idol": np.random.randn(VECTOR_DIM)        # 派手・アイドル度
}
for k in ATTR_VECTORS:
    ATTR_VECTORS[k] = ATTR_VECTORS[k] / np.linalg.norm(ATTR_VECTORS[k])

class ItemDatabase:
    def __init__(self):
        self.items = []
        self._load_items()

    def _load_items(self):
        np.random.seed(42)
        image_paths = glob.glob(os.path.join(IMAGE_DIR, "*", "*.jpg"))
        
        if not image_paths:
            image_paths = [
                f"images/SKYXROS/{FIXED_PRODUCER_NAME}.jpg",
                "images/GroupA/Member1.jpg",
                "images/GroupA/Member2.jpg",
                "images/GroupB/Member3.jpg",
                "images/GroupB/Member4.jpg",
            ]

        for path in image_paths:
            norm_path = unicodedata.normalize('NFC', path.replace('\\', '/'))
            parts = norm_path.split('/')
            
            if len(parts) >= 3:
                group = parts[-2]
                name = os.path.splitext(parts[-1])[0]
            else:
                group = "Unknown"
                name = os.path.splitext(parts[-1])[0]

            item_id = f"{group}__{name}"
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

def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(v1, v2) / (norm1 * norm2))

@app.post("/api/next-candidates")
def get_next_candidates(req: NextCandidatesRequest):
    all_items = db.get_all()
    if len(all_items) < 4:
        raise HTTPException(status_code=500, detail="Not enough images in database.")

    shown_set = set(req.shown_ids or [])
    available = [item for item in all_items if item["id"] not in shown_set]
    if len(available) < 4:
        available = all_items

    if req.step <= 3 or req.user_vector is None:
        indices = np.random.choice(len(available), size=4, replace=False)
        selected = [available[i] for i in indices]
    else:
        u_vec = np.array(req.user_vector)
        scored = [(cosine_similarity(u_vec, item["vector"]), item) for item in available]
        scored.sort(key=lambda x: x[0], reverse=True)
        top_pool = scored[:max(8, len(scored) // 2)]
        chosen_indices = np.random.choice(len(top_pool), size=4, replace=False)
        selected = [top_pool[i][1] for i in chosen_indices]

    return {
        "candidates": [
            {"id": item["id"], "group": item["group"], "name": item["name"], "image_url": item["image_url"]}
            for item in selected
        ]
    }

@app.post("/api/update-vector")
def update_vector(req: UpdateVectorRequest):
    if not req.selected_ids or len(req.selected_ids) > 2:
        raise HTTPException(status_code=400, detail="Must select 1 or 2 items.")

    u_vec = np.array(req.user_vector, dtype=float)
    if np.linalg.norm(u_vec) == 0:
        u_vec = np.random.randn(VECTOR_DIM)

    selected_vecs = [db.get_by_id(s_id)["vector"] for s_id in req.selected_ids if db.get_by_id(s_id)]
    if not selected_vecs:
        raise HTTPException(status_code=404, detail="Selected items not found.")

    target_vec = np.mean(selected_vecs, axis=0)
    target_vec = target_vec / np.linalg.norm(target_vec)

    new_u_vec = (1 - LEARNING_RATE) * u_vec + LEARNING_RATE * target_vec
    new_u_vec = new_u_vec / np.linalg.norm(new_u_vec)

    return {"user_vector": new_u_vec.tolist()}

@app.post("/api/top9")
def get_top9(req: Top9Request):
    all_items = db.get_all()
    u_vec = np.array(req.user_vector, dtype=float)

    # TOP 9 計算
    scored_items = []
    for item in all_items:
        sim = cosine_similarity(u_vec, item["vector"])
        match_score = int(np.clip((sim + 1) / 2 * 100, 50, 99))
        scored_items.append({
            "id": item["id"], "group": item["group"], "name": item["name"],
            "image_url": item["image_url"], "match_score": match_score
        })

    scored_items.sort(key=lambda x: x["match_score"], reverse=True)
    true_no1 = scored_items[0] if scored_items else None

    producer_item = {
        "id": f"{FIXED_PRODUCER_GROUP}__{FIXED_PRODUCER_NAME}",
        "group": FIXED_PRODUCER_GROUP, "name": FIXED_PRODUCER_NAME,
        "image_url": f"/{FIXED_PRODUCER_PATH}", "match_score": 100, "is_producer": True
    }

    other_items = [
        item for item in scored_items 
        if not (item["group"] == FIXED_PRODUCER_GROUP and item["name"] == FIXED_PRODUCER_NAME)
    ]
    final_top9 = [producer_item] + other_items[:8]

    # 属性分析
    attr_scores = {}
    for attr_name, attr_vec in ATTR_VECTORS.items():
        sim = cosine_similarity(u_vec, attr_vec)
        score = int(np.clip((sim + 1) / 2 * 100, 50, 98))
        attr_scores[attr_name] = score

    sorted_attrs = sorted(attr_scores.items(), key=lambda x: x[1], reverse=True)
    top1_attr, top1_score = sorted_attrs[0]
    top2_attr, top2_score = sorted_attrs[1]

    # 12タイプ判定と毒舌テキスト分岐
    if top1_score > 82 and (top1_score - top2_score) > 8:
        if top1_attr == "cute":
            melo_type_name = "あざと甘々子犬めろ顔"
            melo_comment = "あざとくて可愛い顔に上目遣いされたら何でも許すでしょ？ わかりやすくてチョロい。どうせ「僕のこと好き？」って甘い声で言われたら一発で落ちるタイプ。"
        elif top1_attr == "handsome":
            melo_type_name = "正統派スタイリッシュイケメンめろ顔"
            melo_comment = "結局ミーハーだから分かりやすい王道イケメンが好きなんだよね。クラスで一番モテる奴に振り回されて勝手に自滅するタイプだから、少しは警戒心持ちな。"
        elif top1_attr == "pure":
            melo_type_name = "消えちゃいそうな透明感儚げめろ顔"
            melo_comment = "「俺、いつかいなくなっちゃうかも…」みたいな薄幸そうな男に弱いよね？ 影のある男を「私が救ってあげなきゃ」って勘違いして、沼にハマる典型的なタイプ。"
        elif top1_attr == "dark":
            melo_type_name = "危険な香りのダーク色気めろ顔"
            melo_comment = "クズだって分かってるのに、ちょっと冷たくされた後に優しくされるとコロッといくだろ？ 危険な匂いのする男に人生狂わされるのが一番好きなタイプ。"
        else: # idol
            melo_type_name = "オーラ全開キラキラアイドルめろ顔"
            melo_comment = "ステージの上で一番輝いてる男の「特別なファン」になりたくて必死でしょ？ 営業トークだと分かってても「君だけだよ」に全財産注ぎ込むタイプ。"

    elif top1_score > 70 and top2_score > 65:
        pair = set([top1_attr, top2_attr])
        if pair == {"cute", "dark"}:
            melo_type_name = "ギャップで落とす小悪魔系沼めろ顔"
            melo_comment = "普段甘えてくるのにたまにドSな一面見せられたら狂うでしょ？ 自分の前でしか見せない裏の顔に優越感感じて、一生抜け出せなくなる一番危険なオタク。"
        elif pair == {"handsome", "pure"}:
            melo_type_name = "王子様オーラの国宝級美形めろ顔"
            melo_comment = "面食いの極み。顔面偏差値が高ければ高いほど良いと思ってない？ 遠くから眺めて神様扱いして、一言会話しただけで呼吸忘れてパニックになるタイプ。"
        elif pair == {"cute", "idol"}:
            melo_type_name = "天性の愛されあざとアイドルめろ顔"
            melo_comment = "プロ意識の高いファンサと甘え上手なレスに狂わされてるね。他担狩りされて「私だけを見てくれてる」って頭お花畑になって財布の紐ゆるゆるになるタイプ。"
        elif pair == {"dark", "pure"}:
            melo_type_name = "ミステリアスな病み系儚げめろ顔"
            melo_comment = "闇を抱えてそうな男のメンヘラ部分に付き合ってあげるのが好きでしょ？ 共依存に陥って「私がいないとこの人ダメになっちゃう」って自爆するタイプ。"
        elif pair == {"handsome", "dark"}:
            melo_type_name = "フェロモン溢れる大人の色気ハンサムめろ顔"
            melo_comment = "男の余裕と色気に弱すぎる。ちょっと低音ボイスで耳元で囁かれたり、強引に引き寄せられたりしたら即オチするでしょ。本能に素直になりすぎ。"
        else:
            melo_type_name = "罪深きハイブリッドめろ顔"
            melo_comment = "欲張りすぎでしょ。いろんなタイプのイケメンに目移りして、結局全員の沼に片足突っ込んで身動き取れなくなってるタイプ。"

    elif all(score > 68 for score in attr_scores.values()):
        melo_type_name = "全方位無敵の神バランスめろ顔"
        melo_comment = "何でも持ってる完璧な男が好きとか、理想高すぎて現実見えてる？ 全部の要素が高水準じゃないと満足できない、一番ワガママで強欲なタイプ。"
    else:
        melo_type_name = "何にも染まらない未知のミステリアスめろ顔"
        melo_comment = "自分の好みすらよく分かってないでしょ？ ちょっとミステリアスで読めない男に振り回されて「なんで気になっちゃうんだろう」って一人で勝手に沼るタイプ。"

    return {
        "top9": final_top9,
        "true_no1": true_no1,
        "attr_scores": attr_scores,
        "melo_type_name": melo_type_name,
        "melo_comment": melo_comment
    }

@app.get("/")
def read_root():
    if os.path.exists("index.html"):
        from fastapi.responses import FileResponse
        return FileResponse("index.html")
    return {"message": "メン地下めろ顔AI診断 API is running."}