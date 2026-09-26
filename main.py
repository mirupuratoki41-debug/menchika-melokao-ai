from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
from typing import List, Optional

app = FastAPI(title="Men-Chika Mero-Face AI Diagnosis API")

# CORS設定（フロントエンドからの通信を許可）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------
# 1. 12タイプ判定データ & 詳細解説・毒舌アドバイス
# ---------------------------------------------------------
TYPE_DETAILS = {
    "王道キラキラアイドル沼": {
        "description": "王道の爽やかさとアイドル性を備えた、誰もが一度はハマる正統派の沼。",
        "advice": "「こういうのが一番好きでしょ？」と見透かされています。王道だからこそ競合も多いですが、一回ハマったら二度と抜け出せません。"
    },
    "ギャップ小悪魔系沼": {
        "description": "あざとさとダークな魅力を併せ持ち、振り回されることで癖になる最高に危険な沼。",
        "advice": "「絶対幸せになれない」と分かっているのに好きになるタイプです。彼の計算通りのあざとさに喜んで騙されに行っていませんか？"
    },
    "ミステリアスダーク沼": {
        "description": "影のある雰囲気とクールな表情で、知れば知るほど奥が深くなる沼。",
        "advice": "「私だけが彼の理解者」と思い込みやすい危険信号。冷たくされた後のふとした笑顔に人生を狂わされないよう注意してください。"
    },
    "圧倒的ビジュアル狂い沼": {
        "description": "顔面の造形美だけで全てをねじ伏せる、圧倒的顔面偏差値の沼。",
        "advice": "中身や性格なんてどうでも良くなるレベルの顔面力。彼の顔を見ているだけでIQが下がっているので、お財布の紐は固く締めておきましょう。"
    },
    "母性くすぐり弟系沼": {
        "description": "ほっとけない可愛さと守りたくなる雰囲気で、甘やかしたくなる沼。",
        "advice": "「私が養わなきゃ」と思ったら末期症状です。彼はあなたの母性本能を的確に刺激して甘えてくるプロですよ。"
    },
    "塩対応オラオラ沼": {
        "description": "普段は冷たいのに、たまに見せるデレのギャップにやられる沼。",
        "advice": "「今日は優しかった…！」で喜べるのは完全に飼い慣らされている証拠です。たまのご褒美（デレ）に依存しないよう気を付けて。"
    },
    "沼らせリアコお兄さん": {
        "description": "包容力と大人の余裕で、気付いた時には本気で好きになっている沼。",
        "advice": "一番タチが悪い「自然体で優しいお兄さん」。全員に優しいだけなのに「自分だけ特別」と錯覚しないように！"
    },
    "わんこ系子犬沼": {
        "description": "無邪気な笑顔と素直なリアクションで、見ているだけで癒される沼。",
        "advice": "人懐っこい笑顔の裏で、あなたの反応をしっかり観察しています。無害そうな顔をした一番の沼男かもしれません。"
    },
    "デンジャラス危険系沼": {
        "description": "危険な香りとスリルに満ちた、一度踏み入れたら戻れない底なし沼。",
        "advice": "友達に相談したら100%「やめときな」と言われるタイプ。そのスリルと中毒性を楽しんでいるのはあなた自身です。"
    },
    "完璧主義プロアイドル沼": {
        "description": "プロ意識が高く、ステージ上のパフォーマンスで魅了する沼。",
        "advice": "ファンサービスも完璧で隙がありません。完璧すぎるがゆえに、彼の本当の素顔が見えなくて永遠に追い続けちゃいます。"
    },
    "サブカルツンデレ沼": {
        "description": "独自の空気感とこだわりを持ち、ツンとした態度の中に優しさが見える沼。",
        "advice": "「自分の世界観を持っている人」に弱いあなた。彼のマニアックな好みに合わせようとして自分を見失わないように。"
    },
    "未知のミステリアス": {
        "description": "既存の枠にはまらない、唯一無二の存在感を放つ未知の沼。",
        "advice": "あなたの好みがマニアックすぎるか、直感で選びすぎかも？でもその独特なフェチズムこそがあなたの個性です。"
    }
}

# ---------------------------------------------------------
# 2. アイドルデータベース（サンプル・テスト用データ）
# ---------------------------------------------------------
IDOL_DATABASE = [
    {
        "id": "producer_01",
        "name": "大月とき",
        "group": "プロデューサー",
        "image_url": "https://via.placeholder.com/300x300?text=Otsuki+Toki",
        "vector": [0.8, 0.9, 0.7, 0.6, 0.9],
        "scores": {"azato": 85, "handsome": 95, "haganse": 75, "dark": 65, "idol": 98},
        "is_producer": True
    },
    {
        "id": "idol_01",
        "name": "レン",
        "group": "グループA",
        "image_url": "https://via.placeholder.com/300x300?text=Ren",
        "vector": [0.9, 0.4, 0.3, 0.2, 0.9],
        "scores": {"azato": 90, "handsome": 60, "haganse": 40, "dark": 30, "idol": 92},
        "is_producer": False
    },
    {
        "id": "idol_02",
        "name": "カイ",
        "group": "グループB",
        "image_url": "https://via.placeholder.com/300x300?text=Kai",
        "vector": [0.2, 0.8, 0.9, 0.8, 0.5],
        "scores": {"azato": 30, "handsome": 88, "haganse": 90, "dark": 85, "idol": 60},
        "is_producer": False
    },
    {
        "id": "idol_03",
        "name": "リク",
        "group": "グループC",
        "image_url": "https://via.placeholder.com/300x300?text=Riku",
        "vector": [0.5, 0.6, 0.5, 0.9, 0.4],
        "scores": {"azato": 50, "handsome": 70, "haganse": 60, "dark": 92, "idol": 50},
        "is_producer": False
    }
]

# ---------------------------------------------------------
# 3. 補助関数（数学計算 & タイプ判定）
# ---------------------------------------------------------
def cosine_similarity(v1, v2):
    a = np.array(v1)
    b = np.array(v2)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        return 0.0
    return float(np.dot(a, b) / norm)

def update_user_vector(current_vec, idol_vec, learning_rate=0.3):
    c_arr = np.array(current_vec)
    i_arr = np.array(idol_vec)
    updated = c_arr + learning_rate * (i_arr - c_arr)
    return updated.tolist()

def determine_type(scores):
    azato = scores.get("azato", 50)
    handsome = scores.get("handsome", 50)
    haganse = scores.get("haganse", 50)
    dark = scores.get("dark", 50)
    idol = scores.get("idol", 50)

    if idol >= 85 and handsome >= 80:
        return "王道キラキラアイドル沼"
    elif azato >= 80 and dark >= 70:
        return "ギャップ小悪魔系沼"
    elif dark >= 85 and haganse >= 70:
        return "ミステリアスダーク沼"
    elif handsome >= 90:
        return "圧倒的ビジュアル狂い沼"
    elif azato >= 85 and haganse < 50:
        return "母性くぐり弟系沼"
    elif haganse >= 85 and azato < 50:
        return "塩対応オラオラ沼"
    elif handsome >= 75 and idol >= 75:
        return "沼らせリアコお兄さん"
    elif azato >= 75 and idol >= 70:
        return "わんこ系子犬沼"
    elif dark >= 80 and azato >= 60:
        return "デンジャラス危険系沼"
    elif idol >= 90:
        return "完璧主義プロアイドル沼"
    elif haganse >= 70 and dark >= 60:
        return "サブカルツンデレ沼"
    else:
        return "未知のミステリアス"

# ---------------------------------------------------------
# 4. データモデル
# ---------------------------------------------------------
class SelectionRequest(BaseModel):
    user_vector: Optional[List[float]] = None
    selected_idol_ids: List[str]  # スキップされた場合は選択された分のIDのみ入る

# ---------------------------------------------------------
# 5. API エンドポイント
# ---------------------------------------------------------
@app.get("/api/next-candidates")
async def get_next_candidates():
    """対戦用のアイドル候補を取得"""
    candidates = [item for item in IDOL_DATABASE if not item.get("is_producer")]
    if len(candidates) >= 2:
        selected = np.random.choice(candidates, size=2, replace=False).tolist()
    else:
        selected = candidates
    return {"candidates": selected}

@app.post("/diagnose")
async def diagnose(payload: SelectionRequest, background_tasks: BackgroundTasks):
    """全問回答後の診断結果算出"""
    user_vec = payload.user_vector if payload.user_vector else [0.5] * 5

    # 選択された画像のベクトルで学習更新（スキップされた問題のIDは入ってこないため学習スキップ）
    for idol_id in payload.selected_idol_ids:
        idol = next((item for item in IDOL_DATABASE if item["id"] == idol_id), None)
        if idol:
            user_vec = update_user_vector(user_vec, idol["vector"])

    rankings = []
    producer_item = None

    for idol in IDOL_DATABASE:
        sim = cosine_similarity(user_vec, idol["vector"])
        match_percent = min(98, max(60, int(sim * 100)))

        item = {
            "id": idol["id"],
            "name": idol["name"],
            "group": idol["group"],
            "image_url": idol["image_url"],
            "match_percent": match_percent,
            "scores": idol["scores"]
        }

        if idol.get("is_producer"):
            item["match_percent"] = 100
            producer_item = item
        else:
            rankings.append(item)

    rankings.sort(key=lambda x: x["match_percent"], reverse=True)

    # TOP 9 形成（第1位にプロデューサー固定）
    top_9 = []
    if producer_item:
        top_9.append(producer_item)

    for item in rankings:
        if len(top_9) < 9:
            top_9.append(item)

    # 5軸スコアの集計（TOP 9の平均値）
    avg_scores = {"azato": 0, "handsome": 0, "haganse": 0, "dark": 0, "idol": 0}
    for member in top_9:
        for key in avg_scores:
            avg_scores[key] += member["scores"].get(key, 0)

    for key in avg_scores:
        avg_scores[key] = min(98, max(50, int(avg_scores[key] / len(top_9))))

    final_type = determine_type(avg_scores)
    type_info = TYPE_DETAILS.get(final_type, TYPE_DETAILS["未知のミステリアス"])

    # ※スプレッドシートログ保存は一時停止中

    return {
        "user_vector": user_vec,
        "result_type": final_type,
        "type_description": type_info["description"],
        "toxic_advice": type_info["advice"],
        "radar_scores": avg_scores,
        "top_9": top_9
    }