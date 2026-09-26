import math
import requests
from datetime import datetime
from typing import List, Optional, Dict
from pydantic import BaseModel
from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="メン地下めろ顔AI診断 API")

# CORS設定（フロントエンドからのアクセスを許可）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# ⚙️ 設定・定数
# ==========================================

# 1. Google Apps Script のデプロイURLをここに貼り付け
GAS_WEB_APP_URL = "https://script.google.com/macros/s/AKfycbxzlFZKBUI-obTPAnzwQS1H-i88lo8kq1lh11OP4diofry8yRtyIh-DQaDVyhwDWV5gCg/exec"

# 2. 学習率 (ユーザーの選択によるベクトル更新率)
LEARNING_RATE = 0.35

# 3. アイドルデータ例 (64次元ベクトルの代わりに5軸の基礎能力値を持つモデル)
# ※ 本番では64次元の画像特徴量ベクトルを配置します
IDOL_DATABASE = [
    {
        "id": "idol_001",
        "name": "大月とき",
        "group": "SKYXROS",
        "image_url": "https://example.com/images/toki.jpg",
        "is_producer": True, # プロデューサーフラグ
        "scores": {"azato": 80, "handsome": 95, "haganse": 70, "dark": 85, "idol": 98},
        "vector": [0.8, 0.95, 0.7, 0.85, 0.98] + [0.1] * 59 # 64次元ダミー
    },
    {
        "id": "idol_002",
        "name": "りく",
        "group": "SKYXROS",
        "image_url": "https://example.com/images/riku.jpg",
        "is_producer": False,
        "scores": {"azato": 95, "handsome": 60, "haganse": 80, "dark": 30, "idol": 90},
        "vector": [0.95, 0.6, 0.8, 0.3, 0.9] + [0.1] * 59
    },
    {
        "id": "idol_003",
        "name": "れん",
        "group": "SKYXROS",
        "image_url": "https://example.com/images/ren.jpg",
        "is_producer": False,
        "scores": {"azato": 40, "handsome": 85, "haganse": 60, "dark": 95, "idol": 80},
        "vector": [0.4, 0.85, 0.6, 0.95, 0.8] + [0.1] * 59
    },
    {
        "id": "idol_004",
        "name": "あおい",
        "group": "SKYXROS",
        "image_url": "https://example.com/images/aoi.jpg",
        "is_producer": False,
        "scores": {"azato": 60, "handsome": 70, "haganse": 95, "dark": 40, "idol": 85},
        "vector": [0.6, 0.7, 0.95, 0.4, 0.85] + [0.1] * 59
    },
    # 必要に応じてメンバーや候補画像を追加
]

# 12タイプの定義と毒舌メッセージマッピング
TYPE_DETAILS = {
    "あざと甘々子犬": {
        "description": "あざとさと子犬感を兼ね備えた天性の弟系めろ顔。",
        "advice": "「守ってあげたい」と思わせる顔に弱いあなた。あざとい計算の上目遣いにまんまと引っかかるタイプです。気づいた時にはお財布の紐がゆるゆるになってるので気をつけて。"
    },
    "正統派スタイリッシュイケメン": {
        "description": "無駄のない綺麗な顔立ちの王道イケメン。",
        "advice": "結局顔重視！分かりやすいイケメンが好きな素直なオタクです。整いすぎた顔を眺めているだけで白飯が食えるタイプ。直球の格好良さに永遠に狂わされてください。"
    },
    "消えちゃいそうな透明感儚げ": {
        "description": "色素薄い系でどこか影のある儚気な顔立ち。",
        "advice": "今にも消えていなくなりそうな美しさに弱い悲劇のヒロインタイプ。彼が理由もなくSNS更新を止めただけで「生きてる…？」と動悸がする重症度高めのオタクです。"
    },
    "危険な香りのダーク色気": {
        "description": "危険な雰囲気と色気が漂う沼系イケメン。",
        "advice": "「絶対に幸せになれない」と分かっているのに惹かれてしまうダメンズホイホイ。冷たい目で見下されたい願望を隠し持っていませんか？引き返すなら今のうちです。"
    },
    "オーラ全開キラキラアイドル": {
        "description": "ステージ上で輝く天性の天職アイドル。",
        "advice": "圧倒的レスとプロ意識に狂わされる王道ファン。レスをもらった瞬間「私だけに微笑んでくれた」と勘違いできる幸せな脳内お花畑素質を持っています。"
    },
    "ギャップ小悪魔系沼": {
        "description": "あざとさとダークな色気が交差する危険なギャップ顔。",
        "advice": "普段は甘えてくるのにたまに見せる冷たい表情のギャップに狂うタイプ。完全に翻弄されて彼の掌の上で転がされるのが大好物でしょ？お見通しです。"
    },
    "王子様オーラ国宝級美形": {
        "description": "整った顔立ちと圧倒的オーラを放つビジュアル勝者。",
        "advice": "面食いの頂点。彼と同じ空気を吸っているだけで徳を積んだ気になっているおめでたい性格です。高嶺の花すぎて一生遠くから眺めているのがお似合い。"
    },
    "天性の愛されあざとアイドル": {
        "description": "可愛さとプロ意識が融合した完璧なアイドル顔。",
        "advice": "彼のビジネス可愛さにまんまと騙されるタイプ。ファンサ1つで生活のすべてを狂わされるチョロすぎるオタク。でもそれが一番楽しいからOK！"
    },
    "ミステリアス病み系儚げ": {
        "description": "闇を感じさせる美しさと儚さを持ち合わせた顔立ち。",
        "advice": "「私だけが彼の理解者」という謎の使命感を抱きがちなオタク。病みツイートを見るたびに胸を痛めつつ、どこか喜んでいませんか？業が深いです。"
    },
    "フェロモン大人の色気ハンサム": {
        "description": "大人の色気とスタイリッシュさを持ち合わせた大人顔。",
        "advice": "色気に負けるタイプ。目線ひとつ、手つきひとつで心を撃ち抜かれて声が出なくなるタイプです。彼の前では全人類が赤子と化します。"
    },
    "全方位無敵神バランス": {
        "description": "すべての要素が高次元でまとまった最強のバランス顔。",
        "advice": "強欲！何もかもを手に入れたい欲張りなオタクです。隙のない完璧なビジュアルに文句のつけようがなく、最終的に平伏すしかなくなります。"
    },
    "未知のミステリアス": {
        "description": "どのカテゴリにも収まらない独特の魅力を秘めた顔。",
        "advice": "誰も理解できないマニアックな癖（ヘキ）の持ち主。周りが「どこが良いの？」と言えば言うほど燃え上がる、ひねくれ沼落ちタイプです。"
    }
}

# ==========================================
# 📐 ベクトル計算・ヘルパー関数
# ==========================================

def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """2つのベクトルのコサイン類似度を計算"""
    dot_product = sum(a * b for a, b in zip(v1, v2))
    norm_v1 = math.sqrt(sum(a * a for a in v1))
    norm_v2 = math.sqrt(sum(b * b for b in v2))
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    return dot_product / (norm_v1 * norm_v2)

def update_user_vector(user_vector: List[float], selected_vector: List[float]) -> List[float]:
    """選択された画像ベクトルでユーザーの好みベクトルを学習・更新"""
    return [
        u + LEARNING_RATE * (s - u)
        for u, s in zip(user_vector, selected_vector)
    ]

def determine_type(scores: Dict[str, int]) -> str:
    """スコア傾向から12タイプを判定"""
    azato = scores.get("azato", 0)
    handsome = scores.get("handsome", 0)
    haganse = scores.get("haganse", 0)
    dark = scores.get("dark", 0)
    idol = scores.get("idol", 0)

    # 複合・ハイブリッド判定
    if azato >= 75 and dark >= 75:
        return "ギャップ小悪魔系沼"
    if handsome >= 75 and idol >= 75:
        return "王子様オーラ国宝級美形"
    if azato >= 75 and idol >= 75:
        return "天性の愛されあざとアイドル"
    if haganse >= 75 and dark >= 75:
        return "ミステリアス病み系儚げ"
    if handsome >= 75 and dark >= 75:
        return "フェロモン大人の色気ハンサム"
    
    # 特殊判定（全全体的に高い場合）
    if all(s >= 70 for s in [azato, handsome, haganse, dark, idol]):
        return "全方位無敵神バランス"

    # 単一突出判定
    max_key = max(scores, key=scores.get)
    if max_key == "azato":
        return "あざと甘々子犬"
    elif max_key == "handsome":
        return "正統派スタイリッシュイケメン"
    elif max_key == "haganse":
        return "消えちゃいそうな透明感儚げ"
    elif max_key == "dark":
        return "危険な香りのダーク色気"
    elif max_key == "idol":
        return "オーラ全開キラキラアイドル"

    return "未知のミステリアス"

def log_to_spreadsheet(result_type: str, scores: dict):
    """Googleスプレッドシートへ非同期でログ送信を行う関数"""
    if "YOUR_GAS_DEPLOYMENT_ID" in GAS_WEB_APP_URL:
        print("[Warning] GAS_WEB_APP_URL が設定されていないためログ送信をスキップします。")
        return
    try:
        payload = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "result_type": result_type,
            "scores": scores
        }
        requests.post(GAS_WEB_APP_URL, json=payload, timeout=5)
    except Exception as e:
        print(f"[Error] スプレッドシート送信失敗: {e}")

# ==========================================
# 📥 リクエスト/レスポンスモデル
# ==========================================

class SelectionRequest(BaseModel):
    user_vector: Optional[List[float]] = None
    selected_idol_ids: List[str] # ユーザーが選択したアイドルのIDリスト

# ==========================================
# 🚀 API エンドポイント
# ==========================================

@app.get("/")
def read_root():
    return {"message": "メン地下めろ顔AI診断 API is running!"}

@app.post("/diagnose")
async def diagnose(payload: SelectionRequest, background_tasks: BackgroundTasks):
    """全10問選択後の最終結果算出エンドポイント"""
    
    # ユーザーベクトルの初期化 (未設定時は中立)
    user_vec = payload.user_vector if payload.user_vector else [0.5] * 64
    
    # 選択された画像のベクトルを取り出してユーザーベクトルを学習更新
    for idol_id in payload.selected_idol_ids:
        idol = next((item for item in IDOL_DATABASE if item["id"] == idol_id), None)
        if idol:
            user_vec = update_user_vector(user_vec, idol["vector"])

    # 全アイドルとの類似度（マッチ度%）を計算
    rankings = []
    producer_item = None

    for idol in IDOL_DATABASE:
        sim = cosine_similarity(user_vec, idol["vector"])
        # スコアを60%〜98%の範囲にマッピング
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
            # プロデューサー枠はマッチ度100%固定
            item["match_percent"] = 100
            producer_item = item
        else:
            rankings.append(item)

    # 類似度順にソート
    rankings.sort(key=lambda x: x["match_percent"], reverse=True)

    # MY TOP 9 の形成 (1位にプロデューサー枠を固定挿入)
    top_9 = []
    if producer_item:
        top_9.append(producer_item)
    
    # 残りを埋めてTOP9にする
    for item in rankings:
        if len(top_9) < 9:
            top_9.append(item)

    # 5軸スコアの集計（TOP9の平均値）
    avg_scores = {"azato": 0, "handsome": 0, "haganse": 0, "dark": 0, "idol": 0}
    for member in top_9:
        for key in avg_scores:
            avg_scores[key] += member["scores"].get(key, 0)
    
    for key in avg_scores:
        avg_scores[key] = min(98, max(50, int(avg_scores[key] / len(top_9))))

    # 12タイプ判定
    final_type = determine_type(avg_scores)
    type_info = TYPE_DETAILS.get(final_type, TYPE_DETAILS["未知のミステリアス"])

    # 📊 スプレッドシートへのログ保存をバックグラウンドタスクに追加（応答速度を下げないため）
    background_tasks.add_task(log_to_spreadsheet, final_type, avg_scores)

    return {
        "user_vector": user_vec,
        "result_type": final_type,
        "type_description": type_info["description"],
        "toxic_advice": type_info["advice"],
        "radar_scores": avg_scores,
        "top_9": top_9
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)