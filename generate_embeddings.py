import os
import json
import cv2
import numpy as np
from insightface.app import FaceAnalysis

def generate():
    print("AI顔認識モデルを準備中...")
    # 顔検出モデルの読み込み
    app = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))

    items = []
    images_dir = "images"

    if not os.path.exists(images_dir):
        print("エラー: 'images' フォルダが見つかりません。")
        return

    print("画像から顔の特徴量を抽出しています...")
    for root, dirs, files in os.walk(images_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path).replace("\\", "/")

                img = cv2.imread(full_path)
                if img is None:
                    continue

                # 写真から顔を検出してAIで解析
                faces = app.get(img)

                if len(faces) > 0:
                    # 一番大きく写っている顔を解析
                    faces = sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]), reverse=True)
                    embedding = faces[0].embedding
                    
                    # データの長さ（ベクトル）を整える
                    norm = np.linalg.norm(embedding)
                    if norm > 0:
                        embedding = embedding / norm

                    items.append({
                        "id": rel_path,
                        "image_url": rel_path,
                        "vector": embedding.tolist()
                    })
                    print(f"解析成功: {rel_path}")
                else:
                    print(f"顔が検出できませんでした: {rel_path}")

    # 解析結果を json ファイルとして保存
    with open("embeddings.json", "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)

    print(f"\n完了！合計 {len(items)} 件の顔データを作成しました。")

if __name__ == "__main__":
    generate()
    