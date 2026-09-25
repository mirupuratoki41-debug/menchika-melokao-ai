import os
import json
import numpy as np
from PIL import Image

IMAGE_DIR = "images"
DB_FILE = "database.json"

def extract_features_from_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img_resized = img.resize((32, 32))
    img_array = np.array(img_resized) / 255.0
    return img_array.flatten().tolist()

def extract_all_features():
    if not os.path.exists(IMAGE_DIR):
        os.makedirs(IMAGE_DIR)
        return

    dataset = []
    supported_extensions = (".jpg", ".jpeg", ".png")
    image_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(supported_extensions)]

    print(f"{len(image_files)} 枚の画像を処理中...")

    for filename in image_files:
        path = os.path.join(IMAGE_DIR, filename)
        try:
            vector = extract_features_from_image(path)
            dataset.append({
                "id": filename,
                "image_url": f"/images/{filename}",
                "vector": vector
            })
            print(f"〇 成功: {filename}")
        except Exception as e:
            print(f"× スキップ ({filename}): {e}")

    with open(DB_FILE, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print(f"\n完了: {len(dataset)} 件のデータを保存しました。")

if __name__ == "__main__":
    extract_all_features()
    