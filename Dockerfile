FROM python:3.12-slim

# C++ビルドツールやOpenCVに必要なシステムパッケージをインストール
# (libgl1-mesa-glx を libgl1 に変更)
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# サーバー起動コマンド
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]