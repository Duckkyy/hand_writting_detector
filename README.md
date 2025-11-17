# ✍️ Hand Writting Detection

本プロジェクトは、画像から手書きの数字を検出・認識するための Python ベースのツールです。  
YOLO などの検出モデル、および OCR 処理を組み合わせて、Excel や書類上の手書き修正部分を自動的に読み取ることを目的としています。

---

## 1. 環境構築

本プロジェクトは Conda 環境を使用しています。  
以下のコマンドで `environment.yml` を読み込み、必要な依存関係をすべてインストールしてください。

```bash
conda env create -f environment.yml
conda activate handwritting_detection
```

## 2. 環境変数（API Key を使用する場合）
本プロジェクトは .env ファイルを使用して Gemini API Key を安全に管理しています。
レポジトリには含まれないため、以下のように手動で作成してください。
```bash
nano .env
VLM_API_KEY="your-gemini-api-key"
```

## 3. 実行方法
環境構築後、各種スクリプトを実行することで手書き数字の検出・認識処理を行うことができます
```bash
python pipeline.py --image path/to/image --output path/to/json_file
```



