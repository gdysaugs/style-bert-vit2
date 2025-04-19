import os
import requests
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import logging

# .envファイルから環境変数を読み込む
load_dotenv()

# ロギング設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

TTS_API_URL = os.getenv("TTS_API_URL")

if not TTS_API_URL:
    logger.error("Environment variable TTS_API_URL is not set.")
    # アプリケーションを終了させる
    raise ValueError("TTS_API_URL must be set in the environment variables")


class TextPayload(BaseModel):
    text: str
    # オプションで他のパラメータ（例: スピーカーID、スタイルなど）も追加可能
    # speaker_id: int = 0
    # style: str = "Neutral"

@app.get("/")
def read_root():
    return {"message": "Backend API is running!"}

@app.post("/synthesize")
async def synthesize_speech(payload: TextPayload):
    """
    テキストを受け取り、TTSサービスに音声合成を依頼し、
    音声ファイル（ストリーミングレスポンス）を返すエンドポイント。
    """
    if not TTS_API_URL:
        logger.error("TTS_API_URL is not configured. Cannot process request.")
        raise HTTPException(status_code=500, detail="TTS service is not configured.")

    logger.info(f"Received synthesis request for text: {payload.text}")

    try:
        # Style-Bert-VITS2のserver_fastapi.pyの /voice エンドポイントに合わせる
        # Content-Type は multipart/form-data が期待される場合があるため要確認
        # ここでは仮にJSONで送信する例を示すが、実際のTTS API仕様に合わせること
        tts_payload = {
            "text": payload.text,
            # "speaker_id": payload.speaker_id, # 必要なら追加
            # "style": payload.style,         # 必要なら追加
            # "format": "wav" # 必要ならフォーマット指定
        }
        # Timeout を設定
        timeout_seconds = 180 # タイムアウトを180秒（3分）に延長

        logger.info(f"Sending request to TTS service at {TTS_API_URL}/voice")
        response = requests.post(f"{TTS_API_URL}/voice", json=tts_payload, stream=True, timeout=timeout_seconds)
        response.raise_for_status() # HTTPエラーがあれば例外を発生させる

        # Content-Type を確認して適切に設定する
        # 例: 'audio/wav' や 'audio/mpeg' など
        content_type = response.headers.get('Content-Type', 'application/octet-stream')
        logger.info(f"Received response from TTS service with status {response.status_code} and content-type {content_type}")

        # ストリーミングレスポンスとして返す
        return StreamingResponse(response.iter_content(chunk_size=1024), media_type=content_type)

    except requests.exceptions.RequestException as e:
        logger.error(f"Error connecting to TTS service: {e}")
        raise HTTPException(status_code=503, detail=f"TTS service connection error: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")

# CORS設定 (開発用にすべて許可 - 本番では制限すること)
from fastapi.middleware.cors import CORSMiddleware

origins = [
    "http://localhost",      # React開発サーバー(デフォルト)
    "http://localhost:8080", # docker-compose.ymlで設定したホストポート
    "http://127.0.0.1:8080",
    # 必要に応じて他のオリジンも追加
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins, # すべてのオリジンを許可する場合は ["*"]
    allow_credentials=True,
    allow_methods=["*"], # すべてのメソッド (GET, POST, etc.) を許可
    allow_headers=["*"], # すべてのヘッダーを許可
)

if __name__ == "__main__":
    # このファイルが直接実行された場合（Docker外でのデバッグ用）
    import uvicorn
    port = int(os.getenv("BACKEND_PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port) 