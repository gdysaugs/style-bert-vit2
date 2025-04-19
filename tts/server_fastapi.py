"""
API server for TTS
TODO: server_editor.pyと統合する?
"""

import argparse
import os
import sys
from io import BytesIO
from pathlib import Path
from typing import Any, Optional
from urllib.parse import unquote

import GPUtil
import psutil
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from scipy.io import wavfile
from pydantic import BaseModel, Field

from config import get_config
from style_bert_vits2.constants import (
    DEFAULT_ASSIST_TEXT_WEIGHT,
    DEFAULT_LENGTH,
    DEFAULT_LINE_SPLIT,
    DEFAULT_NOISE,
    DEFAULT_NOISEW,
    DEFAULT_SDP_RATIO,
    DEFAULT_SPLIT_INTERVAL,
    DEFAULT_STYLE,
    DEFAULT_STYLE_WEIGHT,
    Languages,
)
from style_bert_vits2.logging import logger
from style_bert_vits2.nlp import bert_models
from style_bert_vits2.nlp.japanese import pyopenjtalk_worker as pyopenjtalk
from style_bert_vits2.nlp.japanese.user_dict import update_dict
from style_bert_vits2.tts_model import TTSModel, TTSModelHolder


# 追加: PyTorch CUDA 可用性チェック
logger.info(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
if torch.cuda.is_available():
    logger.info(f"torch.cuda.device_count(): {torch.cuda.device_count()}")
    logger.info(f"torch.cuda.current_device(): {torch.cuda.current_device()}")
    logger.info(f"torch.cuda.get_device_name(0): {torch.cuda.get_device_name(0)}")
else:
    logger.warning("CUDA not available, PyTorch will use CPU.")

config = get_config()
ln = config.server_config.language


# pyopenjtalk_worker を起動
## pyopenjtalk_worker は TCP ソケットサーバーのため、ここで起動する
pyopenjtalk.initialize_worker()

# dict_data/ 以下の辞書データを pyopenjtalk に適用
# update_dict() # 起動時の辞書更新を一時的にコメントアウト (タイムアウトエラー回避)

# 事前に BERT モデル/トークナイザーをロードしておく
## ここでロードしなくても必要になった際に自動ロードされるが、時間がかかるため事前にロードしておいた方が体験が良い
bert_models.load_model(Languages.JP)
bert_models.load_tokenizer(Languages.JP)
# bert_models.load_model(Languages.EN) # 日本語のみ利用するためコメントアウト
# bert_models.load_tokenizer(Languages.EN) # 日本語のみ利用するためコメントアウト
# bert_models.load_model(Languages.ZH) # 日本語のみ利用するためコメントアウト
# bert_models.load_tokenizer(Languages.ZH) # 日本語のみ利用するためコメントアウト


def raise_validation_error(msg: str, param: str):
    logger.warning(f"Validation error: {msg}")
    raise HTTPException(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        detail=[dict(type="invalid_params", msg=msg, loc=["query", param])],
    )


class AudioResponse(Response):
    media_type = "audio/wav"


loaded_models: list[TTSModel] = []


def load_models(model_holder: TTSModelHolder):
    global loaded_models
    loaded_models = []
    for model_name, model_paths in model_holder.model_files_dict.items():
        # 追加: モデル初期化時のデバイスをログ出力
        logger.info(f"Initializing TTSModel '{model_name}' with device: {model_holder.device}")
        model = TTSModel(
            model_path=model_paths[0],
            config_path=model_holder.root_dir / model_name / "config.json",
            style_vec_path=model_holder.root_dir / model_name / "style_vectors.npy",
            device=model_holder.device,
        )
        # 起動時に全てのモデルを読み込むのは時間がかかりメモリを食うのでやめる
        # model.load()
        loaded_models.append(model)


# 追加: /voice エンドポイントのペイロード定義
class VoicePayload(BaseModel):
    text: str = Field(..., min_length=1, description="セリフ") # max_length は後で適用
    # encoding: Optional[str] = Field(None, description="textをURLデコードする(ex, `utf-8`) ※基本不要になるはず")
    model_name: Optional[str] = Field(None, description="モデル名(model_idより優先)。model_assets内のディレクトリ名を指定")
    model_id: int = Field(0, description="モデルID。`GET /models/info`のkeyの値を指定ください")
    speaker_name: Optional[str] = Field(None, description="話者名(speaker_idより優先)。esd.listの2列目の文字列を指定")
    speaker_id: int = Field(0, description="話者ID。model_assets>[model]>config.json内のspk2idを確認")
    sdp_ratio: float = Field(DEFAULT_SDP_RATIO, ge=0.0, description="SDP(Stochastic Duration Predictor)/DP混合比。比率が高くなるほどトーンのばらつきが大きくなる")
    noise: float = Field(DEFAULT_NOISE, ge=0.0, description="サンプルノイズの割合。大きくするほどランダム性が高まる")
    noisew: float = Field(DEFAULT_NOISEW, ge=0.0, description="SDPノイズ。大きくするほど発音の間隔にばらつきが出やすくなる")
    length: float = Field(DEFAULT_LENGTH, ge=0.1, description="話速。基準は1で大きくするほど音声は長くなり読み上げが遅まる")
    language: Languages = Field(ln, description="textの言語")
    auto_split: bool = Field(DEFAULT_LINE_SPLIT, description="改行で分けて生成")
    split_interval: float = Field(DEFAULT_SPLIT_INTERVAL, ge=0.0, description="分けた場合に挟む無音の長さ（秒）")
    assist_text: Optional[str] = Field(None, description="このテキストの読み上げと似た声音・感情になりやすくなる。ただし抑揚やテンポ等が犠牲になる傾向がある")
    assist_text_weight: float = Field(DEFAULT_ASSIST_TEXT_WEIGHT, ge=0.0, le=1.0, description="assist_textの強さ")
    style: Optional[str] = Field(DEFAULT_STYLE, description="スタイル")
    style_weight: float = Field(DEFAULT_STYLE_WEIGHT, ge=0.0, description="スタイルの強さ")
    reference_audio_path: Optional[str] = Field(None, description="スタイルを音声ファイルで行う ※サーバーローカルパス指定のため通常API経由では非推奨")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", action="store_true", help="Use CPU instead of GPU")
    parser.add_argument(
        "--dir", "-d", type=str, help="Model directory", default=config.assets_root
    )
    args = parser.parse_args()

    if args.cpu:
        device = "cpu"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    # 追加: 判定されたデバイスをログ出力
    logger.info(f"Determined device: {device}")

    model_dir = Path(args.dir)
    # 追加: TTSModelHolder 初期化時のデバイスをログ出力
    logger.info(f"Initializing TTSModelHolder with root_dir: {model_dir}, device: {device}")
    model_holder = TTSModelHolder(model_dir, device)
    if len(model_holder.model_names) == 0:
        logger.error(f"Models not found in {model_dir}.")
        sys.exit(1)

    logger.info("Loading models...")
    load_models(model_holder)

    if loaded_models:
        try:
            default_model_index = 0 # 最初のモデルをプリロード対象とする
            default_model = loaded_models[default_model_index]
            # 修正: .config.name を参照しないようにログメッセージを変更
            logger.warning(f"Preloading default model (ID: {default_model_index}) to VRAM. This may take time and consume VRAM at startup.")
            default_model.load() # モデルをVRAMにロード！
            # 修正: .config.name を参照しないようにログメッセージを変更
            logger.info(f"Successfully preloaded default model (ID: {default_model_index}).")
        except Exception as e:
            logger.error(f"Failed to preload default model: {e}", exc_info=True)
            # プリロードに失敗してもサーバー起動は続行する (エラーログは残す)
    else:
        logger.warning("No models found to preload.")

    limit = config.server_config.limit
    if limit < 1:
        limit = None
    else:
        logger.info(
            f"The maximum length of the text is {limit}. If you want to change it, modify config.yml. Set limit to -1 to remove the limit."
        )
    app = FastAPI()
    allow_origins = config.server_config.origins
    if allow_origins:
        logger.warning(
            f"CORS allow_origins={config.server_config.origins}. If you don't want, modify config.yml"
        )
        app.add_middleware(
            CORSMiddleware,
            allow_origins=config.server_config.origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    # app.logger = logger
    # ↑効いていなさそう。loggerをどうやって上書きするかはよく分からなかった。

    @app.post("/voice", response_class=AudioResponse)
    async def voice(
        request: Request,
        payload: VoicePayload # リクエストボディとして受け取る
    ):
        """Infer text to speech(テキストから感情付き音声を生成する)"""
        # 変更: ログ出力をリクエストボディの内容にする (ただし長すぎる可能性に注意)
        # logger.info(f"{request.client.host}:{request.client.port} /voice payload: {payload.dict()}")
        # シンプルにテキストだけログ出力する例
        logger.info(f"{request.client.host}:{request.client.port} /voice requested for text: '{payload.text[:50]}...'" if len(payload.text) > 50 else f"{request.client.host}:{request.client.port} /voice requested for text: '{payload.text}'")

        # 削除: GETメソッドに関する警告
        # if request.method == "GET":
        #     logger.warning(
        #         "The GET method is not recommended for this endpoint due to various restrictions. Please use the POST method."
        #     )

        # 追加: text の長さを limit でチェック
        if limit is not None and len(payload.text) > limit:
            raise_validation_error(f"Text length exceeds the limit ({limit})", "text")

        # 変更: パラメータを payload オブジェクトから取得
        if payload.model_id >= len(loaded_models):
            raise_validation_error(
                f"Invalid model_id: {payload.model_id}. Available models: {list(range(len(loaded_models)))}",
                "model_id",
            )
        if payload.model_name:
            try:
                selected_model = next(
                    m for m in loaded_models if m.config.name == payload.model_name
                )
            except StopIteration:
                raise_validation_error(
                    f"Invalid model_name: {payload.model_name}. Available models: {[m.config.name for m in loaded_models]}",
                    "model_name",
                )
        else:
            selected_model = loaded_models[payload.model_id]

        # 修正: .config.name を参照しないようにログメッセージを変更 (デバイス情報のみ表示)
        logger.info(f"Performing inference with selected model (ID: {payload.model_id if not payload.model_name else payload.model_name}) on device: {selected_model.device}")

        # 話者ID/名前の解決と検証
        selected_speaker_id = payload.speaker_id # まずは payload の speaker_id を使う

        if payload.speaker_name:
            # speaker_name が指定されている場合
            if hasattr(selected_model, 'spk2id') and payload.speaker_name in selected_model.spk2id:
                # spk2id 辞書に speaker_name があれば、対応する ID を取得
                selected_speaker_id = selected_model.spk2id[payload.speaker_name]
                logger.info(f"Resolved speaker_name '{payload.speaker_name}' to speaker_id {selected_speaker_id}")
            else:
                # spk2id がないか、speaker_name が見つからない場合
                logger.warning(f"Speaker name '{payload.speaker_name}' not found in model '{selected_model.config.name}'. Checking fallback speaker_id {payload.speaker_id}.")
                # fallback として payload.speaker_id の有効性をチェック
                if not hasattr(selected_model, 'id2spk') or payload.speaker_id not in selected_model.id2spk:
                     raise_validation_error(
                         f"Invalid fallback speaker_id: {payload.speaker_id}. Available IDs: {list(selected_model.id2spk.keys()) if hasattr(selected_model, 'id2spk') else 'Unknown'}",
                         "speaker_id"
                     )
                # selected_speaker_id は payload.speaker_id のまま
        else:
            # speaker_name が指定されていない場合、payload.speaker_id の有効性をチェック
            if not hasattr(selected_model, 'id2spk') or payload.speaker_id not in selected_model.id2spk:
                 raise_validation_error(
                     f"Invalid speaker_id: {payload.speaker_id}. Available IDs for model '{selected_model.config.name}': {list(selected_model.id2spk.keys()) if hasattr(selected_model, 'id2spk') else 'Unknown'}",
                     "speaker_id"
                 )
            # selected_speaker_id は payload.speaker_id のまま

        # reference_audio_path はセキュリティリスクと使いにくさから非推奨とするコメントを追加
        if payload.reference_audio_path:
            logger.warning("Using reference_audio_path via API is generally not recommended due to security and path resolution issues.")
            # パスの検証やアクセス制御が必要になる
            ref_path = Path(payload.reference_audio_path)
            if not ref_path.is_file():
                 raise_validation_error(f"Reference audio file not found: {payload.reference_audio_path}", "reference_audio_path")

        try:
            sampling_rate, audio = selected_model.infer(
                text=payload.text,
                sdp_ratio=payload.sdp_ratio,
                noise=payload.noise,
                length=payload.length,
                language=payload.language,
                speaker_id=selected_speaker_id, # speaker_id を使用
                split_interval=payload.split_interval,
                assist_text=payload.assist_text,
                assist_text_weight=payload.assist_text_weight,
                style=payload.style,
                style_weight=payload.style_weight,
                reference_audio_path=payload.reference_audio_path,
            )
        except Exception as e:
            logger.exception("Error during inference")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Inference error: {e}"
            )

        with BytesIO() as bio:
            wavfile.write(bio, sampling_rate, audio)
            return AudioResponse(bio.getvalue())

    @app.post("/g2p")
    def g2p(text: str):
        return g2kata_tone(normalize_text(text))

    @app.get("/models/info")
    def get_loaded_models_info():
        """ロードされたモデル情報の取得"""

        result: dict[str, dict[str, Any]] = dict()
        for model_id, model in enumerate(loaded_models):
            result[str(model_id)] = {
                "config_path": model.config_path,
                "model_path": model.model_path,
                "device": model.device,
                "spk2id": model.spk2id,
                "id2spk": model.id2spk,
                "style2id": model.style2id,
            }
        return result

    @app.post("/models/refresh")
    def refresh():
        """モデルをパスに追加/削除した際などに読み込ませる"""
        model_holder.refresh()
        load_models(model_holder)
        return get_loaded_models_info()

    @app.get("/status")
    def get_status():
        """実行環境のステータスを取得"""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory_info = psutil.virtual_memory()
        memory_total = memory_info.total
        memory_available = memory_info.available
        memory_used = memory_info.used
        memory_percent = memory_info.percent
        gpuInfo = []
        devices = ["cpu"]
        for i in range(torch.cuda.device_count()):
            devices.append(f"cuda:{i}")
        gpus = GPUtil.getGPUs()
        for gpu in gpus:
            gpuInfo.append(
                {
                    "gpu_id": gpu.id,
                    "gpu_load": gpu.load,
                    "gpu_memory": {
                        "total": gpu.memoryTotal,
                        "used": gpu.memoryUsed,
                        "free": gpu.memoryFree,
                    },
                }
            )
        return {
            "devices": devices,
            "cpu_percent": cpu_percent,
            "memory_total": memory_total,
            "memory_available": memory_available,
            "memory_used": memory_used,
            "memory_percent": memory_percent,
            "gpu": gpuInfo,
        }

    # 変更: 実際にUvicornがリッスンするホストとポートをログに出す
    listen_host = "0.0.0.0"
    listen_port = config.server_config.port
    logger.info(f"Server starting, configured to listen on: http://{listen_host}:{listen_port}")
    logger.info(f"API documentation available at http://{listen_host}:{listen_port}/docs")
    logger.info(
        f"Input text length limit: {limit}. You can change it in server.limit in config.yml"
    )
    uvicorn.run(
        app, port=listen_port, host=listen_host, log_level="info" # 変更: log_level を "info" に
    )
