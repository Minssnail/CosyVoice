# -*- coding: utf-8 -*-
# CosyVoice 语音克隆 API 服务
# 局域网内通过 HTTP API 调用语音合成/克隆能力

import os
import sys
import io
import argparse
import time
import hashlib
import logging
import wave
import struct
import tempfile
from typing import Optional

import numpy as np
import torch
import torchaudio

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'third_party', 'Matcha-TTS'))

from fastapi import FastAPI, UploadFile, Form, File, HTTPException, Query
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from cosyvoice.cli.cosyvoice import AutoModel
from cosyvoice.utils.file_utils import load_wav
from cosyvoice.utils.common import set_all_random_seed

logging.getLogger('matplotlib').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


# ─── 工具函数 ───────────────────────────────────────────────

def pcm_to_wav_bytes(pcm_data: np.ndarray, sample_rate: int, sample_width: int = 2) -> bytes:
    """将 PCM numpy 数组转为完整的 WAV 文件字节"""
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        # float -> int16
        if pcm_data.dtype in (np.float32, np.float64):
            pcm_data = (pcm_data * 32767).clip(-32768, 32767).astype(np.int16)
        wf.writeframes(pcm_data.tobytes())
    return buf.getvalue()


def save_upload_to_temp(upload_file: UploadFile) -> str:
    """将上传文件保存到临时文件, 返回路径"""
    suffix = os.path.splitext(upload_file.filename or '.wav')[1]
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    content = upload_file.file.read()
    tmp.write(content)
    tmp.flush()
    tmp.close()
    return tmp.name


# ─── 构建 App ───────────────────────────────────────────────

app = FastAPI(
    title="CosyVoice 语音克隆 API",
    description="基于 CosyVoice 的语音合成与克隆服务，支持零样本克隆、跨语种克隆、自然语言控制等",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局变量，在 startup 中初始化
cosyvoice = None
sample_rate = None
model_version = None  # 'v1', 'v2', 'v3'


# ─── 健康检查 ───────────────────────────────────────────────

@app.get("/api/health")
async def health_check():
    return {
        "status": "ok",
        "model_dir": app.state.model_dir,
        "model_version": model_version,
        "sample_rate": sample_rate,
        "device": str(next(iter([
            'cuda' if torch.cuda.is_available() else 'cpu'
        ]))),
        "cuda_available": torch.cuda.is_available(),
    }


# ─── 音色管理 ───────────────────────────────────────────────

@app.get("/api/speakers")
async def list_speakers():
    """列出所有可用音色（预训练 + 已注册的零样本音色）"""
    spks = cosyvoice.list_available_spks()
    return {"speakers": spks, "count": len(spks)}


@app.post("/api/speakers/register")
async def register_speaker(
    speaker_id: str = Form(..., description="自定义音色 ID"),
    prompt_text: str = Form(..., description="参考音频对应的文本"),
    prompt_wav: UploadFile = File(..., description="参考音频文件 (≥16kHz, ≤30s)"),
):
    """
    注册零样本音色：上传一段参考音频和对应文本，注册为可复用的音色 ID。
    后续调用 TTS 接口时可直接通过 speaker_id 使用，无需重复上传。
    """
    if not speaker_id.strip():
        raise HTTPException(400, "speaker_id 不能为空")

    tmp_path = save_upload_to_temp(prompt_wav)
    try:
        success = cosyvoice.add_zero_shot_spk(prompt_text, tmp_path, speaker_id)
        if not success:
            raise HTTPException(500, "注册音色失败")
        # 持久化
        cosyvoice.save_spkinfo()
        return {"message": f"音色 '{speaker_id}' 注册成功", "speaker_id": speaker_id}
    finally:
        os.unlink(tmp_path)


@app.delete("/api/speakers/{speaker_id}")
async def delete_speaker(speaker_id: str):
    """删除已注册的零样本音色"""
    if speaker_id not in cosyvoice.frontend.spk2info:
        raise HTTPException(404, f"音色 '{speaker_id}' 不存在")
    del cosyvoice.frontend.spk2info[speaker_id]
    cosyvoice.save_spkinfo()
    return {"message": f"音色 '{speaker_id}' 已删除"}


# ─── TTS 接口 ───────────────────────────────────────────────

@app.post("/api/tts/sft")
async def tts_sft(
    tts_text: str = Form(..., description="要合成的文本"),
    speaker_id: str = Form(..., description="预训练音色 ID"),
    speed: float = Form(1.0, description="语速 (0.5-2.0)"),
    seed: int = Form(0, description="随机种子, 0=不固定"),
):
    """使用预训练音色合成语音"""
    if seed > 0:
        set_all_random_seed(seed)

    all_pcm = []
    for chunk in cosyvoice.inference_sft(tts_text, speaker_id, stream=False, speed=speed):
        all_pcm.append(chunk['tts_speech'].numpy().flatten())

    if not all_pcm:
        raise HTTPException(500, "合成失败，没有生成音频")

    audio = np.concatenate(all_pcm)
    wav_bytes = pcm_to_wav_bytes(audio, sample_rate)
    return StreamingResponse(
        io.BytesIO(wav_bytes),
        media_type="audio/wav",
        headers={"Content-Disposition": "attachment; filename=tts_sft.wav"}
    )


@app.post("/api/tts/zero_shot")
async def tts_zero_shot(
    tts_text: str = Form(..., description="要合成的文本"),
    prompt_text: str = Form("", description="参考音频对应文本 (使用已注册音色时可留空)"),
    prompt_wav: Optional[UploadFile] = File(None, description="参考音频文件 (使用已注册音色时可不传)"),
    speaker_id: str = Form("", description="已注册的零样本音色 ID (与 prompt_wav 二选一)"),
    speed: float = Form(1.0, description="语速"),
    seed: int = Form(0, description="随机种子"),
):
    """
    零样本语音克隆 (3s 极速复刻)

    两种用法：
    1. 传 prompt_wav + prompt_text → 实时克隆
    2. 传 speaker_id → 使用之前注册的音色
    """
    if seed > 0:
        set_all_random_seed(seed)

    tmp_path = None
    try:
        if prompt_wav is not None:
            # 实时克隆模式
            tmp_path = save_upload_to_temp(prompt_wav)
            all_pcm = []
            for chunk in cosyvoice.inference_zero_shot(
                tts_text, prompt_text, tmp_path,
                stream=False, speed=speed
            ):
                all_pcm.append(chunk['tts_speech'].numpy().flatten())
        elif speaker_id:
            # 已注册音色模式
            all_pcm = []
            for chunk in cosyvoice.inference_zero_shot(
                tts_text, '', '', zero_shot_spk_id=speaker_id,
                stream=False, speed=speed
            ):
                all_pcm.append(chunk['tts_speech'].numpy().flatten())
        else:
            raise HTTPException(400, "必须提供 prompt_wav 或 speaker_id")

        if not all_pcm:
            raise HTTPException(500, "合成失败")

        audio = np.concatenate(all_pcm)
        wav_bytes = pcm_to_wav_bytes(audio, sample_rate)
        return StreamingResponse(
            io.BytesIO(wav_bytes),
            media_type="audio/wav",
            headers={"Content-Disposition": "attachment; filename=tts_zero_shot.wav"}
        )
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.post("/api/tts/cross_lingual")
async def tts_cross_lingual(
    tts_text: str = Form(..., description="要合成的文本 (与参考音频不同语言)"),
    prompt_wav: UploadFile = File(..., description="参考音频文件"),
    speed: float = Form(1.0, description="语速"),
    seed: int = Form(0, description="随机种子"),
):
    """
    跨语种语音克隆

    用 A 语言的参考音频克隆音色，合成 B 语言的语音。
    """
    if seed > 0:
        set_all_random_seed(seed)

    tmp_path = save_upload_to_temp(prompt_wav)
    try:
        all_pcm = []
        for chunk in cosyvoice.inference_cross_lingual(
            tts_text, tmp_path, stream=False, speed=speed
        ):
            all_pcm.append(chunk['tts_speech'].numpy().flatten())

        if not all_pcm:
            raise HTTPException(500, "合成失败")

        audio = np.concatenate(all_pcm)
        wav_bytes = pcm_to_wav_bytes(audio, sample_rate)
        return StreamingResponse(
            io.BytesIO(wav_bytes),
            media_type="audio/wav",
            headers={"Content-Disposition": "attachment; filename=tts_cross_lingual.wav"}
        )
    finally:
        os.unlink(tmp_path)


@app.post("/api/tts/instruct")
async def tts_instruct(
    tts_text: str = Form(..., description="要合成的文本"),
    instruct_text: str = Form(..., description="指令文本 (如: 用四川话说/用开心的语气说)"),
    prompt_wav: Optional[UploadFile] = File(None, description="参考音频 (v2/v3 instruct2 模式需要)"),
    speaker_id: str = Form("", description="预训练音色 ID (v1 instruct 模式需要)"),
    speed: float = Form(1.0, description="语速"),
    seed: int = Form(0, description="随机种子"),
):
    """
    自然语言控制合成

    - CosyVoice v1: 需要 speaker_id + instruct_text
    - CosyVoice v2/v3: 需要 prompt_wav + instruct_text (使用 instruct2)
    """
    if seed > 0:
        set_all_random_seed(seed)

    tmp_path = None
    try:
        all_pcm = []

        if model_version == 'v1':
            # v1 instruct 模式
            if not speaker_id:
                raise HTTPException(400, "v1 模型的 instruct 模式需要 speaker_id")
            for chunk in cosyvoice.inference_instruct(
                tts_text, speaker_id, instruct_text,
                stream=False, speed=speed
            ):
                all_pcm.append(chunk['tts_speech'].numpy().flatten())
        else:
            # v2/v3 instruct2 模式
            if prompt_wav is None:
                raise HTTPException(400, "v2/v3 模型的 instruct 模式 need prompt_wav")
            tmp_path = save_upload_to_temp(prompt_wav)
            for chunk in cosyvoice.inference_instruct2(
                tts_text, instruct_text, tmp_path,
                stream=False, speed=speed
            ):
                all_pcm.append(chunk['tts_speech'].numpy().flatten())

        if not all_pcm:
            raise HTTPException(500, "合成失败")

        audio = np.concatenate(all_pcm)
        wav_bytes = pcm_to_wav_bytes(audio, sample_rate)
        return StreamingResponse(
            io.BytesIO(wav_bytes),
            media_type="audio/wav",
            headers={"Content-Disposition": "attachment; filename=tts_instruct.wav"}
        )
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.post("/api/tts/stream/zero_shot")
async def tts_stream_zero_shot(
    tts_text: str = Form(..., description="要合成的文本"),
    prompt_text: str = Form("", description="参考音频对应文本"),
    prompt_wav: Optional[UploadFile] = File(None, description="参考音频文件"),
    speaker_id: str = Form("", description="已注册的音色 ID"),
    seed: int = Form(0, description="随机种子"),
):
    """
    流式零样本语音克隆 — 边生成边返回 PCM 流

    返回 raw PCM int16 数据流，客户端需自行拼接。
    """
    if seed > 0:
        set_all_random_seed(seed)

    tmp_path = None

    def generate():
        nonlocal tmp_path
        try:
            if prompt_wav is not None:
                tmp_path = save_upload_to_temp(prompt_wav)
                gen = cosyvoice.inference_zero_shot(
                    tts_text, prompt_text, tmp_path, stream=True
                )
            elif speaker_id:
                gen = cosyvoice.inference_zero_shot(
                    tts_text, '', '', zero_shot_spk_id=speaker_id, stream=True
                )
            else:
                return

            for chunk in gen:
                pcm = chunk['tts_speech'].numpy().flatten()
                pcm_int16 = (pcm * 32767).clip(-32768, 32767).astype(np.int16)
                yield pcm_int16.tobytes()
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

    return StreamingResponse(
        generate(),
        media_type="audio/pcm",
        headers={
            "X-Sample-Rate": str(sample_rate),
            "X-Sample-Width": "2",
            "X-Channels": "1",
        }
    )


# ─── 启动入口 ───────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CosyVoice 语音克隆 API 服务')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                        help='监听地址 (默认 0.0.0.0, 局域网可访问)')
    parser.add_argument('--port', type=int, default=9880,
                        help='监听端口')
    parser.add_argument('--model_dir', type=str,
                        default='pretrained_models/Fun-CosyVoice3-0.5B',
                        help='模型目录路径或 ModelScope repo id')
    args = parser.parse_args()

    logger.info(f'正在加载模型: {args.model_dir}')
    cosyvoice = AutoModel(model_dir=args.model_dir)
    sample_rate = cosyvoice.sample_rate

    # 判断模型版本
    from cosyvoice.cli.cosyvoice import CosyVoice as _V1, CosyVoice2 as _V2, CosyVoice3 as _V3
    if isinstance(cosyvoice, _V3):
        model_version = 'v3'
    elif isinstance(cosyvoice, _V2):
        model_version = 'v2'
    else:
        model_version = 'v1'

    app.state.model_dir = args.model_dir
    logger.info(f'模型加载完成: version={model_version}, sample_rate={sample_rate}')
    logger.info(f'可用音色: {cosyvoice.list_available_spks()}')
    logger.info(f'API 文档: http://{args.host}:{args.port}/docs')

    uvicorn.run(app, host=args.host, port=args.port, log_level='info')
