# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
# Licensed under the Apache License, Version 2.0 (the "License");
# ... (保留原有版权声明)

import os
import sys
import argparse
import logging
import tempfile
import shutil
logging.getLogger('matplotlib').setLevel(logging.WARNING)
from fastapi import FastAPI, UploadFile, Form, File, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append('{}/../../..'.format(ROOT_DIR))
sys.path.append('{}/../../../third_party/Matcha-TTS'.format(ROOT_DIR))

from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

def generate_data(model_output):
    """流式生成音频数据"""
    try:
        for chunk in model_output:
            audio_data = chunk['tts_speech'].numpy()
            audio_int16 = (audio_data * 32767).astype(np.int16)
            yield audio_int16.tobytes()
    except Exception as e:
        logging.error(f"生成数据时出错: {e}")
        raise

# ==================== 接口定义 ====================

@app.post("/inference_sft")
async def inference_sft(
    tts_text: str = Form(...), 
    spk_id: str = Form(...)
):
    try:
        model_output = cosyvoice.inference_sft(tts_text, spk_id)
        return StreamingResponse(generate_data(model_output), media_type="application/octet-stream")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/inference_zero_shot")
async def inference_zero_shot(
    background_tasks: BackgroundTasks,
    tts_text: str = Form(...), 
    prompt_text: str = Form(...), 
    prompt_wav: UploadFile = File(...)
):
    temp_path = None
    try:
        audio_content = await prompt_wav.read()
        if not audio_content:
            raise HTTPException(status_code=400, detail="未收到音频文件内容")
        
        # 创建临时文件
        temp_fd, temp_path = tempfile.mkstemp(suffix=".wav")
        os.close(temp_fd)
        
        with open(temp_path, "wb") as f:
            f.write(audio_content)
        
        logging.info(f"✅ 临时文件创建: {temp_path} ({len(audio_content)} bytes)")
        
        # 注册后台清理任务
        background_tasks.add_task(
            lambda p: os.path.exists(p) and os.unlink(p), 
            temp_path
        )
        
        # 调用模型
        model_output = cosyvoice.inference_zero_shot(tts_text, prompt_text, temp_path)
        return StreamingResponse(generate_data(model_output), media_type="application/octet-stream")
        
    except Exception as e:
        logging.error(f"❌ zero_shot推理失败: {e}")
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except:
                pass
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/inference_cross_lingual")
async def inference_cross_lingual(
    background_tasks: BackgroundTasks,
    tts_text: str = Form(...), 
    prompt_wav: UploadFile = File(...)
):
    temp_path = None
    try:
        audio_content = await prompt_wav.read()
        if not audio_content:
            raise HTTPException(status_code=400, detail="未收到音频文件内容")
        
        temp_fd, temp_path = tempfile.mkstemp(suffix=".wav")
        os.close(temp_fd)
        
        with open(temp_path, "wb") as f:
            f.write(audio_content)
        
        logging.info(f"✅ 临时文件创建: {temp_path}")
        background_tasks.add_task(lambda p: os.path.exists(p) and os.unlink(p), temp_path)
        
        model_output = cosyvoice.inference_cross_lingual(tts_text, temp_path)
        return StreamingResponse(generate_data(model_output), media_type="application/octet-stream")
        
    except Exception as e:
        logging.error(f"❌ cross_lingual推理失败: {e}")
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except:
                pass
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/inference_instruct")
async def inference_instruct(
    tts_text: str = Form(...), 
    spk_id: str = Form(...), 
    instruct_text: str = Form(...)
):
    try:
        model_output = cosyvoice.inference_instruct(tts_text, spk_id, instruct_text)
        return StreamingResponse(generate_data(model_output), media_type="application/octet-stream")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/inference_instruct2")
async def inference_instruct2(
    background_tasks: BackgroundTasks,
    tts_text: str = Form(...), 
    instruct_text: str = Form(...), 
    prompt_wav: UploadFile = File(...)
):
    temp_path = None
    try:
        audio_content = await prompt_wav.read()
        if not audio_content:
            raise HTTPException(status_code=400, detail="未收到音频文件内容")
        
        temp_fd, temp_path = tempfile.mkstemp(suffix=".wav")
        os.close(temp_fd)
        
        with open(temp_path, "wb") as f:
            f.write(audio_content)
        
        logging.info(f"✅ 临时文件创建: {temp_path}")
        background_tasks.add_task(lambda p: os.path.exists(p) and os.unlink(p), temp_path)
        
        model_output = cosyvoice.inference_instruct2(tts_text, instruct_text, temp_path)
        return StreamingResponse(generate_data(model_output), media_type="application/octet-stream")
        
    except Exception as e:
        logging.error(f"❌ instruct2推理失败: {e}")
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except:
                pass
        raise HTTPException(status_code=500, detail=str(e))

# ==================== 主入口 ====================

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=50000)
    parser.add_argument('--model_dir', type=str, default='iic/CosyVoice-300M')
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    # 加载模型
    try:
        cosyvoice = CosyVoice(args.model_dir)
        model_type = "CosyVoice"
    except Exception:
        try:
            cosyvoice = CosyVoice2(args.model_dir)
            model_type = "CosyVoice2"
        except Exception:
            logging.error("无法加载模型，请检查 model_dir")
            raise TypeError('no valid model_type!')
    
    logging.info(f"🚀 成功加载 {model_type} 模型: {args.model_dir}")
    uvicorn.run(app, host="10.255.1.115", port=args.port)