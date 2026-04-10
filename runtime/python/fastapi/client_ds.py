# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
# Licensed under the Apache License, Version 2.0 (the "License");
# ... (保留原有版权声明)

import argparse
import logging
import requests
import torch
import torchaudio
import numpy as np

def main():
    url = f"http://{args.host}:{args.port}/inference_{args.mode}"
    
    try:
        # ==================== 准备请求数据 ====================
        if args.mode == 'sft':
            data = {'tts_text': args.tts_text, 'spk_id': args.spk_id}
            response = requests.post(url, data=data, stream=True, timeout=30)
            
        elif args.mode == 'zero_shot':
            data = {'tts_text': args.tts_text, 'prompt_text': args.prompt_text}
            files = {'prompt_wav': open(args.prompt_wav, 'rb')}
            response = requests.post(url, data=data, files=files, stream=True, timeout=30)
            files['prompt_wav'].close()  # ✅ 正确关闭文件
            
        elif args.mode == 'cross_lingual':
            data = {'tts_text': args.tts_text}
            files = {'prompt_wav': open(args.prompt_wav, 'rb')}
            response = requests.post(url, data=data, files=files, stream=True, timeout=30)
            files['prompt_wav'].close()  # ✅ 正确关闭文件
            
        else:  # instruct
            data = {
                'tts_text': args.tts_text,
                'spk_id': args.spk_id,
                'instruct_text': args.instruct_text
            }
            response = requests.post(url, data=data, stream=True, timeout=30)
        
        # ==================== 检查响应 ====================
        if response.status_code != 200:
            print(f"❌ 服务端错误 {response.status_code}: {response.text[:200]}")
            return
        
        # ==================== 接收音频流 ====================
        print("⏳ 接收音频流...")
        tts_audio = b''
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                tts_audio += chunk
        
        if not tts_audio:
            print("❌ 未收到音频数据")
            return
        
        # ==================== 保存音频 ====================
        audio_array = np.frombuffer(tts_audio, dtype=np.int16)
        tts_speech = torch.from_numpy(audio_array).unsqueeze(0).float() / 32767.0
        
        torchaudio.save(args.tts_wav, tts_speech, args.target_sr)
        print(f"✅ 成功！音频已保存至: {args.tts_wav}")
        
    except Exception as e:
        print(f"❌ 客户端错误: {e}")
        if 'response' in locals():
            print(f"响应状态: {response.status_code}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', type=str, default='10.255.1.115')
    parser.add_argument('--port', type=int, default=50000)
    parser.add_argument('--mode', default='zero_shot', choices=['sft', 'zero_shot', 'cross_lingual', 'instruct'])
    parser.add_argument('--tts_text', type=str, default='生活就像海洋，只有意志坚强的人才能到达彼岸。')
    parser.add_argument('--prompt_text', type=str, default='想了解一个专业，您必须要知道专业基本情况、课程开设情况、人才培养模式，还有专业建设和发展。')
    parser.add_argument('--prompt_wav', type=str, default='E:/cosyVoice/prompt_audio/wudi.mp3')
    parser.add_argument('--instruct_text', type=str, default='用温柔的语气说话')
    parser.add_argument('--tts_wav', type=str, default='wudi.wav')
    parser.add_argument('--target_sr', type=int, default=22050)
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    main()