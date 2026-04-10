# -*- coding: utf-8 -*-
"""
CosyVoice API 客户端 SDK

用法示例:
    from api_client import CosyVoiceClient

    client = CosyVoiceClient("http://10.255.1.115:9880")

    # 零样本克隆
    client.zero_shot("你好世界", "参考文本内容", "ref.wav", output="out.wav")

    # 使用已注册音色
    client.register_speaker("my_voice", "参考文本", "ref.wav")
    client.zero_shot("你好世界", speaker_id="my_voice", output="out.wav")

    # 跨语种
    client.cross_lingual("<|en|>Hello world", "ref_cn.wav", output="out.wav")

    # 指令控制
    client.instruct("今天天气真好", "用四川话说", "ref.wav", output="out.wav")
"""

import os
import sys
import argparse
import requests


class CosyVoiceClient:
    def __init__(self, base_url: str = "http://10.255.1.115:9880"):
        self.base_url = base_url.rstrip("/")

    def health(self) -> dict:
        r = requests.get(f"{self.base_url}/api/health")
        r.raise_for_status()
        return r.json()

    def list_speakers(self) -> list:
        r = requests.get(f"{self.base_url}/api/speakers")
        r.raise_for_status()
        return r.json()["speakers"]

    def register_speaker(self, speaker_id: str, prompt_text: str, prompt_wav_path: str) -> dict:
        with open(prompt_wav_path, "rb") as f:
            r = requests.post(
                f"{self.base_url}/api/speakers/register",
                data={"speaker_id": speaker_id, "prompt_text": prompt_text},
                files={"prompt_wav": (os.path.basename(prompt_wav_path), f, "audio/wav")},
            )
        r.raise_for_status()
        return r.json()

    def delete_speaker(self, speaker_id: str) -> dict:
        r = requests.delete(f"{self.base_url}/api/speakers/{speaker_id}")
        r.raise_for_status()
        return r.json()

    def sft(self, tts_text: str, speaker_id: str, output: str = "output.wav",
            speed: float = 1.0, seed: int = 0) -> str:
        r = requests.post(
            f"{self.base_url}/api/tts/sft",
            data={"tts_text": tts_text, "speaker_id": speaker_id,
                  "speed": speed, "seed": seed},
        )
        r.raise_for_status()
        with open(output, "wb") as f:
            f.write(r.content)
        return output

    def zero_shot(self, tts_text: str, prompt_text: str = "",
                  prompt_wav_path: str = None, speaker_id: str = "",
                  output: str = "output.wav", speed: float = 1.0,
                  seed: int = 0) -> str:
        data = {
            "tts_text": tts_text,
            "prompt_text": prompt_text,
            "speaker_id": speaker_id,
            "speed": speed,
            "seed": seed,
        }
        files = {}
        if prompt_wav_path:
            files["prompt_wav"] = (
                os.path.basename(prompt_wav_path),
                open(prompt_wav_path, "rb"),
                "audio/wav",
            )
        r = requests.post(f"{self.base_url}/api/tts/zero_shot", data=data, files=files)
        r.raise_for_status()
        with open(output, "wb") as f:
            f.write(r.content)
        return output

    def cross_lingual(self, tts_text: str, prompt_wav_path: str,
                      output: str = "output.wav", speed: float = 1.0,
                      seed: int = 0) -> str:
        with open(prompt_wav_path, "rb") as f:
            r = requests.post(
                f"{self.base_url}/api/tts/cross_lingual",
                data={"tts_text": tts_text, "speed": speed, "seed": seed},
                files={"prompt_wav": (os.path.basename(prompt_wav_path), f, "audio/wav")},
            )
        r.raise_for_status()
        with open(output, "wb") as f:
            f.write(r.content)
        return output

    def instruct(self, tts_text: str, instruct_text: str,
                 prompt_wav_path: str = None, speaker_id: str = "",
                 output: str = "output.wav", speed: float = 1.0,
                 seed: int = 0) -> str:
        data = {
            "tts_text": tts_text,
            "instruct_text": instruct_text,
            "speaker_id": speaker_id,
            "speed": speed,
            "seed": seed,
        }
        files = {}
        if prompt_wav_path:
            files["prompt_wav"] = (
                os.path.basename(prompt_wav_path),
                open(prompt_wav_path, "rb"),
                "audio/wav",
            )
        r = requests.post(f"{self.base_url}/api/tts/instruct", data=data, files=files)
        r.raise_for_status()
        with open(output, "wb") as f:
            f.write(r.content)
        return output


# ─── CLI 模式 ───────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="CosyVoice API 客户端")
    parser.add_argument("--host", type=str, default="10.255.1.115")
    parser.add_argument("--port", type=int, default=9880)

    sub = parser.add_subparsers(dest="command")

    # health
    sub.add_parser("health", help="检查服务状态")

    # speakers
    sub.add_parser("speakers", help="列出可用音色")

    # register
    p_reg = sub.add_parser("register", help="注册零样本音色")
    p_reg.add_argument("--id", required=True, help="音色 ID")
    p_reg.add_argument("--text", required=True, help="参考文本")
    p_reg.add_argument("--wav", required=True, help="参考音频路径")

    # zero_shot
    p_zs = sub.add_parser("zero_shot", help="零样本语音克隆")
    p_zs.add_argument("--text", required=True, help="合成文本")
    p_zs.add_argument("--prompt_text", default="", help="参考音频文本")
    p_zs.add_argument("--prompt_wav", default=None, help="参考音频路径")
    p_zs.add_argument("--speaker_id", default="", help="已注册音色 ID")
    p_zs.add_argument("--output", default="output.wav", help="输出文件")
    p_zs.add_argument("--speed", type=float, default=1.0)

    # sft
    p_sft = sub.add_parser("sft", help="预训练音色合成")
    p_sft.add_argument("--text", required=True, help="合成文本")
    p_sft.add_argument("--speaker_id", required=True, help="音色 ID")
    p_sft.add_argument("--output", default="output.wav", help="输出文件")
    p_sft.add_argument("--speed", type=float, default=1.0)

    # cross_lingual
    p_cl = sub.add_parser("cross_lingual", help="跨语种克隆")
    p_cl.add_argument("--text", required=True, help="合成文本")
    p_cl.add_argument("--prompt_wav", required=True, help="参考音频路径")
    p_cl.add_argument("--output", default="output.wav")
    p_cl.add_argument("--speed", type=float, default=1.0)

    # instruct
    p_ins = sub.add_parser("instruct", help="指令控制合成")
    p_ins.add_argument("--text", required=True, help="合成文本")
    p_ins.add_argument("--instruct", required=True, help="指令文本")
    p_ins.add_argument("--prompt_wav", default=None)
    p_ins.add_argument("--speaker_id", default="")
    p_ins.add_argument("--output", default="output.wav")
    p_ins.add_argument("--speed", type=float, default=1.0)

    args = parser.parse_args()
    client = CosyVoiceClient(f"http://{args.host}:{args.port}")

    if args.command == "health":
        import json
        print(json.dumps(client.health(), indent=2, ensure_ascii=False))

    elif args.command == "speakers":
        spks = client.list_speakers()
        print(f"可用音色 ({len(spks)}):")
        for s in spks:
            print(f"  - {s}")

    elif args.command == "register":
        result = client.register_speaker(args.id, args.text, args.wav)
        print(result["message"])

    elif args.command == "zero_shot":
        out = client.zero_shot(args.text, args.prompt_text, args.prompt_wav,
                               args.speaker_id, args.output, args.speed)
        print(f"已保存到: {out}")

    elif args.command == "sft":
        out = client.sft(args.text, args.speaker_id, args.output, args.speed)
        print(f"已保存到: {out}")

    elif args.command == "cross_lingual":
        out = client.cross_lingual(args.text, args.prompt_wav, args.output, args.speed)
        print(f"已保存到: {out}")

    elif args.command == "instruct":
        out = client.instruct(args.text, args.instruct, args.prompt_wav,
                              args.speaker_id, args.output, args.speed)
        print(f"已保存到: {out}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
