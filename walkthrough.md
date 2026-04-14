# CosyVoice 语音克隆 API 服务

## 创建的文件

| 文件 | 说明 |
|------|------|
| [api_server.py](file:///\\10.255.1.115\CosyVoice\api_server.py) | API 服务端 |
| [api_client.py](file:///\\10.255.1.115\CosyVoice\api_client.py) | Python 客户端 SDK + CLI |

---

## 启动服务

```bash
# 在 GPU 服务器上启动 (默认加载 CosyVoice3, 监听 0.0.0.0:9880)
cd /CosyVoice
python api_server.py

# 指定模型和端口
python api_server.py --model_dir pretrained_models/CosyVoice2-0.5B --port 8000
```

启动后访问 `http://10.255.1.115:9880/docs` 可查看交互式 API 文档 (Swagger UI)。

---

## API 端点一览

| 方法 | 路径 | 功能 |
|------|------|------|
| GET | `/api/health` | 健康检查 & 模型信息 |
| GET | `/api/speakers` | 列出所有可用音色 |
| POST | `/api/speakers/register` | 注册零样本音色 (上传参考音频) |
| DELETE | `/api/speakers/{speaker_id}` | 删除已注册音色 |
| POST | `/api/tts/sft` | 预训练音色合成 |
| POST | `/api/tts/zero_shot` | **零样本语音克隆** (核心) |
| POST | `/api/tts/cross_lingual` | 跨语种克隆 |
| POST | `/api/tts/instruct` | 自然语言指令控制 |
| POST | `/api/tts/stream/zero_shot` | 流式零样本克隆 (PCM 流) |

所有 TTS 接口返回 **WAV 文件**（流式接口除外）。

---

## 使用示例

### Python SDK

```python
from api_client import CosyVoiceClient

client = CosyVoiceClient("http://10.255.1.115:9880")

# 1. 零样本克隆 — 上传参考音频实时克隆
client.zero_shot(
    tts_text="收到好友从远方寄来的生日礼物，那份惊喜让我心中充满了快乐。",
    prompt_text="希望你以后能够做的比我还好呦。",
    prompt_wav_path="ref_audio.wav",
    output="cloned.wav"
)

# 2. 注册音色后复用 (无需每次上传)
client.register_speaker("张三", "希望你以后能够做的比我还好呦。", "ref_audio.wav")
client.zero_shot("今天天气真不错", speaker_id="张三", output="reuse.wav")

# 3. 跨语种克隆
client.cross_lingual("<|en|>Hello, nice to meet you!", "ref_cn.wav", output="en.wav")

# 4. 指令控制 (方言/情感/语速)
client.instruct(
    "好少咯，一般系放嗰啲国庆中秋可能会咯。",
    instruct_text="You are a helpful assistant. 请用广东话表达。<|endofprompt|>",
    prompt_wav_path="ref_audio.wav",
    output="cantonese.wav"
)
```

### CLI 模式

```bash
# 检查服务
python api_client.py health

# 列出音色
python api_client.py speakers

# 注册音色
python api_client.py register --id 张三 --text "参考文本" --wav ref.wav

# 零样本克隆
python api_client.py zero_shot --text "要说的话" --prompt_wav ref.wav --prompt_text "参考文本" --output out.wav

# 使用已注册音色
python api_client.py zero_shot --text "要说的话" --speaker_id 张三 --output out.wav
```

### cURL

```bash
# 健康检查
curl http://10.255.1.115:9880/api/health

# 零样本克隆
curl -X POST http://10.255.1.115:9880/api/tts/zero_shot \
  -F "tts_text=你好世界" \
  -F "prompt_text=希望你以后能够做的比我还好呦。" \
  -F "prompt_wav=@ref_audio.wav" \
  -o output.wav

# 使用已注册音色
curl -X POST http://10.255.1.115:9880/api/tts/zero_shot \
  -F "tts_text=你好世界" \
  -F "speaker_id=张三" \
  -o output.wav

# 注册音色
curl -X POST http://10.255.1.115:9880/api/speakers/register \
  -F "speaker_id=张三" \
  -F "prompt_text=参考音频文本" \
  -F "prompt_wav=@ref_audio.wav"
```

---

## 设计要点

- **返回标准 WAV**：非流式接口返回完整 WAV 文件，可直接播放/保存
- **音色持久化**：注册的音色保存到 `spk2info.pt`，重启不丢失
- **自动识别模型版本**：通过 `AutoModel` 自动适配 v1/v2/v3
- **局域网可用**：监听 `0.0.0.0`，同一局域网任何设备可访问
- **Swagger 文档**：`/docs` 路径自带交互式文档，可直接在浏览器里测试
