# Audio8 TTS ONNX Runtime

这是
[Audio8-TTS-Preview-0.6B-ONNX-INT4](https://huggingface.co/Audio8/Audio8-TTS-Preview-0.6B-ONNX-INT4)
的纯 CPU ONNX Runtime 推理实现，包含命令行推理、网页与 HTTP 服务、流式 PCM
输出和参考音频音色注册。模型下载完成后，运行时不依赖 PyTorch、Transformers
或 Hugging Face Hub。

英文文档：[README.md](README.md)

## 精度与资源占用

| 组件 | 精度 |
|---|---|
| Slow/Fast AR 权重 | Weight-only INT4 |
| Activation、hidden state 和 KV cache | FP16 |
| Codec encoder/decoder | FP16 |
| 波形输出 | FP32 |

普通合成只加载 Slow AR、Fast AR 和 codec decoder 三个 session。在 16 GB
Apple M2 MacBook Air、5 个 ONNX Runtime 线程的测试环境中，服务加载后约占
1004 MiB，合成峰值约 1.1-1.2 GiB。注册音色前会释放在线模型，再单独加载
encoder；实测注册峰值约 1.55 GiB。不同系统和 ONNX Runtime 内存分配器下的
数值可能有所变化。

在线推理模型文件约 572 MiB；加上可选的音色注册 encoder，完整下载约 968 MiB。

## 安装

需要 Python 3.11 或更高版本。当前版本已在 macOS arm64 的
`CPUExecutionProvider` 上验证。

```bash
git clone https://github.com/Audio8-AI/Audio8_TTS.git
cd Audio8_TTS/onnx_runtime

python3 -m pip install -U "huggingface_hub[cli]"
hf download Audio8/Audio8-TTS-Preview-0.6B-ONNX-INT4 --local-dir model
bash setup.sh
```

模型目录结构应为：

```text
model/
|- slow_ar_int4.onnx(.data)
|- fast_ar_int4.onnx(.data)
|- codec_decoder_fp16.onnx(.data)
|- runtime_manifest.json
|- tokenizer/tokenizer.json
`- registration/
   |- codec_encoder_fp16.onnx(.data)
   `- registration_manifest.json
```

模型放在其他位置时设置 `ARKTTS_MODEL_DIR`。音色默认写入 `voices/`，可用
`ARKTTS_VOICES_DIR` 修改。

## 注册音色

音色 profile 由参考音频的 codec codes 和准确原文组成。参考音频必须为
0.5-30 秒、不超过 50 MiB，并且可由 libsndfile 读取。服务会自动转为单声道并
重采样到 44.1 kHz。

启动服务并打开 <http://127.0.0.1:8024>。没有音色时，页面会优先显示注册界面。

```bash
bash start_server.sh
```

也可以通过 HTTP 注册：

```bash
curl http://127.0.0.1:8024/api/voices/register \
  -F 'audio=@/absolute/path/reference.wav' \
  -F 'text=参考音频对应的准确原文' \
  -F 'name=speaker_a' \
  -F 'overwrite=false'
```

Encoder 只在注册期间加载。注册前会释放在线推理模型，完成后自动恢复。

## 推理

命令行：

```bash
bash run_infer.sh \
  --text "你好，这是 Audio8 TTS ONNX Runtime。" \
  --voice speaker_a \
  --max-new-tokens 256 \
  --output outputs/example.wav
```

命令同时写入 `[10,T]` 形状的 `outputs/example.npy` codec codes。

HTTP WAV：

```bash
curl http://127.0.0.1:8024/api/tts \
  -H 'Content-Type: application/json' \
  -d '{"text":"你好，这是 Audio8 TTS。","voice_name":"speaker_a","max_new_tokens":256}' \
  -o outputs/api.wav
```

OpenAI 兼容接口：

```bash
curl http://127.0.0.1:8024/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model":"arktts","input":"你好，这是 Audio8 TTS。","voice":"speaker_a","response_format":"wav"}' \
  -o outputs/openai.wav
```

`POST /api/tts/stream` 返回 NDJSON，其中音频块为 44.1 kHz、单声道、signed
16-bit little-endian PCM 的 Base64 数据。使用 `POST /api/tts/cancel` 取消当前流。

服务会串行处理合成和注册请求以限制内存占用，定位是本地使用和低并发 CPU 部署。

## 配置

| 环境变量 | 默认值 | 用途 |
|---|---|---|
| `ARKTTS_MODEL_DIR` | `onnx_runtime/model` | Hugging Face 模型目录 |
| `ARKTTS_VOICES_DIR` | `onnx_runtime/voices` | 已注册音色目录 |
| `ARKTTS_REGISTRATION_DIR` | `$ARKTTS_MODEL_DIR/registration` | Codec encoder 目录 |
| `ARKTTS_THREADS` | `5` | ONNX Runtime CPU 线程数 |
| `HOST` | `127.0.0.1` | 服务监听地址 |
| `PORT` | `8024` | 服务端口 |

使用 `bash stop_server.sh` 停止后台服务，日志写入 `service.log`。

## 负责任使用

参考原文必须与录音内容一致。噪声、过长或原文错误的参考音频会降低稳定性和
说话人相似度。克隆声音前应取得授权，并在适当场景明确披露合成音频。部署前请
评估准确性、安全性和法律合规性。

代码和模型权重采用 Apache License 2.0，详见仓库的 [LICENSE](../LICENSE) 和
[NOTICE](../NOTICE)。
