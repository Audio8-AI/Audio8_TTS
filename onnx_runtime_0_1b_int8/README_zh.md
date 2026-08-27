# Audio8 0.1B INT8 ONNX Runtime

这是 `Audio8/audio8-TTS-0.1B-ONNX-INT8` 的独立 CPU ONNX Runtime。它放在
单独目录中，因为仓库现有的 `onnx_runtime/` 是给 0.6B INT4 图使用的，不能
直接处理 0.1B Falcon-H1 hybrid 图。

0.1B Slow AR 每次只接受一个 `[1, 11, 1]` token column，并且要在调用之间保存：

```text
cache_keys    [24, 1, 2, 2048, 64]
cache_values  [24, 1, 2, 2048, 64]
conv_states   [24, 1, 896, 4]
ssm_states    [24, 1, 24, 32, 64]
```

本实现会逐位置执行 prompt prefill，写回 Slow AR 返回的 state delta，处理
4097 维 compact semantic logits，然后调用四层 Fast AR 和 FP16 codec decoder。

## 安装

支持 Python 3.10 及以上版本，Jetson issue 中的 Python 3.10 也可以使用：

```bash
python3 -m pip install -U "huggingface_hub[cli]"
hf download Audio8/audio8-TTS-0.1B-ONNX-INT8 --local-dir model
bash setup.sh
python scripts/register_default_voice.py
```

模型文件放在本目录的 `model/` 下。`registration/` 下的 encoder 是可选的，
只在从音频注册新音色时需要。默认音色脚本只读取 `reference_codes.npy`。

## 命令行推理

```bash
bash run_infer.sh \
  --text "这是一个中文测试" \
  --voice default \
  --max-new-tokens 128 \
  --output outputs/test.wav
```

同时会生成 `[10, frames]` 形状的 `outputs/test.npy`。模型或音色放在其他位置时，
可设置 `ARKTTS_MODEL_DIR`、`ARKTTS_VOICES_DIR`。

## HTTP 服务

服务同时提供与 0.6B ONNX Runtime 相同的本地网页界面。启动后打开
`http://127.0.0.1:8024`，即可输入文本、选择音色、播放或下载 WAV、注册参考音色、
查看内存状态和重新加载运行时；开发者 API 文档仍在 `/docs`。

```bash
bash start_server.sh
curl http://127.0.0.1:8024/api/health
```

服务提供 `/api/tts`、`/api/tts/stream`、`/api/tts/cancel`、音色查询和注册接口，
以及兼容 OpenAI 的 `/v1/audio/speech`。使用 `bash stop_server.sh` 停止服务。

## 测试

```bash
"$PWD/.venv/bin/python" -m pytest -q tests
```

请在获得授权后进行音色克隆，并在适当场景披露合成音频。
