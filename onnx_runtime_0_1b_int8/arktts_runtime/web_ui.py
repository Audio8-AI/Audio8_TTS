from __future__ import annotations


TEST_PAGE = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Audio8 TTS 0.1B INT8 本地测试</title>
  <style>
    :root { color-scheme:light; --ink:#18201f; --muted:#65706e; --line:#d7dddb; --paper:#f5f7f6; --accent:#087f5b; --accent-dark:#066447; --danger:#b42318; }
    * { box-sizing:border-box; }
    body { margin:0; min-width:320px; background:var(--paper); color:var(--ink); font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; letter-spacing:0; }
    header { border-bottom:1px solid var(--line); background:#fff; }
    .bar, main { width:min(880px, calc(100% - 32px)); margin:auto; }
    .bar { min-height:64px; display:flex; align-items:center; justify-content:space-between; gap:16px; }
    h1 { margin:0; font-size:20px; font-weight:680; }
    .header-actions, .health { display:flex; align-items:center; gap:8px; }
    .health { color:var(--muted); font-size:13px; }
    .icon-button { width:36px; height:36px; padding:0; border:1px solid var(--line); border-radius:6px; background:#fff; color:var(--ink); font:22px/34px -apple-system,BlinkMacSystemFont,sans-serif; }
    .icon-button:hover { border-color:#aeb8b5; background:#f7f9f8; }
    .icon-button:disabled { opacity:.45; }
    .dot { width:9px; height:9px; border-radius:50%; background:#a3aaa8; }
    .dot.ok { background:#12a371; }
    main { padding:32px 0 48px; }
    .meta { display:flex; flex-wrap:wrap; gap:8px 18px; margin-bottom:18px; color:var(--muted); font-size:13px; }
    .metrics { display:grid; grid-template-columns:repeat(3, 1fr); border-top:1px solid var(--line); border-bottom:1px solid var(--line); margin-bottom:24px; }
    .metric { min-width:0; padding:15px 0; }
    .metric + .metric { padding-left:18px; border-left:1px solid var(--line); }
    .metric span { display:block; color:var(--muted); font-size:12px; }
    .metric strong { display:block; margin-top:2px; font-size:19px; font-weight:680; font-variant-numeric:tabular-nums; }
    .memory-track { grid-column:1 / -1; height:3px; margin-top:-3px; overflow:hidden; background:#dfe5e3; }
    .memory-fill { width:0; height:100%; background:var(--accent); transition:width .35s ease; }
    .tabs { display:flex; width:100%; margin-bottom:24px; border-bottom:1px solid var(--line); }
    .tab { height:42px; padding:0 18px; border:0; border-bottom:2px solid transparent; border-radius:0; background:transparent; color:var(--muted); }
    .tab:hover { background:transparent; color:var(--ink); }
    .tab.active { border-bottom-color:var(--accent); color:var(--ink); }
    .view[hidden] { display:none; }
    label { display:block; margin:0 0 7px; font-size:13px; font-weight:650; }
    textarea, select, input[type=number], input[type=text], input[type=file] { width:100%; border:1px solid #bfc8c5; border-radius:6px; background:#fff; color:var(--ink); font:inherit; outline:none; }
    textarea:focus, select:focus, input:focus { border-color:var(--accent); box-shadow:0 0 0 3px rgba(8,127,91,.12); }
    textarea { min-height:156px; resize:vertical; padding:13px 14px; }
    select, input[type=number], input[type=text], input[type=file] { height:42px; padding:0 11px; }
    input[type=file] { padding:8px; }
    .row { display:grid; grid-template-columns:1.35fr repeat(3, 1fr); gap:14px; margin-top:18px; }
    .actions { display:flex; align-items:center; gap:10px; margin-top:24px; }
    button, .download { height:42px; border-radius:6px; padding:0 17px; font:650 14px/40px inherit; cursor:pointer; text-decoration:none; }
    button { border:1px solid var(--accent); background:var(--accent); color:#fff; }
    button:hover { background:var(--accent-dark); }
    button:disabled { cursor:not-allowed; opacity:.55; }
    .download { display:none; border:1px solid #bfc8c5; background:#fff; color:var(--ink); }
    .status { min-height:22px; margin:18px 0 10px; color:var(--muted); font-size:13px; }
    .status.error { color:var(--danger); }
    audio { display:none; width:100%; margin-top:10px; }
    audio.ready { display:block; }
    .register-grid { display:grid; grid-template-columns:1fr 1fr; gap:14px; margin-bottom:18px; }
    .register-text { min-height:112px; }
    .check { display:flex; align-items:center; gap:8px; margin:14px 0 0; font-weight:500; }
    .check input { width:16px; height:16px; accent-color:var(--accent); }
    .audio-preview { display:none; margin:14px 0 0; }
    .audio-preview.ready { display:block; }
    @media (max-width:700px) { .row { grid-template-columns:1fr 1fr; } main { padding-top:24px; } }
    @media (max-width:440px) { .row, .register-grid { grid-template-columns:1fr; } .metrics { grid-template-columns:1fr 1fr; } .metric:nth-child(3) { grid-column:1 / -1; padding-left:0; border-left:0; border-top:1px solid var(--line); } .actions { align-items:stretch; flex-direction:column; } button, .download { width:100%; text-align:center; } }
  </style>
</head>
<body>
  <header><div class="bar"><h1>Audio8 TTS 0.1B INT8</h1><div class="header-actions"><div class="health"><span id="dot" class="dot"></span><span id="health">正在加载模型</span></div><button id="reloadRuntime" class="icon-button" type="button" title="重新加载模型" aria-label="重新加载模型">&#8635;</button></div></div></header>
  <main>
    <div class="meta"><span id="model">模型：-</span><span id="precision">DualAR：-</span><span id="codecPrecision">Codec：-</span><span id="provider">执行器：-</span></div>
    <section class="metrics" aria-label="运行状态">
      <div class="metric"><span>当前内存</span><strong id="memoryCurrent">读取中</strong></div>
      <div class="metric"><span>峰值内存</span><strong id="memoryPeak">读取中</strong></div>
      <div class="metric"><span>运行时间</span><strong id="uptime">-</strong></div>
      <div class="memory-track"><div id="memoryFill" class="memory-fill"></div></div>
    </section>
    <div class="tabs" role="tablist"><button id="ttsTab" class="tab active" role="tab">语音合成</button><button id="registerTab" class="tab" role="tab">音频注册</button></div>
    <section id="ttsView" class="view">
      <label for="text">合成文本</label>
      <textarea id="text">你好，这是 Audio8 TTS 0.1B INT8 的本地语音合成测试。</textarea>
      <div class="row">
        <div><label for="voice">音色</label><select id="voice"></select></div>
        <div><label for="temperature">Temperature</label><input id="temperature" type="number" value="0.7" min="0.05" max="2" step="0.05"></div>
        <div><label for="topP">Top P</label><input id="topP" type="number" value="0.9" min="0.05" max="1" step="0.05"></div>
        <div><label for="seed">Seed</label><input id="seed" type="number" value="42" min="0" step="1"></div>
      </div>
      <div class="actions"><button id="generate" disabled>生成语音</button><a id="download" class="download" download="audio8_0.1b_int8_output.wav">下载 WAV</a></div>
      <div id="status" class="status"></div>
      <audio id="audio" controls></audio>
    </section>
    <section id="registerView" class="view" hidden>
      <div class="register-grid">
        <div><label for="referenceAudio">参考音频</label><input id="referenceAudio" type="file" accept="audio/*,.wav,.flac,.mp3,.m4a"></div>
        <div><label for="voiceName">音色名称</label><input id="voiceName" type="text" maxlength="64" placeholder="speaker_a"></div>
      </div>
      <label for="referenceText">参考音频原文</label>
      <textarea id="referenceText" class="register-text"></textarea>
      <audio id="referencePreview" class="audio-preview" controls></audio>
      <label class="check"><input id="overwriteVoice" type="checkbox">覆盖同名音色</label>
      <div class="actions"><button id="registerVoice" disabled>注册音色</button></div>
      <div id="registerStatus" class="status"></div>
    </section>
  </main>
  <script>
    const $ = id => document.getElementById(id);
    let audioUrl = null;
    let referenceUrl = null;
    let registrationAvailable = false;
    function selectView(name) {
      const register = name === 'register';
      $('ttsView').hidden = register;
      $('registerView').hidden = !register;
      $('ttsTab').classList.toggle('active', !register);
      $('registerTab').classList.toggle('active', register);
    }
    async function errorMessage(response) {
      const text = await response.text();
      try { return JSON.parse(text).detail || text; } catch (_) { return text || `HTTP ${response.status}`; }
    }
    async function refreshVoices(selected) {
      const response = await fetch('/api/voices');
      if (!response.ok) throw new Error('读取音色失败');
      const data = await response.json();
      const names = data.voices.map(voice => typeof voice === 'string' ? voice : voice.name);
      $('voice').replaceChildren(...names.map(name => new Option(name, name)));
      if (selected && names.includes(selected)) $('voice').value = selected;
      $('generate').disabled = names.length === 0;
      return names;
    }
    function formatUptime(seconds) {
      const value = Math.max(0, Math.floor(seconds));
      const hours = Math.floor(value / 3600);
      const minutes = Math.floor((value % 3600) / 60);
      const secs = value % 60;
      return hours ? `${hours}时 ${minutes}分 ${secs}秒` : `${minutes}分 ${secs}秒`;
    }
    async function refreshMetrics() {
      try {
        const response = await fetch('/api/system', {cache:'no-store'});
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const data = await response.json();
        $('memoryCurrent').textContent = data.memory.current_mb == null ? '不可用' : `${(data.memory.current_mb / 1024).toFixed(2)} GB`;
        $('memoryPeak').textContent = data.memory.peak_mb == null ? '不可用' : `${(data.memory.peak_mb / 1024).toFixed(2)} GB`;
        $('uptime').textContent = formatUptime(data.uptime_seconds);
        const scale = Math.max(4096, data.memory.peak_mb || 0);
        $('memoryFill').style.width = `${Math.min(100, ((data.memory.current_mb || 0) / scale) * 100)}%`;
      } catch (_) {
        $('memoryCurrent').textContent = '读取失败';
      }
    }
    async function initialize() {
      try {
        const [health, registrationStatus] = await Promise.all([fetch('/api/health'), fetch('/api/registration/status')]);
        if (!health.ok || !registrationStatus.ok) throw new Error('服务初始化失败');
        const healthData = await health.json();
        const registrationData = await registrationStatus.json();
        $('model').textContent = `模型：${healthData.model || 'Audio8 0.1B'}`;
        $('precision').textContent = `DualAR：${healthData.precision.toUpperCase()}`;
        $('codecPrecision').textContent = `Codec：${healthData.codec_precision.toUpperCase()}`;
        $('provider').textContent = `执行器：${healthData.providers.slow[0]}`;
        const voiceNames = await refreshVoices();
        registrationAvailable = registrationData.available;
        $('registerVoice').disabled = !registrationAvailable;
        $('registerStatus').textContent = registrationAvailable ? '' : registrationData.reason;
        $('dot').classList.add('ok');
        $('health').textContent = '模型已就绪';
        $('generate').disabled = voiceNames.length === 0;
        $('status').textContent = voiceNames.length ? '' : '没有可用音色';
        if (voiceNames.length === 0 && registrationAvailable) selectView('register');
      } catch (error) {
        $('health').textContent = '服务不可用';
        $('status').textContent = error.message;
        $('status').classList.add('error');
      }
    }
    $('generate').addEventListener('click', async () => {
      const text = $('text').value.trim();
      if (!text) { $('status').textContent = '请输入合成文本'; return; }
      $('generate').disabled = true;
      $('status').classList.remove('error');
      $('status').textContent = '正在生成语音...';
      $('audio').classList.remove('ready');
      $('download').style.display = 'none';
      const started = performance.now();
      try {
        const response = await fetch('/api/tts', {
          method:'POST', headers:{'Content-Type':'application/json'},
          body:JSON.stringify({text, voice_name:$('voice').value, max_new_tokens:1024, temperature:Number($('temperature').value), top_p:Number($('topP').value), seed:Number($('seed').value)})
        });
        if (!response.ok) throw new Error(await errorMessage(response));
        const blob = await response.blob();
        if (audioUrl) URL.revokeObjectURL(audioUrl);
        audioUrl = URL.createObjectURL(blob);
        $('audio').src = audioUrl;
        $('audio').classList.add('ready');
        $('download').href = audioUrl;
        $('download').style.display = 'inline-block';
        $('status').textContent = `生成完成，用时 ${((performance.now()-started)/1000).toFixed(2)} 秒`;
        await $('audio').play().catch(() => {});
      } catch (error) {
        $('status').textContent = `生成失败：${error.message}`;
        $('status').classList.add('error');
      } finally { $('generate').disabled = false; }
    });
    $('ttsTab').addEventListener('click', () => selectView('tts'));
    $('registerTab').addEventListener('click', () => selectView('register'));
    $('referenceAudio').addEventListener('change', () => {
      const file = $('referenceAudio').files[0];
      if (referenceUrl) URL.revokeObjectURL(referenceUrl);
      if (!file) { $('referencePreview').classList.remove('ready'); return; }
      referenceUrl = URL.createObjectURL(file);
      $('referencePreview').src = referenceUrl;
      $('referencePreview').classList.add('ready');
    });
    $('registerVoice').addEventListener('click', async () => {
      const file = $('referenceAudio').files[0];
      const name = $('voiceName').value.trim();
      const text = $('referenceText').value.trim();
      if (!file || !name || !text) { $('registerStatus').textContent = '请选择音频并填写音色名称和准确原文'; return; }
      $('registerVoice').disabled = true;
      $('registerStatus').classList.remove('error');
      $('registerStatus').textContent = '正在注册音色...';
      const form = new FormData();
      form.append('audio', file);
      form.append('name', name);
      form.append('text', text);
      form.append('overwrite', $('overwriteVoice').checked ? 'true' : 'false');
      try {
        const response = await fetch('/api/voices/register', {method:'POST', body:form});
        if (!response.ok) throw new Error(await errorMessage(response));
        const result = await response.json();
        await refreshVoices(result.voice.name);
        $('registerStatus').textContent = `注册完成：${result.voice.name}（${result.voice.shape[1]} 帧）`;
      } catch (error) {
        $('registerStatus').textContent = `注册失败：${error.message}`;
        $('registerStatus').classList.add('error');
      } finally { $('registerVoice').disabled = !registrationAvailable; }
    });
    $('reloadRuntime').addEventListener('click', async () => {
      const button = $('reloadRuntime');
      button.disabled = true;
      $('dot').classList.remove('ok');
      $('health').textContent = '正在重新加载模型';
      try {
        const response = await fetch('/api/runtime/reload', {method:'POST'});
        const result = await response.json();
        if (!response.ok) throw new Error(result.detail || `HTTP ${response.status}`);
        $('precision').textContent = `DualAR：${result.precision.toUpperCase()}`;
        $('codecPrecision').textContent = `Codec：${result.codec_precision.toUpperCase()}`;
        await refreshVoices();
        await refreshMetrics();
        $('dot').classList.add('ok');
        $('health').textContent = `模型已重新加载（${result.reload_seconds.toFixed(2)} 秒）`;
      } catch (error) {
        $('health').textContent = `重新加载失败：${error.message}`;
      } finally { button.disabled = false; }
    });
    initialize();
    refreshMetrics();
    setInterval(refreshMetrics, 2000);
  </script>
</body>
</html>"""
