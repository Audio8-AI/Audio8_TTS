const DATA_URL = "./data.json?v=20260819-multilingual-5";
const MODEL_NAME = "Audio8-TTS-0.1B";

const LANGUAGE_GROUPS = [
  {
    id: "mandarin",
    label: "Mandarin",
    lang: "zh-CN",
    aliases: new Set(["zh", "zh-cn", "cmn", "chinese", "mandarin"]),
  },
  {
    id: "english",
    label: "English",
    lang: "en-US",
    aliases: new Set(["en", "en-us", "english"]),
  },
];

function escapeHTML(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function languageGroup(sample) {
  const value = String(sample.targetLanguage || sample.language || "").trim().toLowerCase();
  return LANGUAGE_GROUPS.find((group) => group.aliases.has(value));
}

function audioPlayer(path, label) {
  return `<audio controls preload="none" playsinline src="${escapeHTML(path)}" aria-label="${escapeHTML(label)}"></audio>`;
}

function validateMultilingual(samples) {
  if (samples === undefined) return [];
  if (!Array.isArray(samples)) {
    throw new Error("The multilingual data must contain an array.");
  }
  if (samples.length !== 5) {
    throw new Error(`Multilingual capability requires 5 non-Mandarin/English samples; found ${samples.length}.`);
  }

  return samples
    .map((sample, index) => {
      if (
        !sample?.id
        || !sample.language
        || !sample.nativeName
        || !sample.targetLanguage
        || !sample.targetText
        || !sample.reference?.language
        || !sample.reference?.text
        || !sample.reference?.audio
        || !sample.output?.audio
      ) {
        throw new Error(`Multilingual sample ${index + 1} is missing required demo data.`);
      }
      const targetLanguage = String(sample.targetLanguage).trim().toLowerCase();
      if (["zh", "zh-cn", "cmn", "chinese", "mandarin", "en", "en-us", "english"].includes(targetLanguage)) {
        throw new Error(`Multilingual sample ${index + 1} must not duplicate Mandarin or English.`);
      }
      return sample;
    })
    .sort((left, right) => {
      const leftOrder = Number(left.order);
      const rightOrder = Number(right.order);
      const leftRank = Number.isFinite(leftOrder) ? leftOrder : Number.MAX_SAFE_INTEGER;
      const rightRank = Number.isFinite(rightOrder) ? rightOrder : Number.MAX_SAFE_INTEGER;
      return leftRank - rightRank;
    });
}

function multilingualRow(sample, number) {
  const language = sample.language || sample.nativeName || sample.targetLanguage;
  return `
    <article class="language-row flat-multilingual-row" id="${escapeHTML(sample.id)}" data-sample-id="${escapeHTML(sample.id)}">
      <header class="language-cell">
        <span class="language-index">${String(number).padStart(2, "0")}</span>
        <div><h3>${escapeHTML(language)}</h3><p>${escapeHTML(sample.nativeName || "")}</p></div>
      </header>
      <div class="language-target">
        <p class="field-label">Target text</p>
        <p lang="${escapeHTML(sample.targetLanguage)}">${escapeHTML(sample.targetText)}</p>
      </div>
      <section class="compact-audio" aria-label="Reference audio">
        <div class="compact-heading"><span>Reference</span><small>${escapeHTML(language)}</small></div>
        ${audioPlayer(sample.reference.audio, `Reference audio for ${language}`)}
        <p lang="${escapeHTML(sample.reference.language)}">${escapeHTML(sample.reference.text)}</p>
      </section>
      <section class="compact-audio output-audio" aria-label="Audio8 generated audio">
        <div class="compact-heading"><span>Audio8 output</span><small>${escapeHTML(sample.nativeName || language)}</small></div>
        ${audioPlayer(sample.output.audio, `${MODEL_NAME} output in ${language}`)}
      </section>
    </article>`;
}

function renderMultilingual(samples) {
  if (!samples.length) return "";
  const rows = samples.map((sample, index) => multilingualRow(sample, index + 1)).join("");
  return `
    <section class="demo-section multilingual-section flat-multilingual-section" id="multilingual">
      <div class="shell">
        <header class="section-heading">
          <p class="section-index">${String(samples.length).padStart(2, "0")} languages</p>
          <h2>Multilingual Capability</h2>
          <p>The same everyday sentence is synthesized in each language, using a native reference recording for comparison.</p>
        </header>
        <div class="language-table-head" aria-hidden="true">
          <span>Language</span><span>Target text</span><span>Reference</span><span>Generated audio</span>
        </div>
        <div class="language-list">${rows}</div>
      </div>
    </section>`;
}

function validateSamples(payload) {
  if (!payload || !Array.isArray(payload.samples)) {
    throw new Error("The demo data must contain a samples array.");
  }

  const grouped = new Map(LANGUAGE_GROUPS.map((group) => [group.id, []]));
  payload.samples.forEach((sample, index) => {
    const group = languageGroup(sample);
    if (!group) {
      throw new Error(`Sample ${index + 1} has an unsupported target language.`);
    }
    if (
      !sample.id
      || !sample.targetText
      || !sample.reference?.language
      || !sample.reference?.text
      || !sample.reference?.audio
      || !sample.output?.audio
    ) {
      throw new Error(`Sample ${index + 1} is missing required demo data.`);
    }
    grouped.get(group.id).push({ sample, sourceIndex: index });
  });

  LANGUAGE_GROUPS.forEach((group) => {
    const samples = grouped.get(group.id);
    if (samples.length !== 15) {
      throw new Error(`${group.label} requires 15 samples; found ${samples.length}.`);
    }
    samples.sort((left, right) => {
      const leftOrder = Number(left.sample.order);
      const rightOrder = Number(right.sample.order);
      const orderDifference = (Number.isFinite(leftOrder) ? leftOrder : left.sourceIndex)
        - (Number.isFinite(rightOrder) ? rightOrder : right.sourceIndex);
      return orderDifference || left.sourceIndex - right.sourceIndex;
    });
  });

  return grouped;
}

function sampleCard(sample, group, number) {
  const referenceLabel = `${group.label} reference audio for sample ${number}`;
  const outputLabel = `${MODEL_NAME} output for sample ${number}`;
  return `
    <article class="sample-card flat-sample-card" id="${escapeHTML(sample.id)}" data-sample-id="${escapeHTML(sample.id)}">
      <header class="flat-sample-head">
        <span class="sample-number">${String(number).padStart(2, "0")}</span>
        <span class="flat-language-label">${escapeHTML(group.label)}</span>
      </header>
      <div class="target-block">
        <p class="field-label">Target text</p>
        <p class="target-copy" lang="${escapeHTML(group.lang)}">${escapeHTML(sample.targetText)}</p>
      </div>
      <div class="audio-pair">
        <section class="audio-block reference-block" aria-label="Reference audio">
          <div class="audio-heading"><span>Reference Audio</span><small>${escapeHTML(group.label)}</small></div>
          <div>
            ${audioPlayer(sample.reference.audio, referenceLabel)}
            <p class="reference-copy" lang="${escapeHTML(sample.reference.language)}">${escapeHTML(sample.reference.text)}</p>
          </div>
        </section>
        <section class="audio-block generated-block" aria-label="Audio8 generated audio">
          <div class="audio-heading"><span>Generated Audio</span><small>${escapeHTML(group.label)}</small></div>
          <div>${audioPlayer(sample.output.audio, outputLabel)}<p class="model-name">${MODEL_NAME}</p></div>
        </section>
      </div>
    </article>`;
}

function renderSamples(grouped) {
  const orderedSamples = [];
  const sampleCount = Math.max(...LANGUAGE_GROUPS.map((group) => grouped.get(group.id).length));
  for (let index = 0; index < sampleCount; index += 1) {
    LANGUAGE_GROUPS.forEach((group) => {
      const entry = grouped.get(group.id)[index];
      if (entry) orderedSamples.push({ sample: entry.sample, group });
    });
  }
  const cards = orderedSamples.map(({ sample, group }, index) => sampleCard(sample, group, index + 1));
  return `
    <section class="demo-section flat-demo-section" id="demo-samples">
      <div class="shell">
        <header class="section-heading flat-section-heading">
          <p class="section-index">30 samples / 15 per language</p>
          <h2>Demo Samples</h2>
        </header>
        <div class="sample-list">${cards.join("")}</div>
      </div>
    </section>`;
}

function bindAudio() {
  const players = [...document.querySelectorAll("audio")];
  players.forEach((audio) => audio.addEventListener("play", () => {
    players.forEach((other) => {
      if (other !== audio && !other.paused) other.pause();
    });
  }));
}

async function initialize() {
  const response = await fetch(DATA_URL);
  if (!response.ok) throw new Error(`Unable to load demo data: ${response.status}`);
  const payload = await response.json();
  const grouped = validateSamples(payload);
  const multilingual = validateMultilingual(payload.multilingual);
  document.querySelector("#demo-root").innerHTML = renderMultilingual(multilingual) + renderSamples(grouped);
  bindAudio();
}

initialize().catch((error) => {
  document.querySelector("#demo-root").innerHTML = `<p class="loading shell" role="alert">${escapeHTML(error.message)}</p>`;
  console.error(error);
});
