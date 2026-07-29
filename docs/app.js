const CASE_SECTIONS = [
  {
    id: "pronunciation-challenges",
    index: "02 / 06",
    title: "Pronunciation Challenges",
    description: "Short cases for irregular spelling, contextual readings, place names, and classical readings.",
    groups: [
      {
        title: "Selected challenges",
        match: (sample) => sample.group === "selected_short",
        order: [
          "classical_polyphone_zh_01",
          "cross_enref_zh_places_01",
          "classical_polyphone_zh_02",
          "chaos_tear_en_01",
          "chaos_schism_en_01",
        ],
      },
    ],
  },
  {
    id: "tongue-twisters",
    index: "03 / 06",
    title: "Tongue Twisters",
    description: "Dense consonants and repeated syllables in English and Mandarin.",
    groups: [
      { title: "English", match: (sample) => sample.group === "tongue_twisters_en" },
      { title: "Mandarin", match: (sample) => sample.group === "tongue_twisters_zh" },
    ],
  },
  {
    id: "classical-chinese",
    index: "04 / 06",
    title: "Complete Classical Chinese",
    description: "Complete works with sustained continuity, controlled phrasing, and full closing lines.",
    groups: [
      { title: "Full works", match: (sample) => sample.group === "classical_chinese" },
    ],
  },
  {
    id: "cross-lingual",
    index: "05 / 06",
    title: "Cross-Lingual Voice Cloning",
    description: "Reference and target languages cross in both directions while speaker identity remains the anchor.",
    groups: [
      { title: "Mandarin reference to English", match: (sample) => sample.direction === "zh_ref_to_en" },
      {
        title: "English reference to Mandarin",
        match: (sample) => sample.direction === "en_ref_to_zh" && sample.group !== "selected_short",
      },
    ],
  },
  {
    id: "hard-cases",
    index: "06 / 06",
    title: "Hard Cases",
    description: "Uncommon vocabulary, technical terminology, acronyms, and initialisms.",
    groups: [
      { title: "Difficult terms", match: (sample) => sample.hardCaseType === "difficult_terms_names" },
      { title: "Acronyms and initialisms", match: (sample) => sample.hardCaseType === "acronyms_initialisms" },
    ],
  },
];

const LANGUAGE_LABELS = {
  "zh-CN": "Mandarin",
  "en-US": "English",
  zh: "Mandarin",
  en: "English",
  ja: "Japanese",
};

function escapeHTML(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function languageLabel(value) {
  return LANGUAGE_LABELS[value] || value;
}

function player(path, label) {
  return `<audio controls preload="none" playsinline aria-label="${escapeHTML(label)}"><source src="${escapeHTML(path)}" type="audio/mpeg"></audio>`;
}

function compareSamples(left, right, preferredOrder = []) {
  const leftPreferred = preferredOrder.indexOf(left.id);
  const rightPreferred = preferredOrder.indexOf(right.id);
  const leftRank = leftPreferred === -1 ? Number.MAX_SAFE_INTEGER : leftPreferred;
  const rightRank = rightPreferred === -1 ? Number.MAX_SAFE_INTEGER : rightPreferred;
  if (leftRank !== rightRank) return leftRank - rightRank;
  const languageRank = { "en-US": 0, "zh-CN": 1 };
  return (languageRank[left.targetLanguage] ?? 2) - (languageRank[right.targetLanguage] ?? 2) || left.order - right.order;
}

function multilingualRow(sample) {
  return `
    <article class="language-row" data-sample-id="${escapeHTML(sample.id)}">
      <header class="language-cell">
        <span class="language-index">${String(sample.order).padStart(2, "0")}</span>
        <div><h3>${escapeHTML(sample.language)}</h3><p>${escapeHTML(sample.nativeName)}</p></div>
      </header>
      <div class="language-target">
        <p class="field-label">Target text</p>
        <p lang="${escapeHTML(sample.targetLanguage)}">${escapeHTML(sample.targetText)}</p>
      </div>
      <section class="compact-audio" aria-label="Reference audio">
        <div class="compact-heading"><span>Reference</span><small>${escapeHTML(languageLabel(sample.reference.language))}</small></div>
        ${player(sample.reference.audio, `Reference audio for ${sample.language}`)}
        <p lang="${escapeHTML(sample.reference.language)}">${escapeHTML(sample.reference.text)}</p>
      </section>
      <section class="compact-audio output-audio" aria-label="Generated audio">
        <div class="compact-heading"><span>Audio8 output</span><small>${escapeHTML(sample.nativeName)}</small></div>
        ${player(sample.output.audio, `Audio8 output in ${sample.language}`)}
      </section>
    </article>`;
}

function renderMultilingual(samples) {
  return `
    <section class="demo-section multilingual-section" id="multilingual">
      <div class="shell">
        <header class="section-heading">
          <p class="section-index">01 / 06</p>
          <h2>Multilingual Capability</h2>
          <p>Zero-shot synthesis across the 11 languages recommended for the preview checkpoint.</p>
        </header>
        <div class="language-table-head" aria-hidden="true">
          <span>Language</span><span>Target text</span><span>Reference</span><span>Generated audio</span>
        </div>
        <div class="language-list">${samples.sort((a, b) => a.order - b.order).map(multilingualRow).join("")}</div>
      </div>
    </section>`;
}

function sampleCard(sample, number) {
  const referenceLanguage = languageLabel(sample.reference.language);
  const targetLanguage = languageLabel(sample.targetLanguage);
  const direction = sample.direction ? `${referenceLanguage} ref / ${targetLanguage} out` : targetLanguage;
  const expected = sample.expectedReading
    ? `<p class="expected"><strong>Expected reading</strong>${escapeHTML(sample.expectedReading)}</p>`
    : "";
  const source = sample.source
    ? `<p class="source"><strong>Source</strong><a href="${escapeHTML(sample.source.url)}" target="_blank" rel="noreferrer">${escapeHTML(sample.source.label)}</a></p>`
    : "";
  return `
    <article class="sample-card" id="${escapeHTML(sample.id)}" data-sample-id="${escapeHTML(sample.id)}">
      <header class="sample-head">
        <span class="sample-number">${String(number).padStart(2, "0")}</span>
        <div><h3>${escapeHTML(sample.title)}</h3><p>${escapeHTML(sample.listenFor)}</p></div>
        <span class="language-badge">${escapeHTML(direction)}</span>
      </header>
      <div class="target-block">
        <p class="field-label">Target text</p>
        <div><p class="target-copy" lang="${escapeHTML(sample.targetLanguage)}">${escapeHTML(sample.targetText)}</p>${expected}${source}</div>
      </div>
      <div class="audio-pair">
        <section class="audio-block reference-block" aria-label="Reference audio">
          <div class="audio-heading"><span>Reference audio</span><small>${escapeHTML(referenceLanguage)}</small></div>
          <div>${player(sample.reference.audio, `Reference audio for ${sample.title}`)}<p class="reference-copy" lang="${escapeHTML(sample.reference.language)}">${escapeHTML(sample.reference.text)}</p></div>
        </section>
        <section class="audio-block generated-block" aria-label="Audio8 generated audio">
          <div class="audio-heading"><span>Generated audio</span><small>${escapeHTML(targetLanguage)}</small></div>
          <div>${player(sample.output.audio, `Audio8 generated audio for ${sample.title}`)}<p class="model-name">Audio8-TTS-0.6B</p></div>
        </section>
      </div>
    </article>`;
}

function renderCaseSection(definition, samples, state) {
  const groups = definition.groups.map((group) => {
    const matching = samples.filter(group.match).sort((a, b) => compareSamples(a, b, group.order));
    return `<div class="subsection"><h3 class="subsection-title">${escapeHTML(group.title)} <span>${String(matching.length).padStart(2, "0")}</span></h3><div class="sample-list">${matching.map((sample) => sampleCard(sample, state.next++)).join("")}</div></div>`;
  }).join("");
  return `<section class="demo-section" id="${definition.id}"><div class="shell"><header class="section-heading"><p class="section-index">${definition.index}</p><h2>${escapeHTML(definition.title)}</h2><p>${escapeHTML(definition.description)}</p></header>${groups}</div></section>`;
}

function bindAudio() {
  const players = [...document.querySelectorAll("audio")];
  players.forEach((audio) => audio.addEventListener("play", () => {
    players.forEach((other) => { if (other !== audio && !other.paused) other.pause(); });
  }));
}

async function initialize() {
  const response = await fetch("data.json?v=20260728-small-codec-r7");
  if (!response.ok) throw new Error(`Unable to load preview data: ${response.status}`);
  const payload = await response.json();
  const state = { next: 1 };
  document.querySelector("#demo-root").innerHTML = renderMultilingual(payload.multilingual)
    + CASE_SECTIONS.map((section) => renderCaseSection(section, payload.cases, state)).join("");
  bindAudio();
}

initialize().catch((error) => {
  document.querySelector("#demo-root").innerHTML = `<p class="loading shell">${escapeHTML(error.message)}</p>`;
  console.error(error);
});
