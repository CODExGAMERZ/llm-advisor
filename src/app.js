/**
 * LLM Hardware Advisor — Main Application Controller
 * Client-side ESM frontend. No bundler, no build step.
 */

import {
  calcTotalVRAM,
  calcWeightMemory,
  calcMinRAM,
  calcDiskSpace,
  calcMaxContext,
} from './engine/memoryCalc.js';

import {
  calcFitScore,
  getFitLabel,
  getFitLabelModeB,
  getUsagePercent,
  getUsageColor,
  findBestQuant,
} from './engine/fitScore.js';

import {
  getSpeedTiers,
  estimateSpeedForGPU,
  formatSpeedRange,
} from './engine/speedEstimate.js';

// ─── Constants ────────────────────────────────────────────────────────────────

const CTX_STEPS = [512, 2048, 4096, 8192, 16384, 32768, 65536, 131072];
const RAM_STEPS = [8, 16, 32, 48, 64, 96, 128, 192, 256];

const QUANT_INFO = {
  'FP16':    'Full precision baseline',
  'BF16':    'Brain Float — same size, better training',
  'Q8_0':    'Near-lossless quantization',
  'Q6_K':    'Excellent quality, good compression',
  'Q5_K_M':  'Great balance of quality & size',
  'Q5_K_S':  'Slightly smaller than Q5_K_M',
  'Q4_K_M':  'Best quality-to-size for most users',
  'Q4_K_S':  'Slightly smaller, marginal loss',
  'Q4_0':    '4-bit basic — fast legacy',
  'Q3_K_M':  'Noticeable quality loss, very small',
  'Q3_K_S':  'Aggressive compression',
  'Q2_K':    'Extreme compression, significant loss',
  'IQ2_XXS': 'Max compression, for experimentation',
  'AWQ':     'Activation-aware 4-bit — GPU optimized',
  'GPTQ':    'GPU 4-bit — optimized for GPU inference',
};

const QUANT_BITS = {
  'FP32': 32, 'FP16': 16, 'BF16': 16, 'INT8': 8, 'Q8_0': 8, 'Q6_K': 6.5,
  'Q5_K_M': 5.5, 'Q5_K_S': 5, 'Q4_K_M': 4.5, 'Q4_K_S': 4.25, 'Q4_0': 4,
  'Q3_K_M': 3.35, 'Q3_K_S': 3, 'Q2_K': 2.5, 'IQ2_XXS': 2, 'AWQ': 4, 'GPTQ': 4,
};

const QUANT_QUALITY = {
  'FP32':    { score: '100%', loss: '0.0%', text: 'Baseline',  level: 'lossless', color: '#00D4AA' },
  'FP16':    { score: '100%', loss: '0.0%', text: 'Baseline',  level: 'lossless', color: '#00D4AA' },
  'BF16':    { score: '100%', loss: '0.0%', text: 'Baseline',  level: 'lossless', color: '#00D4AA' },
  'INT8':    { score: '99.9%', loss: '0.1%', text: 'Negligible', level: 'low', color: '#00D4AA' },
  'Q8_0':    { score: '99.9%', loss: '0.1%', text: 'Negligible', level: 'low', color: '#00D4AA' },
  'Q6_K':    { score: '99.8%', loss: '0.2%', text: 'Negligible', level: 'low', color: '#00D4AA' },
  'Q5_K_M':  { score: '99.6%', loss: '0.4%', text: 'Very Low',  level: 'low', color: '#00D4AA' },
  'Q5_K_S':  { score: '99.3%', loss: '0.7%', text: 'Very Low',  level: 'low', color: '#00D4AA' },
  'Q4_K_M':  { score: '99.0%', loss: '1.0%', text: 'Low',       level: 'low', color: '#00D4AA' },
  'Q4_K_S':  { score: '98.5%', loss: '1.5%', text: 'Low-Mid',   level: 'mid', color: '#F59E0B' },
  'Q4_0':    { score: '98.0%', loss: '2.0%', text: 'Moderate',  level: 'mid', color: '#F59E0B' },
  'Q3_K_M':  { score: '94.0%', loss: '6.0%', text: 'High',      level: 'high', color: '#EF4444' },
  'Q3_K_S':  { score: '91.5%', loss: '8.5%', text: 'High',      level: 'high', color: '#EF4444' },
  'Q2_K':    { score: '78.0%', loss: '22.0%', text: 'Severe',   level: 'severe', color: '#EF4444' },
  'IQ2_XXS': { score: '65.0%', loss: '35.0%', text: 'Extreme',  level: 'severe', color: '#EF4444' },
  'AWQ':     { score: '99.0%', loss: '1.0%', text: 'Low',        level: 'low', color: '#00D4AA' },
  'GPTQ':    { score: '99.0%', loss: '1.0%', text: 'Low',        level: 'low', color: '#00D4AA' },
};

// ─── Data ────────────────────────────────────────────────────────────────────

let MODELS = [];
let GPUS = [];

// ─── Application State ────────────────────────────────────────────────────────

const state = {
  selectedModel:      null,
  selectedQuant:      'Q4_K_M',
  selectedCtxIdx:     2,        // index into CTX_STEPS → 4096
  selectedBackend:    'ollama',
  selectedGPU:        null,
  selectedRamIdx:     3,        // index into RAM_STEPS → 48 GB
  selectedCPU:        'x86',
  selectedSpecsBackend: 'ollama',
  activeTags:         ['chat'],
  activeFamily:       'all',
  sortBy:             'best',
};

// ─── Theme ────────────────────────────────────────────────────────────────────

function initTheme() {
  const saved = localStorage.getItem('theme');
  const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
  const theme = saved || (prefersDark ? 'dark' : 'light');
  document.documentElement.setAttribute('data-theme', theme);
  document.getElementById('theme-toggle').textContent = theme === 'dark' ? '🌙' : '☀️';
}

function toggleTheme() {
  const current = document.documentElement.getAttribute('data-theme');
  const next = current === 'dark' ? 'light' : 'dark';
  document.documentElement.setAttribute('data-theme', next);
  document.getElementById('theme-toggle').textContent = next === 'dark' ? '🌙' : '☀️';
  localStorage.setItem('theme', next);
}

// ─── Page Navigation ──────────────────────────────────────────────────────────

function showLanding() {
  document.getElementById('page-landing').classList.remove('hidden');
  document.getElementById('page-model').classList.add('hidden');
  document.getElementById('page-specs').classList.add('hidden');
  document.querySelectorAll('.mode-tab').forEach(t => t.classList.remove('active'));
}

function showMode(mode) {
  const isModel = mode === 'model';
  document.getElementById('page-landing').classList.add('hidden');
  document.getElementById('page-model').classList.toggle('hidden', !isModel);
  document.getElementById('page-specs').classList.toggle('hidden', isModel);
  document.getElementById('tab-model').classList.toggle('active', isModel);
  document.getElementById('tab-specs').classList.toggle('active', !isModel);
}

// ─── Search Dropdown ──────────────────────────────────────────────────────────

function setupSearch(inputId, dropdownId, items, labelFn, subFn, onSelect) {
  const input    = document.getElementById(inputId);
  const dropdown = document.getElementById(dropdownId);

  function renderDropdown(query) {
    const q = query.toLowerCase();
    const filtered = q
      ? items.filter(item => labelFn(item).toLowerCase().includes(q))
      : items;
    dropdown.innerHTML = filtered.slice(0, 40).map(item => `
      <div class="dropdown-item" onclick="void(0)">
        <span class="dropdown-label">${labelFn(item)}</span>
        <span class="dropdown-sub">${subFn(item)}</span>
      </div>
    `).join('');
    // Attach click handlers (avoids HTML injection via model names)
    dropdown.querySelectorAll('.dropdown-item').forEach((el, i) => {
      el.addEventListener('mousedown', (e) => {
        e.preventDefault();
        onSelect(filtered[i]);
        input.value = labelFn(filtered[i]);
        dropdown.classList.remove('open');
      });
    });
    dropdown.classList.toggle('open', filtered.length > 0);
  }

  input.addEventListener('focus', () => renderDropdown(input.value));
  input.addEventListener('input', () => renderDropdown(input.value));
  input.addEventListener('blur', () => {
    setTimeout(() => dropdown.classList.remove('open'), 150);
  });
}

// ─── Mode A: Model Selection ──────────────────────────────────────────────────

function selectModel(model) {
  state.selectedModel = model;
  document.getElementById('model-param-tag').textContent = model.params_B + 'B';
  buildQuantPicker();
  calculateModelSpecs();
}

function buildQuantPicker() {
  const model = state.selectedModel;
  if (!model) return;
  const quants = model.supported_quants || ['FP16', 'Q8_0', 'Q5_K_M', 'Q4_K_M', 'Q4_0', 'Q3_K_M', 'Q2_K'];
  const picker = document.getElementById('quant-picker');
  picker.innerHTML = quants.map(q => {
    const sel = q === state.selectedQuant;
    const bits = QUANT_BITS[q] || 4;
    const size = calcWeightMemory(model.params_B, q).toFixed(1);
    const desc = QUANT_INFO[q] || '';
    return `
      <div class="quant-row ${sel ? 'selected' : ''}" onclick="selectQuant('${q}')">
        <div class="quant-dot"></div>
        <span class="quant-name">${q}</span>
        <span class="quant-desc">${desc}</span>
        <span class="quant-size">${size} GB</span>
      </div>`;
  }).join('');
  // If current quant not supported, pick first available
  if (!quants.includes(state.selectedQuant)) {
    state.selectedQuant = quants.includes('Q4_K_M') ? 'Q4_K_M' : quants[0];
    buildQuantPicker();
  }
}

function selectQuant(q) {
  state.selectedQuant = q;
  buildQuantPicker();
  calculateModelSpecs();
}

// ─── Mode A: Calculate Model Requirements ─────────────────────────────────────

function calculateModelSpecs() {
  const model = state.selectedModel;
  if (!model) return;

  const quant   = state.selectedQuant;
  const ctxLen  = CTX_STEPS[state.selectedCtxIdx];
  const backend = state.selectedBackend;
  const v       = calcTotalVRAM(model.params_B, model, quant, ctxLen, backend);
  const speedT  = getSpeedTiers(v.weightsGB);
  const disk    = calcDiskSpace(model.params_B, quant);
  const ramNeeded = calcMinRAM(model.params_B, quant);
  const maxCtx    = calcMaxContext(model.params_B, model, quant, v.weightsGB + v.overheadGB + 2, backend);

  const wPct  = v.total > 0 ? (v.weightsGB / v.totalGB * 100).toFixed(1) : 0;
  const kvPct = v.total > 0 ? (v.kvCacheGB / v.totalGB * 100).toFixed(1) : 0;
  const ohPct = v.total > 0 ? (v.overheadGB / v.totalGB * 100).toFixed(1) : 0;

  const quality = QUANT_QUALITY[quant] || { score: '—', loss: '—', text: 'Unknown', color: '#8B8FA8' };

  // GPU compatibility rows
  const gpuRows = [...GPUS]
    .sort((a, b) => b.vram_gb - a.vram_gb)
    .map(gpu => {
      const score = calcFitScore(gpu.vram_gb, v.totalGB);
      const label = getFitLabel(score);
      const spd   = estimateSpeedForGPU(gpu.bandwidth_GBs, v.weightsGB);
      return { gpu, score, label, speed: spd };
    });

  const ollamaCmd = model.ollama_name ? `ollama run ${model.ollama_name}` : null;

  const container = document.getElementById('model-results-content');
  container.innerHTML = `
    <div class="result-card vram-card">
      <div class="vram-header">
        <span class="vram-title">VRAM Required</span>
        <div><span class="vram-number">${v.totalGB.toFixed(1)}</span><span class="vram-unit"> GB</span></div>
      </div>
      <div class="bar-container">
        <div class="bar-seg seg-weights"  style="width:${wPct}%"  title="Weights: ${v.weightsGB.toFixed(1)} GB"></div>
        <div class="bar-seg seg-kv"       style="width:${kvPct}%" title="KV Cache: ${v.kvCacheGB.toFixed(1)} GB"></div>
        <div class="bar-seg seg-overhead" style="width:${ohPct}%" title="Overhead: ${v.overheadGB.toFixed(1)} GB"></div>
      </div>
      <div class="bar-legend">
        <div class="legend-item"><div class="legend-dot" style="background:var(--accent)"></div>weights <span class="legend-val">${v.weightsGB.toFixed(1)} GB</span></div>
        <div class="legend-item"><div class="legend-dot" style="background:var(--warn)"></div>kv-cache <span class="legend-val">${v.kvCacheGB.toFixed(1)} GB</span></div>
        <div class="legend-item"><div class="legend-dot" style="background:var(--text-muted)"></div>overhead <span class="legend-val">${v.overheadGB.toFixed(1)} GB</span></div>
      </div>
    </div>

    <div class="stat-row">
      <div class="stat-card">
        <div class="stat-label">Disk Space</div>
        <div class="stat-value">${disk.toFixed(1)} <span class="stat-unit">GB</span></div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Min System RAM</div>
        <div class="stat-value">${ramNeeded} <span class="stat-unit">GB</span></div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Quality Loss</div>
        <div class="stat-value" style="color:${quality.color}">${quality.text}</div>
        <div class="stat-sub">${quality.loss} degradation</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Context Length</div>
        <div class="stat-value">${ctxLen.toLocaleString()} <span class="stat-unit">tok</span></div>
        <div class="stat-sub">max: ${(model.context_max_k * 1024).toLocaleString()}</div>
      </div>
    </div>

    <div class="result-card">
      <div class="section-label">⚡ Speed Estimates (approximate)</div>
      <div class="speed-grid">
        <div class="speed-card">
          <div class="speed-tier">CPU-only</div>
          <div class="speed-val">${formatSpeedRange(speedT.low.min, speedT.low.max)}</div>
          <div class="speed-sub">tokens/sec · DDR5</div>
        </div>
        <div class="speed-card">
          <div class="speed-tier">Mid GPU</div>
          <div class="speed-val">${formatSpeedRange(speedT.mid.min, speedT.mid.max)}</div>
          <div class="speed-sub">tokens/sec · RTX 3090</div>
        </div>
        <div class="speed-card">
          <div class="speed-tier">High-end</div>
          <div class="speed-val">${formatSpeedRange(speedT.high.min, speedT.high.max)}</div>
          <div class="speed-sub">tokens/sec · 2× A100</div>
        </div>
      </div>
    </div>

    ${ollamaCmd ? `
    <div class="result-card">
      <div class="section-label">📋 Quick Commands</div>
      <div class="cmd-row">
        <code class="cmd-code">${ollamaCmd}</code>
        <button class="cmd-copy" onclick="copyText('${ollamaCmd}')">Copy</button>
      </div>
      ${model.hf_id ? `<div class="cmd-row">
        <code class="cmd-code">huggingface-cli download ${model.hf_id}</code>
        <button class="cmd-copy" onclick="copyText('huggingface-cli download ${model.hf_id}')">Copy</button>
      </div>` : ''}
    </div>` : ''}

    <div class="result-card">
      <div class="section-label">🖥️ GPU Compatibility</div>
      <div class="gpu-compat-list">
        ${gpuRows.map(({ gpu, label, speed }) => `
          <div class="gpu-row">
            <span class="gpu-name">${gpu.display_name}</span>
            <span class="gpu-vram-badge">${gpu.vram_gb}GB</span>
            <span class="fit-badge ${label.cssClass}">${label.label}</span>
            <span class="gpu-speed">~${speed.speed} t/s</span>
          </div>
        `).join('')}
      </div>
    </div>
  `;

  document.getElementById('model-placeholder').classList.add('hidden');
  container.classList.remove('hidden');
}

// ─── Mode B: GPU Selection ────────────────────────────────────────────────────

function selectGPU(gpu) {
  state.selectedGPU = gpu;
  document.getElementById('gpu-vram-pill').textContent = gpu.vram_gb + ' GB VRAM';
}

// ─── Mode B: Calculate Compatible Models ──────────────────────────────────────

function calculateSpecsModels() {
  const gpu = state.selectedGPU;
  if (!gpu) {
    showToast('Please select a GPU first');
    return;
  }

  const ramGB   = RAM_STEPS[state.selectedRamIdx];
  const backend = state.selectedSpecsBackend;
  const ctxLen  = CTX_STEPS[state.selectedCtxIdx] || 4096;
  const tags    = state.activeTags;

  // Filter by task tags if any selected
  let filtered = MODELS;
  if (tags.length > 0) {
    filtered = MODELS.filter(m => tags.some(t => (m.task_tags || []).includes(t)));
  }
  // Filter by family
  if (state.activeFamily !== 'all') {
    filtered = filtered.filter(m => m.family === state.activeFamily);
  }

  // Score each model
  const results = filtered.map(model => {
    const best = findBestQuant(model.params_B, model, gpu.vram_gb, model.supported_quants || ['FP16','Q4_K_M'], ctxLen, backend);
    if (!best) return null;
    const spd = estimateSpeedForGPU(gpu.bandwidth_GBs, best.vram.weightsGB);
    const disk = calcDiskSpace(model.params_B, best.quant);
    return { model, best, speed: spd, disk };
  }).filter(Boolean);

  // Sort
  results.sort((a, b) => {
    if (state.sortBy === 'speed') return b.speed.speed - a.speed.speed;
    if (state.sortBy === 'margin') return b.best.score - a.best.score;
    // Default: best model (size desc, then quant quality desc)
    if (b.model.params_B !== a.model.params_B) return b.model.params_B - a.model.params_B;
    return (QUANT_BITS[b.best.quant] || 4) - (QUANT_BITS[a.best.quant] || 4);
  });

  // Upgrade callout: what if GPU had 8 more GB?
  const nextVram = gpu.vram_gb + 8;
  const moreModels = MODELS.filter(m => {
    if (results.find(r => r.model.id === m.id)) return false;
    const best = findBestQuant(m.params_B, m, nextVram, m.supported_quants || ['FP16','Q4_K_M'], ctxLen, backend);
    return best && best.score >= 1.0;
  });

  const container = document.getElementById('specs-results-content');

  if (results.length === 0) {
    container.innerHTML = `<div class="no-results"><span class="empty-icon">😬</span><div class="empty-title">No models fit this GPU</div><div class="empty-desc">Try a lower context length or different backend.</div></div>`;
  } else {
    // Family filter bar
    const families = ['all', ...new Set(MODELS.map(m => m.family))].sort();
    container.innerHTML = `
      <div class="results-controls">
        <div class="family-filters">
          ${families.map(f => `<button class="tag-btn ${state.activeFamily === f ? 'active' : ''}" onclick="setFamily('${f}')">${f}</button>`).join('')}
        </div>
        <div class="sort-wrap">
          <label class="sort-label">Sort:</label>
          <select class="sort-select" onchange="setSort(this.value)">
            <option value="best" ${state.sortBy === 'best' ? 'selected' : ''}>Best Model</option>
            <option value="speed" ${state.sortBy === 'speed' ? 'selected' : ''}>High Speed</option>
            <option value="margin" ${state.sortBy === 'margin' ? 'selected' : ''}>VRAM Margin</option>
          </select>
        </div>
      </div>

      <div class="results-count">${results.length} model${results.length !== 1 ? 's' : ''} fit your rig</div>

      ${results.map(({ model, best, speed, disk }) => `
        <div class="model-card" onclick="openDrawer('${model.id}')">
          <div class="model-card-header">
            <div>
              <div class="model-card-name">${model.display_name}</div>
              <div class="model-card-meta">${model.params_B}B · ${model.family} · ${model.architecture}</div>
            </div>
            <span class="fit-badge ${best.label.cssClass}">${best.label.label}</span>
          </div>
          <div class="model-card-stats">
            <span class="model-stat">🔢 ${best.quant}</span>
            <span class="model-stat">💾 ${disk.toFixed(1)} GB disk</span>
            <span class="model-stat">⚡ ~${speed.speed} t/s</span>
            <span class="model-stat">📊 ${Math.round(best.score * 100)}% fit</span>
          </div>
          ${model.ollama_name ? `
          <div class="cmd-row" onclick="event.stopPropagation()">
            <code class="cmd-code">ollama run ${model.ollama_name}</code>
            <button class="cmd-copy" onclick="copyText('ollama run ${model.ollama_name}')">Copy</button>
          </div>` : ''}
        </div>
      `).join('')}

      ${moreModels.length > 0 ? `
        <div class="upgrade-callout">
          <span class="callout-icon">💡</span>
          Adding 8 GB VRAM (→ ${nextVram} GB) would unlock <strong>${moreModels.length} more model${moreModels.length !== 1 ? 's' : ''}</strong>
          including ${moreModels.slice(0, 3).map(m => m.display_name).join(', ')}${moreModels.length > 3 ? ` and ${moreModels.length - 3} more` : ''}.
        </div>
      ` : ''}
    `;
  }

  document.getElementById('specs-placeholder').classList.add('hidden');
  container.classList.remove('hidden');
}

// ─── Sort & Filter Controls ───────────────────────────────────────────────────

function setFamily(family) {
  state.activeFamily = family;
  calculateSpecsModels();
}

function setSort(value) {
  state.sortBy = value;
  calculateSpecsModels();
}

// ─── Compatibility Drawer ─────────────────────────────────────────────────────

function openDrawer(modelId) {
  const model = MODELS.find(m => m.id === modelId);
  if (!model) return;

  const gpu     = state.selectedGPU;
  const ctxLen  = CTX_STEPS[state.selectedCtxIdx] || 4096;
  const backend = state.selectedSpecsBackend || state.selectedBackend;

  const allQuants = Object.keys(QUANT_BITS);
  const supported = model.supported_quants || ['FP16', 'Q4_K_M'];

  const quantTable = allQuants.filter(q => supported.includes(q)).map(q => {
    const v     = calcTotalVRAM(model.params_B, model, q, ctxLen, backend);
    const score = gpu ? calcFitScore(gpu.vram_gb, v.totalGB) : null;
    const label = score !== null ? getFitLabel(score) : null;
    const disk  = calcDiskSpace(model.params_B, q).toFixed(1);
    return { q, v, score, label, disk };
  });

  const gpuSpeed = gpu
    ? estimateSpeedForGPU(gpu.bandwidth_GBs, calcWeightMemory(model.params_B, supported.includes('Q4_K_M') ? 'Q4_K_M' : supported[0]))
    : null;

  document.getElementById('drawer-content').innerHTML = `
    <div class="drawer-model-name">${model.display_name}</div>
    <div class="drawer-model-meta">
      ${model.params_B}B params · ${model.architecture} · ${model.family} · ${model.year || ''}
      ${model.license ? `· <span class="license-tag">${model.license}</span>` : ''}
    </div>

    ${gpu ? `<div class="drawer-section">
      <div class="drawer-section-title">on ${gpu.display_name}</div>
      ${gpuSpeed ? `<div class="stat-row"><div class="stat-card"><div class="stat-label">Est. Speed</div><div class="stat-value">~${gpuSpeed.speed} <span class="stat-unit">t/s</span></div></div></div>` : ''}
    </div>` : ''}

    <div class="drawer-section">
      <div class="drawer-section-title">Quantization Compatibility${gpu ? ` on ${gpu.vram_gb}GB VRAM` : ''}</div>
      <table class="compat-table">
        <thead><tr><th>Format</th><th>Size</th>${gpu ? '<th>Fits?</th>' : ''}<th>Quality</th></tr></thead>
        <tbody>
          ${quantTable.map(({ q, v, label, disk }) => `
            <tr>
              <td><strong>${q}</strong></td>
              <td>${disk} GB</td>
              ${gpu ? `<td><span class="fit-badge ${label ? label.cssClass : 'fit-fail'}">${label ? label.label : "WON'T FIT"}</span></td>` : ''}
              <td style="color:${QUANT_QUALITY[q]?.color || '#8B8FA8'}">${QUANT_QUALITY[q]?.text || '—'}</td>
            </tr>
          `).join('')}
        </tbody>
      </table>
    </div>

    <div class="drawer-section">
      <div class="drawer-section-title">Task Tags</div>
      <div class="tag-row">${(model.task_tags || []).map(t => `<span class="tag-chip">${t}</span>`).join('')}</div>
    </div>

    <div class="drawer-section">
      <div class="drawer-section-title">Links & Commands</div>
      ${model.ollama_name ? `
        <div class="cmd-row">
          <code class="cmd-code">ollama run ${model.ollama_name}</code>
          <button class="cmd-copy" onclick="copyText('ollama run ${model.ollama_name}')">Copy</button>
        </div>` : ''}
      ${model.hf_id ? `
        <div class="cmd-row">
          <code class="cmd-code">huggingface-cli download ${model.hf_id}</code>
          <button class="cmd-copy" onclick="copyText('huggingface-cli download ${model.hf_id}')">Copy</button>
        </div>` : ''}
      <div class="drawer-links">
        ${model.ollama_name ? `<a class="drawer-link" href="https://ollama.com/library/${model.ollama_name.split(':')[0]}" target="_blank" rel="noopener">🦙 Ollama</a>` : ''}
        ${model.hf_id ? `<a class="drawer-link" href="https://huggingface.co/${model.hf_id}" target="_blank" rel="noopener">🤗 HuggingFace</a>` : ''}
      </div>
    </div>
  `;

  document.getElementById('drawer-overlay').classList.add('open');
  document.getElementById('drawer').classList.add('open');
}

function closeDrawer() {
  document.getElementById('drawer').classList.remove('open');
  document.getElementById('drawer-overlay').classList.remove('open');
}

// ─── Utilities ────────────────────────────────────────────────────────────────

function copyText(text) {
  navigator.clipboard.writeText(text).then(() => showToast('Copied!')).catch(() => {
    // Fallback for non-HTTPS
    const el = document.createElement('textarea');
    el.value = text;
    document.body.appendChild(el);
    el.select();
    document.execCommand('copy');
    document.body.removeChild(el);
    showToast('Copied!');
  });
}

function showToast(msg) {
  const toast = document.getElementById('toast');
  toast.textContent = msg;
  toast.classList.add('show');
  setTimeout(() => toast.classList.remove('show'), 2000);
}

// ─── Initialization ────────────────────────────────────────────────────────────

function init() {
  initTheme();

  // Context slider
  const ctxSlider = document.getElementById('ctx-slider');
  const ctxVal    = document.getElementById('ctx-val');
  ctxSlider.addEventListener('input', () => {
    state.selectedCtxIdx = parseInt(ctxSlider.value);
    ctxVal.textContent = CTX_STEPS[state.selectedCtxIdx].toLocaleString() + ' tokens';
    if (state.selectedModel) calculateModelSpecs();
  });

  // Backend select (Mode A)
  document.getElementById('backend-select').addEventListener('change', (e) => {
    state.selectedBackend = e.target.value;
    if (state.selectedModel) calculateModelSpecs();
  });

  // RAM slider (Mode B)
  const ramSlider = document.getElementById('ram-slider');
  const ramVal    = document.getElementById('ram-val');
  ramSlider.addEventListener('input', () => {
    state.selectedRamIdx = parseInt(ramSlider.value);
    ramVal.textContent = RAM_STEPS[state.selectedRamIdx] + ' GB';
  });

  // Backend select (Mode B)
  document.getElementById('specs-backend-select').addEventListener('change', (e) => {
    state.selectedSpecsBackend = e.target.value;
  });

  // CPU select
  document.getElementById('cpu-select').addEventListener('change', (e) => {
    state.selectedCPU = e.target.value;
  });

  // Task tags
  document.querySelectorAll('#task-tags .tag-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const tag = btn.dataset.tag;
      btn.classList.toggle('active');
      if (state.activeTags.includes(tag)) {
        state.activeTags = state.activeTags.filter(t => t !== tag);
      } else {
        state.activeTags.push(tag);
      }
    });
  });

  // Escape key closes drawer
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') closeDrawer();
  });

  // Model search
  setupSearch('model-search', 'model-dropdown', MODELS,
    m => m.display_name,
    m => m.params_B + 'B · ' + m.family,
    selectModel
  );

  // GPU search
  setupSearch('gpu-search', 'gpu-dropdown', GPUS,
    g => g.display_name,
    g => g.vram_gb + ' GB VRAM',
    selectGPU
  );
}

async function loadData() {
  try {
    const [modelsRes, gpusRes] = await Promise.all([
      fetch('./src/data/models.json'),
      fetch('./src/data/gpus.json'),
    ]);
    MODELS = await modelsRes.json();
    GPUS   = await gpusRes.json();
    // Sort GPUs: descending VRAM
    GPUS.sort((a, b) => b.vram_gb - a.vram_gb);
    init();
  } catch (err) {
    console.error('Failed to load data:', err);
    document.body.innerHTML = `<div style="padding:40px;color:#EF4444;font-family:monospace">
      Error loading data: ${err.message}<br><br>
      Run from a local server: <code>python -m http.server 8080</code>
    </div>`;
  }
}

// Boot
loadData();

// ─── Global Bindings for Inline Event Handlers ────────────────────────────────

window.showLanding         = showLanding;
window.showMode            = showMode;
window.toggleTheme         = toggleTheme;
window.calculateModelSpecs = calculateModelSpecs;
window.calculateSpecsModels= calculateSpecsModels;
window.closeDrawer         = closeDrawer;
window.openDrawer          = openDrawer;
window.selectQuant         = selectQuant;
window.setFamily           = setFamily;
window.setSort             = setSort;
window.copyText            = copyText;
window.selectModel         = selectModel;
window.selectGPU           = selectGPU;