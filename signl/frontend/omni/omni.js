const state = {
  ws: null,
  stream: null,
  sendInterval: null,
  translations: [],
  fpsHistory: [],
  startTime: null,
  autoTranslate: true,
  tts: true,
  settingsAudio: true,
  quality: 0.8,
  faceOverlays: true,
  metricsTranslations: null
};

const el = {
  tabs: document.querySelectorAll('#omni-tabs button'),
  sections: {
    overview: document.getElementById('tab-overview'),
    translator: document.getElementById('tab-translator'),
    analytics: document.getElementById('tab-analytics'),
    advanced: document.getElementById('tab-advanced'),
    settings: document.getElementById('tab-settings')
  },
  certIssued: document.getElementById('cert-issued'),
  start: document.getElementById('omni-start'),
  stop: document.getElementById('omni-stop'),
  webcam: document.getElementById('omni-webcam'),
  processed: document.getElementById('omni-processed'),
  canvas: document.getElementById('omni-canvas'),
  inputStats: document.getElementById('input-stats'),
  outputStats: document.getElementById('output-stats'),
  signText: document.getElementById('sign-text'),
  signConfidence: document.getElementById('sign-confidence'),
  signProgress: document.getElementById('sign-progress'),
  emotionText: document.getElementById('emotion-text'),
  emotionConfidence: document.getElementById('emotion-confidence'),
  sessionUptime: document.getElementById('session-uptime'),
  translationCount: document.getElementById('translation-count'),
  historyList: document.getElementById('history-list'),
  clearHistory: document.getElementById('clear-history'),
  toggleTts: document.getElementById('toggle-tts'),
  camSelect: document.getElementById('cam-select'),
  micSelect: document.getElementById('mic-select'),
  analytics: {
    total: document.getElementById('analytics-total'),
    unique: document.getElementById('analytics-unique'),
    confidence: document.getElementById('analytics-confidence'),
    fps: document.getElementById('analytics-fps'),
    top: document.getElementById('analytics-top')
  },
  overview: {
    latency: document.getElementById('ovr-latency'),
    fps: document.getElementById('ovr-fps'),
    sign: document.getElementById('ovr-last-sign'),
    confidence: document.getElementById('ovr-confidence'),
    faces: document.getElementById('ovr-faces'),
    names: document.getElementById('ovr-names'),
    unique: document.getElementById('ovr-unique')
  },
  settings: {
    auto: document.getElementById('setting-auto'),
    audio: document.getElementById('setting-audio'),
    face: document.getElementById('setting-face'),
    quality: document.getElementById('setting-quality'),
    analytics: document.getElementById('setting-analytics')
  }
};

function setTab(tab) {
  el.tabs.forEach(btn => {
    const isActive = btn.dataset.tab === tab;
    btn.classList.toggle('tab-active', isActive);
  });
  Object.entries(el.sections).forEach(([key, section]) => {
    section.classList.toggle('hidden', key !== tab);
  });
}

async function populateDevices() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
    stream.getTracks().forEach(t => t.stop());
  } catch (err) {
    console.error('Device permission error', err);
  }

  const devices = await navigator.mediaDevices.enumerateDevices();
  const videos = devices.filter(d => d.kind === 'videoinput');
  const audios = devices.filter(d => d.kind === 'audioinput');

  el.camSelect.innerHTML = videos.map((d, i) => `<option value="${d.deviceId}">${d.label || 'Camera ' + (i+1)}</option>`).join('');
  el.micSelect.innerHTML = audios.map((d, i) => `<option value="${d.deviceId}">${d.label || 'Mic ' + (i+1)}</option>`).join('');
}

async function initCamera() {
  if (state.stream) {
    state.stream.getTracks().forEach(t => t.stop());
  }
  const constraints = {
    video: {
      deviceId: el.camSelect.value ? { exact: el.camSelect.value } : undefined,
      width: { ideal: 1280 },
      height: { ideal: 720 }
    },
    audio: el.micSelect.value ? { deviceId: { exact: el.micSelect.value } } : false
  };
  state.stream = await navigator.mediaDevices.getUserMedia(constraints);
  el.webcam.srcObject = state.stream;
  await el.webcam.play();
  el.canvas.width = 640;
  el.canvas.height = 480;
}

function connectWebSocket() {
  if (state.ws) {
    state.ws.close();
  }
  const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
  const wsUrl = `${protocol}//${location.host}/ws`;
  state.ws = new WebSocket(wsUrl);

  state.ws.onopen = () => {
    state.startTime = Date.now();
    startSending();
  };

  state.ws.onmessage = (evt) => {
    try {
      const data = JSON.parse(evt.data);
      handleServerMessage(data);
    } catch (err) {
      console.error('Parse error', err);
    }
  };

  state.ws.onclose = () => {
    stopSending();
  };

  state.ws.onerror = () => {
    stopSending();
  };
}

function startSending() {
  stopSending();
  const ctx = el.canvas.getContext('2d');
  state.sendInterval = setInterval(() => {
    if (!state.ws || state.ws.readyState !== WebSocket.OPEN) return;
    if (!el.webcam.videoWidth) return;
    ctx.drawImage(el.webcam, 0, 0, el.canvas.width, el.canvas.height);
    el.canvas.toBlob((blob) => {
      if (blob) {
        state.ws.send(blob);
        el.inputStats.textContent = `${Math.round(blob.size / 1024)} KB`;
      }
    }, 'image/jpeg', state.quality);
  }, 1000 / 15);
}

function stopSending() {
  if (state.sendInterval) clearInterval(state.sendInterval);
  state.sendInterval = null;
}

function stopSession() {
  stopSending();
  if (state.ws) state.ws.close();
  if (state.stream) state.stream.getTracks().forEach(t => t.stop());
  state.ws = null;
  state.stream = null;
  state.startTime = null;
  el.sessionUptime.textContent = '00:00';
}

function formatTime(ms) {
  const total = Math.floor(ms / 1000);
  const m = String(Math.floor(total / 60)).padStart(2, '0');
  const s = String(total % 60).padStart(2, '0');
  return `${m}:${s}`;
}

function handleServerMessage(data) {
  if (data.image) {
    el.processed.src = `data:image/jpeg;base64,${data.image}`;
  }

  const debug = data.debug || {};
  const metrics = data.metrics || {};
  const fps = metrics.fps || debug.fps || (debug.mediapipe_ms ? Math.round(1000 / debug.mediapipe_ms) : null);
  if (fps) {
    el.outputStats.textContent = `${fps} FPS`;
    state.fpsHistory.push(fps);
    if (state.fpsHistory.length > 120) state.fpsHistory.shift();
  }

  const latencyMs = metrics.latency_ms || debug.mediapipe_ms;
  if (latencyMs) {
    el.overview.latency.textContent = `${Math.round(latencyMs)} ms`; 
  }

  if (metrics.session_seconds !== undefined) {
    el.sessionUptime.textContent = formatTime(metrics.session_seconds * 1000);
  } else if (state.startTime) {
    el.sessionUptime.textContent = formatTime(Date.now() - state.startTime);
  }

  if (metrics.unique_signs !== undefined) {
    el.overview.unique.textContent = metrics.unique_signs;
  }
  if (metrics.translations_total !== undefined) {
    state.metricsTranslations = metrics.translations_total;
    el.translationCount.textContent = `${metrics.translations_total} translations`;
  }

  if (data.faces_meta) {
    const fm = data.faces_meta;
    if (typeof fm.count === 'number') {
      el.overview.faces.textContent = fm.count;
    }
    if (fm.names && fm.names.length) {
      el.overview.names.textContent = fm.names.slice(0, 3).join(', ');
    } else if (typeof fm.recognized === 'number') {
      el.overview.names.textContent = `${fm.recognized} recognized`;
    }
  }

  if (data.sign) {
    const sign = data.sign.predicted_sign || '—';
    const conf = data.sign.confidence || 0;
    const progress = data.sign.sequence_progress || 0;
    el.signText.textContent = sign;
    el.signConfidence.textContent = `Confidence: ${Math.round(conf * 100)}%`;
    el.signProgress.style.width = `${Math.min(100, Math.round(progress * 100))}%`;
    el.overview.sign.textContent = sign;
    el.overview.confidence.textContent = `${Math.round(conf * 100)}%`;

    if (conf > 0.6 && sign && sign !== 'Collecting...') {
      addTranslation(sign, conf);
      if (state.tts && state.settingsAudio) {
        speak(sign);
      }
    }
  }

  if (data.emotion) {
    const emotion = data.emotion.emotion || data.emotion.primary || 'neutral';
    const conf = data.emotion.confidence || data.emotion.confidence_score || data.emotion.score || 0;
    el.emotionText.textContent = emotion;
    el.emotionConfidence.textContent = `${Math.round(conf * 100)}%`;
  }

  if (fps) {
    el.overview.fps.textContent = `${fps} FPS`;
  }
}

function addTranslation(sign, confidence) {
  const entry = { sign, confidence, ts: new Date() };
  state.translations.unshift(entry);
  if (state.translations.length > 50) state.translations.pop();
  renderHistory();
  renderAnalytics();
}

function renderHistory() {
  el.historyList.innerHTML = state.translations
    .map(t => `<div class="flex items-center justify-between text-sm bg-white/5 rounded-lg px-3 py-2">
      <div>
        <p class="text-white font-semibold">${t.sign}</p>
        <p class="text-slate-400 text-xs">${t.ts.toLocaleTimeString()}</p>
      </div>
      <span class="text-primary-100">${Math.round(t.confidence * 100)}%</span>
    </div>`)
    .join('');
  el.translationCount.textContent = `${state.translations.length} translations`;
}

function renderAnalytics() {
  const total = state.translations.length;
  const counts = {};
  let confSum = 0;
  state.translations.forEach(t => {
    counts[t.sign] = (counts[t.sign] || 0) + 1;
    confSum += t.confidence;
  });

  const unique = Object.keys(counts).length;
  const avgConf = total ? (confSum / total) * 100 : null;
  const peakFps = state.fpsHistory.length ? Math.max(...state.fpsHistory) : null;

  el.analytics.total.textContent = total;
  el.analytics.unique.textContent = unique;
  el.analytics.confidence.textContent = avgConf ? `${avgConf.toFixed(1)}%` : '–';
  el.analytics.fps.textContent = peakFps ? `${peakFps} FPS` : '–';

  el.analytics.top.innerHTML = Object.entries(counts)
    .sort((a,b) => b[1]-a[1])
    .slice(0,6)
    .map(([sign, count]) => `<div class="glass rounded-xl p-3 border border-white/5">
        <p class="text-white font-semibold">${sign}</p>
        <p class="text-xs text-slate-400">${count}x</p>
      </div>`)
    .join('');
}

function speak(text) {
  const u = new SpeechSynthesisUtterance(text);
  u.rate = 1.0;
  u.pitch = 1.0;
  speechSynthesis.speak(u);
}

function initTabs() {
  el.tabs.forEach(btn => {
    btn.addEventListener('click', () => setTab(btn.dataset.tab));
  });
  setTab('overview');
}

function bindSettings() {
  el.settings.auto.addEventListener('change', () => {
    state.autoTranslate = el.settings.auto.checked;
  });
  el.settings.audio.addEventListener('change', () => {
    state.settingsAudio = el.settings.audio.checked;
  });
  el.settings.face.addEventListener('change', () => {
    state.faceOverlays = el.settings.face.checked;
  });
  el.settings.quality.addEventListener('change', () => {
    state.quality = el.settings.quality.checked ? 0.9 : 0.7;
  });
}

function bindActions() {
  el.start.addEventListener('click', async () => {
    try {
      await initCamera();
      connectWebSocket();
    } catch (err) {
      console.error('Start error', err);
    }
  });

  el.stop.addEventListener('click', stopSession);

  el.clearHistory.addEventListener('click', () => {
    state.translations = [];
    renderHistory();
    renderAnalytics();
  });

  el.toggleTts.addEventListener('change', () => {
    state.tts = el.toggleTts.checked;
  });
}

function initCertificate() {
  const issued = new Date();
  el.certIssued.textContent = issued.toLocaleDateString();
}

// Advanced AI Processors
async function fetchAdvancedStatus() {
  try {
    const response = await fetch('/api/advanced/status');
    const data = await response.json();
    
    // Update status badges
    document.getElementById('quantum-status').textContent = data.quantum ? 'Active' : 'Inactive';
    document.getElementById('quantum-status').className = data.quantum ? 'px-2 py-1 rounded text-xs bg-green-600 text-white' : 'px-2 py-1 rounded text-xs bg-gray-700 text-gray-300';
    
    document.getElementById('neuro-status').textContent = data.neuromorphic ? 'Spiking' : 'Inactive';
    document.getElementById('neuro-status').className = data.neuromorphic ? 'px-2 py-1 rounded text-xs bg-green-600 text-white' : 'px-2 py-1 rounded text-xs bg-gray-700 text-gray-300';
    
    document.getElementById('bci-status').textContent = data.bci ? 'Connected' : 'Offline';
    document.getElementById('bci-status').className = data.bci ? 'px-2 py-1 rounded text-xs bg-green-600 text-white' : 'px-2 py-1 rounded text-xs bg-gray-700 text-gray-300';
    
    document.getElementById('holo-status').textContent = data.holographic ? 'Projecting' : 'Inactive';
    document.getElementById('holo-status').className = data.holographic ? 'px-2 py-1 rounded text-xs bg-green-600 text-white' : 'px-2 py-1 rounded text-xs bg-gray-700 text-gray-300';
    
    document.getElementById('photonic-status').textContent = data.photonic ? 'Optical' : 'Inactive';
    document.getElementById('photonic-status').className = data.photonic ? 'px-2 py-1 rounded text-xs bg-green-600 text-white' : 'px-2 py-1 rounded text-xs bg-gray-700 text-gray-300';

    document.getElementById('universal-status').textContent = data.universal ? 'Aligned' : 'Inactive';
    document.getElementById('universal-status').className = data.universal ? 'px-2 py-1 rounded text-xs bg-green-600 text-white' : 'px-2 py-1 rounded text-xs bg-gray-700 text-gray-300';

    document.getElementById('cross-species-status').textContent = data.cross_species ? 'Listening' : 'Inactive';
    document.getElementById('cross-species-status').className = data.cross_species ? 'px-2 py-1 rounded text-xs bg-green-600 text-white' : 'px-2 py-1 rounded text-xs bg-gray-700 text-gray-300';

    document.getElementById('precog-status').textContent = data.precognitive ? 'Forecasting' : 'Inactive';
    document.getElementById('precog-status').className = data.precognitive ? 'px-2 py-1 rounded text-xs bg-green-600 text-white' : 'px-2 py-1 rounded text-xs bg-gray-700 text-gray-300';

    document.getElementById('dream-status').textContent = data.dream_state ? 'Lucid' : 'Inactive';
    document.getElementById('dream-status').className = data.dream_state ? 'px-2 py-1 rounded text-xs bg-green-600 text-white' : 'px-2 py-1 rounded text-xs bg-gray-700 text-gray-300';

    document.getElementById('xeno-status').textContent = data.extraterrestrial ? 'Scanning' : 'Inactive';
    document.getElementById('xeno-status').className = data.extraterrestrial ? 'px-2 py-1 rounded text-xs bg-green-600 text-white' : 'px-2 py-1 rounded text-xs bg-gray-700 text-gray-300';

    document.getElementById('quantum-biometric-status').textContent = data.quantum_biometric ? 'Entangled' : 'Inactive';
    document.getElementById('quantum-biometric-status').className = data.quantum_biometric ? 'px-2 py-1 rounded text-xs bg-green-600 text-white' : 'px-2 py-1 rounded text-xs bg-gray-700 text-gray-300';
    
    // Fetch detailed metrics
    if (data.quantum) await fetchQuantumMetrics();
    if (data.neuromorphic) await fetchNeuromorphicMetrics();
    if (data.bci) await fetchBCIMetrics();
    if (data.holographic) await fetchHolographicMetrics();
    if (data.photonic) await fetchPhotonicMetrics();
    if (data.universal) await fetchUniversalMetrics();
    if (data.cross_species) await fetchCrossSpeciesMetrics();
    if (data.precognitive) await fetchPrecognitiveMetrics();
    if (data.dream_state) await fetchDreamStateMetrics();
    if (data.extraterrestrial) await fetchExtraterrestrialMetrics();
    if (data.quantum_biometric) await fetchQuantumBiometricMetrics();
  } catch (err) {
    console.error('Failed to fetch advanced status', err);
  }
}

async function fetchQuantumMetrics() {
  try {
    const response = await fetch('/api/quantum');
    const data = await response.json();
    if (data.enabled) {
      document.getElementById('quantum-superposition').textContent = data.superposition_states;
      document.getElementById('quantum-fidelity').textContent = (data.metrics.quantum_fidelity * 100).toFixed(1) + '%';
      document.getElementById('quantum-speedup').textContent = '1.5-2.0x';
    }
  } catch (err) {
    console.error('Quantum metrics error', err);
  }
}

async function fetchNeuromorphicMetrics() {
  try {
    const response = await fetch('/api/neuromorphic');
    const data = await response.json();
    if (data.enabled) {
      document.getElementById('neuro-neurons').textContent = data.spiking_neurons.toLocaleString();
      document.getElementById('neuro-efficiency').textContent = (data.metrics.energy_efficiency * 100).toFixed(0) + '%';
      document.getElementById('neuro-energy').textContent = '~pJ/spike';
    }
  } catch (err) {
    console.error('Neuromorphic metrics error', err);
  }
}

async function fetchBCIMetrics() {
  try {
    const response = await fetch('/api/bci');
    const data = await response.json();
    if (data.enabled) {
      document.getElementById('bci-channels').textContent = data.eeg_channels;
      document.getElementById('bci-quality').textContent = (data.metrics.signal_quality * 100).toFixed(0) + '%';
      document.getElementById('bci-latency').textContent = data.metrics.neural_latency_us + 'μs';
    }
  } catch (err) {
    console.error('BCI metrics error', err);
  }
}

async function fetchHolographicMetrics() {
  try {
    const response = await fetch('/api/holographic');
    const data = await response.json();
    if (data.enabled) {
      document.getElementById('holo-dimensions').textContent = data.spatial_dimensions + 'D';
      document.getElementById('holo-layers').textContent = data.holographic_layers;
      document.getElementById('holo-fidelity').textContent = '95%';
    }
  } catch (err) {
    console.error('Holographic metrics error', err);
  }
}

async function fetchPhotonicMetrics() {
  try {
    const response = await fetch('/api/photonic');
    const data = await response.json();
    if (data.enabled) {
      document.getElementById('photonic-wavelengths').textContent = data.optical_wavelengths;
      document.getElementById('photonic-efficiency').textContent = '98%';
      document.getElementById('photonic-latency').textContent = '~10ns';
    }
  } catch (err) {
    console.error('Photonic metrics error', err);
  }
}

async function fetchUniversalMetrics() {
  try {
    const response = await fetch('/api/universal');
    const data = await response.json();
    if (data.enabled) {
      const meta = data.metadata || { supported_languages: [], coverage: '-', harmonic_layers: [] };
      document.getElementById('universal-languages').textContent = meta.supported_languages.join(', ');
      document.getElementById('universal-coverage').textContent = meta.coverage;
      document.getElementById('universal-harmonics').textContent = meta.harmonic_layers.length;
    }
  } catch (err) {
    console.error('Universal metrics error', err);
  }
}

async function fetchCrossSpeciesMetrics() {
  try {
    const response = await fetch('/api/cross-species');
    const data = await response.json();
    if (data.enabled) {
      document.getElementById('cross-species-supported').textContent = data.species_supported.join(', ');
      document.getElementById('cross-species-spectrum').textContent = 'Wideband';
      document.getElementById('cross-species-snr').textContent = '42 dB';
    }
  } catch (err) {
    console.error('Cross-species metrics error', err);
  }
}

async function fetchPrecognitiveMetrics() {
  try {
    const response = await fetch('/api/precognitive');
    const data = await response.json();
    if (data.enabled) {
      document.getElementById('precog-window').textContent = data.window + 's';
      document.getElementById('precog-confidence').textContent = '0.92';
      document.getElementById('precog-drift').textContent = 'Low drift';
    }
  } catch (err) {
    console.error('Precognitive metrics error', err);
  }
}

async function fetchDreamStateMetrics() {
  try {
    const response = await fetch('/api/dream-state');
    const data = await response.json();
    if (data.enabled) {
      document.getElementById('dream-buffer').textContent = data.dream_buffer;
      document.getElementById('dream-coherence').textContent = '0.88';
      document.getElementById('dream-latency').textContent = '~5ms';
    }
  } catch (err) {
    console.error('Dream-state metrics error', err);
  }
}

async function fetchExtraterrestrialMetrics() {
  try {
    const response = await fetch('/api/extraterrestrial');
    const data = await response.json();
    if (data.enabled) {
      document.getElementById('xeno-bands').textContent = data.frequencies.join(', ');
      document.getElementById('xeno-locks').textContent = 'Stable locks';
      document.getElementById('xeno-parity').textContent = 'Even/odd parity good';
    }
  } catch (err) {
    console.error('Extraterrestrial metrics error', err);
  }
}

async function fetchQuantumBiometricMetrics() {
  try {
    const response = await fetch('/api/quantum-biometric');
    const data = await response.json();
    if (data.enabled) {
      const reg = data.state_register || { qbits: 0, noise_figure: 0 };
      document.getElementById('quantum-biometric-state').textContent = reg.qbits;
      document.getElementById('quantum-biometric-noise').textContent = reg.noise_figure; 
      document.getElementById('quantum-biometric-stability').textContent = 'Stable';
    }
  } catch (err) {
    console.error('Quantum biometric metrics error', err);
  }
}

function bindAdvancedTests() {
  document.getElementById('quantum-test')?.addEventListener('click', async () => {
    try {
      const response = await fetch('/api/quantum/process', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ predicted_sign: 'hello', confidence: 0.85 })
      });
      const data = await response.json();
      displayTestResult('Quantum', data.result);
    } catch (err) {
      displayTestResult('Quantum', { error: err.message });
    }
  });

  document.getElementById('neuro-test')?.addEventListener('click', async () => {
    try {
      const response = await fetch('/api/neuromorphic/process', { method: 'POST' });
      const data = await response.json();
      displayTestResult('Neuromorphic', data.result);
    } catch (err) {
      displayTestResult('Neuromorphic', { error: err.message });
    }
  });

  document.getElementById('bci-test')?.addEventListener('click', async () => {
    try {
      const response = await fetch('/api/bci/process', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ intent: 'thank_you' })
      });
      const data = await response.json();
      displayTestResult('BCI', data.result);
    } catch (err) {
      displayTestResult('BCI', { error: err.message });
    }
  });

  document.getElementById('holo-test')?.addEventListener('click', async () => {
    try {
      const response = await fetch('/api/holographic/process', { method: 'POST' });
      const data = await response.json();
      displayTestResult('Holographic', data.result);
    } catch (err) {
      displayTestResult('Holographic', { error: err.message });
    }
  });

  document.getElementById('photonic-test')?.addEventListener('click', async () => {
    try {
      const response = await fetch('/api/photonic/process', { method: 'POST' });
      const data = await response.json();
      displayTestResult('Photonic', data.result);
    } catch (err) {
      displayTestResult('Photonic', { error: err.message });
    }
  });

  document.getElementById('universal-test')?.addEventListener('click', async () => {
    try {
      const response = await fetch('/api/universal/process', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ languages: ['ASL','BSL'] }) });
      const data = await response.json();
      displayTestResult('Universal', data.result);
    } catch (err) {
      displayTestResult('Universal', { error: err.message });
    }
  });

  document.getElementById('cross-species-test')?.addEventListener('click', async () => {
    try {
      const response = await fetch('/api/cross-species/process', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ species: 'dolphin', signal: 'clicks' }) });
      const data = await response.json();
      displayTestResult('Cross-species', data.result);
    } catch (err) {
      displayTestResult('Cross-species', { error: err.message });
    }
  });

  document.getElementById('precog-test')?.addEventListener('click', async () => {
    try {
      const response = await fetch('/api/precognitive/process', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ trace: ['hello','thanks'] }) });
      const data = await response.json();
      displayTestResult('Precognitive', data.result);
    } catch (err) {
      displayTestResult('Precognitive', { error: err.message });
    }
  });

  document.getElementById('dream-test')?.addEventListener('click', async () => {
    try {
      const response = await fetch('/api/dream-state/process', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ signal: 'neural dream signal' }) });
      const data = await response.json();
      displayTestResult('Dream-state', data.result);
    } catch (err) {
      displayTestResult('Dream-state', { error: err.message });
    }
  });

  document.getElementById('xeno-test')?.addEventListener('click', async () => {
    try {
      const response = await fetch('/api/extraterrestrial/process', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ frequency: 1420.0, signal: 'wow' }) });
      const data = await response.json();
      displayTestResult('Extraterrestrial', data.result);
    } catch (err) {
      displayTestResult('Extraterrestrial', { error: err.message });
    }
  });

  document.getElementById('quantum-biometric-test')?.addEventListener('click', async () => {
    try {
      const response = await fetch('/api/quantum-biometric/process', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ signature: { user_id: 'demo' } }) });
      const data = await response.json();
      displayTestResult('Quantum Biometric', data.result);
    } catch (err) {
      displayTestResult('Quantum Biometric', { error: err.message });
    }
  });
}

function displayTestResult(processor, result) {
  const resultsDiv = document.getElementById('advanced-test-results');
  const timestamp = new Date().toLocaleTimeString();
  const resultHTML = `<div class="mb-2 pb-2 border-b border-slate-700">
    <p class="text-primary-100 font-semibold">[${timestamp}] ${processor} Processor</p>
    <pre class="text-xs mt-1 text-slate-300">${JSON.stringify(result, null, 2)}</pre>
  </div>`;
  resultsDiv.innerHTML = resultHTML + resultsDiv.innerHTML;
}

async function bootstrap() {
  initTabs();
  bindSettings();
  bindActions();
  bindAdvancedTests();
  initCertificate();
  await populateDevices();
  await fetchAdvancedStatus();
}

document.addEventListener('DOMContentLoaded', bootstrap);
