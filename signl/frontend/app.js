// SignL - Real-Time Sign Language Translation
// Frontend JavaScript

// DOM Elements
const video = document.getElementById('webcam');
const captureCanvas = document.getElementById('capture-canvas');
const captureContext = captureCanvas.getContext('2d');
const processedStream = document.getElementById('processed-stream');
const cameraSelect = document.getElementById('camera-select');
const switchBtn = document.getElementById('switch-camera-btn');
const testBtn = document.getElementById('test-connection-btn');
const toggleGenderBtn = document.getElementById('toggle-gender-btn');
const inputStats = document.getElementById('input-stats');
const outputStats = document.getElementById('output-stats');
const statusIndicator = document.getElementById('status-indicator');
const statusText = document.getElementById('status-text');

// Translation elements
const currentSign = document.getElementById('current-sign');
const currentConfidence = document.getElementById('current-confidence');
const sequenceProgress = document.getElementById('sequence-progress');
const translationHistory = document.getElementById('translation-history');
const clearTranslationsBtn = document.getElementById('clear-translations-btn');
const toggleAudioBtn = document.getElementById('toggle-audio-btn');
const audioSpeed = document.getElementById('audio-speed');

// Emotion elements
const emotionDisplayCompact = document.getElementById('emotion-display-compact');
const emotionConfidenceCompact = document.getElementById('emotion-confidence-compact');
const emotionBarFill = document.getElementById('emotion-bar-fill');
const valenceValueCompact = document.getElementById('valence-value-compact');
const arousalValueCompact = document.getElementById('arousal-value-compact');
const tuneEmotionsBtn = document.getElementById('tune-emotions-btn');
const tuneModal = document.getElementById('tune-modal');

// Video test elements
const uploadArea = document.getElementById('upload-area');
const fileInput = document.getElementById('file-input');
const testVideoContainer = document.getElementById('test-video-container');
const testVideo = document.getElementById('test-video');
const analyzeVideoBtn = document.getElementById('analyze-video-btn');

// State
let currentStream;
let ws;
let frameCount = 0;
let lastFrameTime = Date.now();
let translations = [];
let audioEnabled = true;
let lastSpokenSign = '';

// Emotion helper functions
function getEmotionEmoji(emotion) {
    const emojis = {
        'happy': '😊', 'sad': '😢', 'angry': '😠', 'surprised': '😮',
        'disgusted': '🤢', 'fearful': '😨', 'neutral': '😐'
    };
    return emojis[emotion] || '😐';
}

function getEmotionColor(emotion) {
    const colors = {
        'happy': '#00ff00', 'sad': '#4d79ff', 'angry': '#ff4444',
        'surprised': '#ffff00', 'disgusted': '#9933cc', 'fearful': '#ff8800',
        'neutral': '#cccccc'
    };
    return colors[emotion] || '#cccccc';
}

// WebSocket Connection
function connectWebSocket() {
    console.log('🔄 Connecting to WebSocket...');
    setStatus('Connecting...', 'disconnected');
    
    const protocol = (location.protocol === 'https:') ? 'wss:' : 'ws:';
    const wsUrl = protocol + '//' + location.host + '/ws';
    ws = new WebSocket(wsUrl);
    console.log('🌐 WebSocket URL:', wsUrl);
    
    ws.onopen = () => {
        console.log('✅ WebSocket connected');
        setStatus('Connected', 'connected');
        setInterval(sendFrame, 33); // ~30 FPS
    };
    
    ws.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);
            
            // Update processed image
            processedStream.src = "data:image/jpeg;base64," + data.image;
            
            // Update stats
            const debug = data.debug || {};
            outputStats.textContent = `FPS: ${Math.round(1000/debug.mediapipe_ms || 30)} | Faces: ${debug.faces_recognized || 0}`;
            
            // Update emotion
            updateEmotion(data.emotion || {});
            
            // Update sign language detection
            updateSignDetection(data.sign || {});
            
        } catch (e) {
            console.error("Error processing message:", e);
        }
    };
    
    ws.onclose = (event) => {
        console.log('❌ WebSocket closed');
        setStatus('Disconnected', 'disconnected');
        setTimeout(connectWebSocket, 3000);
    };
    
    ws.onerror = (error) => {
        console.error('❌ WebSocket error:', error);
        setStatus('Connection Error', 'disconnected');
    };
}

function sendFrame() {
    if (ws && ws.readyState === WebSocket.OPEN && video.videoWidth > 0) {
        captureContext.drawImage(video, 0, 0, 640, 480);
        captureCanvas.toBlob((blob) => {
            if (blob) {
                ws.send(blob);
                
                frameCount++;
                const now = Date.now();
                if (now - lastFrameTime > 1000) {
                    const fps = Math.round(frameCount * 1000 / (now - lastFrameTime));
                    inputStats.textContent = `${fps} FPS | ${Math.round(blob.size/1024)}KB`;
                    frameCount = 0;
                    lastFrameTime = now;
                }
            }
        }, 'image/jpeg', 0.8);
    } else if (ws && ws.readyState === WebSocket.OPEN && video.videoWidth === 0) {
        // Synthetic frame fallback
        frameCount++;
        const t = Date.now() / 1000;
        captureContext.fillStyle = '#111';
        captureContext.fillRect(0,0,640,480);
        captureContext.fillStyle = '#fff';
        captureContext.font = '20px Arial';
        captureContext.fillText('No webcam - synthetic frames', 20, 240);
        captureCanvas.toBlob(blob => {
            if (blob) ws.send(blob);
        }, 'image/jpeg', 0.7);
    }
}

function setStatus(text, state) {
    statusText.textContent = text;
    statusIndicator.className = 'status-indicator ' + state;
}

// Emotion Update
function updateEmotion(emotion) {
    if (!emotion.emotion) return;
    
    const emoji = getEmotionEmoji(emotion.emotion);
    const color = getEmotionColor(emotion.emotion);
    
    emotionDisplayCompact.textContent = `${emoji} ${emotion.emotion}`;
    emotionDisplayCompact.style.color = color;
    emotionConfidenceCompact.textContent = Math.round(emotion.confidence * 100) + '%';
    emotionBarFill.style.width = (emotion.confidence * 100) + '%';
    emotionBarFill.style.background = `linear-gradient(90deg, ${color}, #667eea)`;
    
    valenceValueCompact.textContent = (emotion.valence || 0).toFixed(2);
    arousalValueCompact.textContent = (emotion.arousal || 0).toFixed(2);
}

// Sign Language Detection
function updateSignDetection(sign) {
    const predictedSign = sign.predicted_sign || 'Waiting...';
    const confidence = sign.confidence || 0;
    const progress = sign.sequence_progress || 0;
    
    // Update current detection
    currentSign.textContent = predictedSign;
    currentConfidence.textContent = `Confidence: ${Math.round(confidence * 100)}%`;
    sequenceProgress.style.width = (progress * 100) + '%';
    
    // Add to translation history if confident
    if (confidence > 0.7 && predictedSign !== 'Waiting...' && 
        predictedSign !== 'Collecting...' && predictedSign !== 'Uncertain' &&
        predictedSign !== lastSpokenSign) {
        
        addTranslation(predictedSign, confidence);
        
        // Text-to-speech
        if (audioEnabled) {
            speak(predictedSign);
        }
        
        lastSpokenSign = predictedSign;
        
        // Reset after speaking to allow repeat
        setTimeout(() => {
            lastSpokenSign = '';
        }, 2000);
    }
}

function addTranslation(sign, confidence) {
    const timestamp = new Date().toLocaleTimeString();
    
    const translation = {
        sign: sign,
        confidence: confidence,
        timestamp: timestamp
    };
    
    translations.unshift(translation);
    if (translations.length > 50) {
        translations.pop();
    }
    
    renderTranslations();
}

function renderTranslations() {
    if (translations.length === 0) {
        translationHistory.innerHTML = '<p style="text-align: center; opacity: 0.5; padding: 40px;">Start signing to see translations here...</p>';
        return;
    }
    
    translationHistory.innerHTML = translations.map((t, index) => `
        <div class="translation-item ${index === 0 ? 'new' : ''}">
            <div class="translation-text">${t.sign}</div>
            <div class="translation-meta">
                <span class="translation-confidence">Confidence: ${Math.round(t.confidence * 100)}%</span>
                <span>${t.timestamp}</span>
            </div>
        </div>
    `).join('');
}

// Text-to-Speech
function speak(text) {
    if ('speechSynthesis' in window) {
        const utterance = new SpeechSynthesisUtterance(text);
        utterance.rate = parseFloat(audioSpeed.value);
        utterance.pitch = 1.0;
        utterance.volume = 1.0;
        
        // Try to use a natural voice
        const voices = speechSynthesis.getVoices();
        const preferredVoice = voices.find(v => v.lang.startsWith('en') && v.name.includes('Natural'));
        if (preferredVoice) {
            utterance.voice = preferredVoice;
        }
        
        speechSynthesis.speak(utterance);
    }
}

// Camera Setup
async function getCameras() {
    try {
        const devices = await navigator.mediaDevices.enumerateDevices();
        const videoDevices = devices.filter(device => device.kind === 'videoinput');
        
        cameraSelect.innerHTML = '';
        videoDevices.forEach((device, index) => {
            const option = document.createElement('option');
            option.value = device.deviceId;
            option.text = device.label || `Camera ${index + 1}`;
            cameraSelect.appendChild(option);
        });
        
        console.log(`Found ${videoDevices.length} cameras`);
    } catch (err) {
        console.error("Error enumerating devices:", err);
    }
}

async function setupWebcam(deviceId) {
    if (currentStream) {
        currentStream.getTracks().forEach(track => track.stop());
    }
    
    const constraints = {
        video: {
            deviceId: deviceId ? { exact: deviceId } : undefined,
            width: { ideal: 640 },
            height: { ideal: 480 },
            frameRate: { ideal: 30 }
        }
    };
    
    try {
        console.log('🔄 Requesting camera...');
        const stream = await navigator.mediaDevices.getUserMedia(constraints);
        console.log('✅ Camera granted');
        
        video.srcObject = stream;
        currentStream = stream;
        
        video.onloadedmetadata = () => {
            console.log(`✅ Video ready: ${video.videoWidth}x${video.videoHeight}`);
        };
        
    } catch (err) {
        console.error('❌ Camera error:', err);
        setStatus('Camera Error', 'disconnected');
    }
}

// Event Listeners
switchBtn.addEventListener('click', () => {
    setupWebcam(cameraSelect.value);
});

testBtn.addEventListener('click', async () => {
    try {
        const response = await fetch('/health');
        const data = await response.json();
        alert(`Server Health:\n${JSON.stringify(data, null, 2)}`);
    } catch (e) {
        alert('Server connection failed!');
    }
});

toggleGenderBtn.addEventListener('click', async () => {
    try {
        const res = await fetch('/gender/toggle', { method: 'POST' });
        const data = await res.json();
        toggleGenderBtn.textContent = data.enabled ? '⚥ Gender: ON' : '⚥ Gender: OFF';
    } catch (e) {
        console.error('Failed to toggle gender:', e);
    }
});

clearTranslationsBtn.addEventListener('click', () => {
    translations = [];
    renderTranslations();
});

toggleAudioBtn.addEventListener('click', () => {
    audioEnabled = !audioEnabled;
    toggleAudioBtn.textContent = audioEnabled ? '🔊 Audio: ON' : '🔇 Audio: OFF';
    toggleAudioBtn.style.background = audioEnabled ? 'var(--success)' : 'var(--danger)';
});

// Emotion Fine-tuning
tuneEmotionsBtn.addEventListener('click', () => {
    tuneModal.style.display = 'block';
});

document.getElementById('close-tune-btn').addEventListener('click', () => {
    tuneModal.style.display = 'none';
});

document.getElementById('happy-threshold').addEventListener('input', (e) => {
    document.getElementById('happy-val').textContent = e.target.value;
});

document.getElementById('sad-threshold').addEventListener('input', (e) => {
    document.getElementById('sad-val').textContent = e.target.value;
});

document.getElementById('neutral-threshold').addEventListener('input', (e) => {
    document.getElementById('neutral-val').textContent = e.target.value;
});

document.getElementById('apply-tune-btn').addEventListener('click', async () => {
    const tuneParams = {
        emotion_thresholds: {
            happy: parseFloat(document.getElementById('happy-threshold').value),
            sad: parseFloat(document.getElementById('sad-threshold').value),
            neutral: parseFloat(document.getElementById('neutral-threshold').value)
        }
    };
    
    try {
        const response = await fetch('/emotion/tune', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(tuneParams)
        });
        
        const result = await response.json();
        alert(result.success ? '✅ Emotion thresholds updated!' : '❌ Update failed: ' + result.error);
    } catch (error) {
        alert('❌ Network error: ' + error.message);
    }
    
    tuneModal.style.display = 'none';
});

// Video Upload Test
uploadArea.addEventListener('click', () => {
    fileInput.click();
});

uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('dragover');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('dragover');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleVideoUpload(files[0]);
    }
});

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleVideoUpload(e.target.files[0]);
    }
});

function handleVideoUpload(file) {
    if (!file.type.startsWith('video/')) {
        alert('Please upload a video file');
        return;
    }
    
    const url = URL.createObjectURL(file);
    testVideo.src = url;
    testVideoContainer.style.display = 'block';
    
    console.log('📹 Video uploaded:', file.name);
}

analyzeVideoBtn.addEventListener('click', () => {
    alert('Video analysis feature coming soon!\nThis will process the entire video and show sign detection accuracy.');
});

// Initialize
async function init() {
    try {
        console.log('🚀 Initializing SignL...');
        setStatus('Getting cameras...', 'disconnected');
        
        await getCameras();
        console.log('✅ Cameras enumerated');
        
        setStatus('Setting up webcam...', 'disconnected');
        await setupWebcam();
        console.log('✅ Webcam ready');
        
        setStatus('Connecting...', 'disconnected');
        connectWebSocket();
        console.log('✅ WebSocket initiated');
        
        // Load speech synthesis voices
        if ('speechSynthesis' in window) {
            speechSynthesis.getVoices();
        }
        
    } catch (error) {
        console.error('❌ Initialization error:', error);
        setStatus('Error: ' + error.message, 'disconnected');
    }
}

// Global error handlers
window.addEventListener('error', function(e) {
    console.error('❌ Global error:', e.error);
});

window.addEventListener('unhandledrejection', function(e) {
    console.error('❌ Unhandled promise rejection:', e.reason);
});

// Start
console.log('🚀 Starting SignL frontend...');
init();
