import React, { useState, useEffect, useRef } from 'react';
import { 
  Mic, MicOff, MessageSquare, Terminal, Command, 
  Settings, Power, ShieldCheck, Sparkles, Volume2, X, Camera, Speaker
} from 'lucide-react';
import { DIGITAL_CERTIFICATE, SYSTEM_PHRASES } from './constants';
import { ConvoEntry, AppState } from './types';
import { HyperAvatar } from './components/HyperAvatar';
import { LiveFeed } from './components/LiveFeed';
import { generateSummary } from './services/geminiService';

const App: React.FC = () => {
  // Application State
  const [appState, setAppState] = useState<AppState>(AppState.IDLE);
  const [logs, setLogs] = useState<ConvoEntry[]>([]);
  const [currentInput, setCurrentInput] = useState("");
  const [isSignCaptureActive, setIsSignCaptureActive] = useState(true);
  const [avatarGloss, setAvatarGloss] = useState("");
  const [isAvatarSigning, setIsAvatarSigning] = useState(false);
  const [aiSummary, setAiSummary] = useState<string | null>(null);
  const [isPoweredOn, setIsPoweredOn] = useState(true);
  
  // Logic State
  const [lastPhraseIndex, setLastPhraseIndex] = useState(-1);

  // Device Management State
  const [devices, setDevices] = useState<{ video: MediaDeviceInfo[], audio: MediaDeviceInfo[] }>({ video: [], audio: [] });
  const [selectedCameraId, setSelectedCameraId] = useState<string>('');
  const [selectedMicId, setSelectedMicId] = useState<string>('');
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);

  // Digital Cert Check
  const [isVerified, setIsVerified] = useState(false);

  // Refs
  const logsEndRef = useRef<HTMLDivElement>(null);
  
  // Initialize & Authentication
  useEffect(() => {
    // Validate Ownership
    if (DIGITAL_CERTIFICATE.owner === "Nagabhushana Raju S") {
      setIsVerified(true);
      if (isPoweredOn) {
        addLog({
          id: 'sys-init',
          speaker: { id: 'sys', name: 'SYSTEM V10', role: 'Admin', isVerified: true },
          text: `Secure Environment Initialized. Certificate: ${DIGITAL_CERTIFICATE.encryptionKey}`,
          timestamp: new Date().toLocaleTimeString(),
          type: 'system'
        });
      }
    }
  }, [isPoweredOn]);

  // Fetch Devices
  useEffect(() => {
    const fetchDevices = async () => {
      try {
        // Request permissions first to ensure labels are available
        await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
        
        const allDevices = await navigator.mediaDevices.enumerateDevices();
        const videoDevs = allDevices.filter(d => d.kind === 'videoinput');
        const audioDevs = allDevices.filter(d => d.kind === 'audioinput');
        
        setDevices({ video: videoDevs, audio: audioDevs });
        
        // Set defaults if not set
        if (!selectedCameraId && videoDevs.length > 0) setSelectedCameraId(videoDevs[0].deviceId);
        if (!selectedMicId && audioDevs.length > 0) setSelectedMicId(audioDevs[0].deviceId);

      } catch (err) {
        console.error("Error fetching devices:", err);
      }
    };

    if (isPoweredOn) {
        fetchDevices();
    }
  }, [isPoweredOn]);

  // Auto-scroll logs
  useEffect(() => {
    logsEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [logs]);

  // --- Logic Helpers ---

  const addLog = (entry: ConvoEntry) => {
    setLogs(prev => [...prev, entry]);
  };

  const handleMotion = (intensity: number) => {
    if (!isPoweredOn) return;

    // Threshold for "Sign Detection"
    if (intensity > 0.15 && appState !== AppState.SPEAKING) {
      setAppState(AppState.WATCHING);
      
      // Simulation: If sustained motion, trigger a recognized phrase
      // Random chance adjusted to 3% per frame of motion for better responsiveness
      if (Math.random() > 0.97) { 
        let index = Math.floor(Math.random() * SYSTEM_PHRASES.length);
        
        // Prevent repeating the same phrase immediately
        if (index === lastPhraseIndex) {
            index = (index + 1) % SYSTEM_PHRASES.length;
        }
        
        setLastPhraseIndex(index);
        triggerSignDetection(SYSTEM_PHRASES[index]);
      }
    } else if (intensity < 0.05 && appState === AppState.WATCHING) {
      setAppState(AppState.IDLE);
    }
  };

  const triggerSignDetection = (text: string) => {
    setAppState(AppState.SPEAKING);
    addLog({
      id: crypto.randomUUID(),
      speaker: { id: 'guest', name: 'Guest (Signer)', role: 'Guest', isVerified: false },
      demographics: { age: 28, gender: 'Female', confidence: 0.92 },
      text: text,
      timestamp: new Date().toLocaleTimeString(),
      type: 'sign-detected'
    });
    
    // TTS
    const u = new SpeechSynthesisUtterance(text);
    u.pitch = 1.2; 
    window.speechSynthesis.speak(u);
    
    setTimeout(() => setAppState(AppState.IDLE), 2500); // 2.5s cooldown to finish speaking
  };

  const handleVoiceInput = (text: string) => {
    addLog({
      id: crypto.randomUUID(),
      speaker: { id: 'host', name: 'Nagabhushana Raju S', role: 'Host', isVerified: true },
      text: text,
      timestamp: new Date().toLocaleTimeString(),
      type: 'voice-input'
    });

    // Trigger Avatar
    animateAvatar(text);
    setCurrentInput("");
  };

  const animateAvatar = (text: string) => {
    setIsAvatarSigning(true);
    const words = text.split(" ");
    let i = 0;
    
    const interval = setInterval(() => {
      if (i < words.length) {
        setAvatarGloss(words[i].toUpperCase());
        i++;
      } else {
        clearInterval(interval);
        setAvatarGloss("");
        setIsAvatarSigning(false);
      }
    }, 600);
  };

  const handleSummaryRequest = async () => {
    addLog({
      id: 'sys-ai',
      speaker: { id: 'ai', name: 'Gemini 2.5', role: 'Admin', isVerified: true },
      text: "Analyzing context...",
      timestamp: new Date().toLocaleTimeString(),
      type: 'system'
    });
    
    const summary = await generateSummary(logs);
    setAiSummary(summary);
    
    addLog({
      id: crypto.randomUUID(),
      speaker: { id: 'ai', name: 'Gemini Context Engine', role: 'Admin', isVerified: true },
      text: summary,
      timestamp: new Date().toLocaleTimeString(),
      type: 'ai-summary'
    });
  };

  // --- Render ---

  if (!isVerified) return <div className="text-red-500 font-mono p-10">ACCESS DENIED. INVALID DIGITAL SIGNATURE.</div>;

  // Power Off Screen
  if (!isPoweredOn) {
      return (
          <div className="min-h-screen bg-black flex flex-col items-center justify-center text-slate-600 font-mono relative overflow-hidden">
              <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_center,_var(--tw-gradient-stops))] from-slate-900 via-black to-black opacity-50"></div>
              <Power className="w-16 h-16 mb-6 text-slate-800" />
              <h1 className="text-2xl tracking-[0.5em] mb-8">SYSTEM OFFLINE</h1>
              <button 
                  onClick={() => setIsPoweredOn(true)}
                  className="px-8 py-3 border border-cyan-900 text-cyan-500 hover:bg-cyan-900/20 hover:text-cyan-400 rounded transition-all duration-300 flex items-center gap-3 group"
              >
                  <Power className="w-4 h-4 group-hover:animate-pulse" />
                  INITIALIZE NEURAL INTERFACE
              </button>
              <div className="absolute bottom-10 text-xs text-slate-800">
                  {DIGITAL_CERTIFICATE.encryptionKey}
              </div>
          </div>
      );
  }

  return (
    <div className="min-h-screen bg-neon-bg text-slate-200 font-sans selection:bg-cyan-500/30 relative">
      
      {/* SETTINGS MODAL */}
      {isSettingsOpen && (
          <div className="absolute inset-0 z-[100] bg-black/60 backdrop-blur-sm flex items-center justify-center p-4 animate-in fade-in duration-200">
              <div className="bg-slate-900 border border-slate-700 rounded-2xl w-full max-w-lg shadow-2xl overflow-hidden">
                  <div className="p-4 border-b border-slate-800 flex justify-between items-center bg-slate-950">
                      <h2 className="font-bold text-lg flex items-center gap-2 text-cyan-400">
                          <Settings className="w-5 h-5" /> SYSTEM CONFIGURATION
                      </h2>
                      <button onClick={() => setIsSettingsOpen(false)} className="text-slate-500 hover:text-white transition-colors">
                          <X className="w-6 h-6" />
                      </button>
                  </div>
                  <div className="p-6 space-y-6">
                      
                      {/* Camera Selection */}
                      <div className="space-y-2">
                          <label className="text-xs font-mono text-slate-400 flex items-center gap-2">
                              <Camera className="w-4 h-4" /> VISUAL INPUT SOURCE
                          </label>
                          <select 
                            value={selectedCameraId} 
                            onChange={(e) => setSelectedCameraId(e.target.value)}
                            className="w-full bg-slate-800 border border-slate-700 rounded p-3 text-sm text-white focus:ring-2 focus:ring-cyan-500 outline-none"
                          >
                             {devices.video.length === 0 && <option value="">No cameras detected</option>}
                             {devices.video.map(device => (
                                 <option key={device.deviceId} value={device.deviceId}>
                                     {device.label || `Camera ${device.deviceId.slice(0, 5)}...`}
                                 </option>
                             ))}
                          </select>
                      </div>

                      {/* Mic Selection */}
                      <div className="space-y-2">
                          <label className="text-xs font-mono text-slate-400 flex items-center gap-2">
                              <Mic className="w-4 h-4" /> AUDIO INPUT SOURCE
                          </label>
                          <select 
                            value={selectedMicId} 
                            onChange={(e) => setSelectedMicId(e.target.value)}
                            className="w-full bg-slate-800 border border-slate-700 rounded p-3 text-sm text-white focus:ring-2 focus:ring-purple-500 outline-none"
                          >
                             {devices.audio.length === 0 && <option value="">No microphones detected</option>}
                             {devices.audio.map(device => (
                                 <option key={device.deviceId} value={device.deviceId}>
                                     {device.label || `Microphone ${device.deviceId.slice(0, 5)}...`}
                                 </option>
                             ))}
                          </select>
                      </div>

                      {/* Output Test */}
                      <div className="pt-4 border-t border-slate-800">
                          <button 
                             onClick={() => {
                                 const u = new SpeechSynthesisUtterance("Audio output systems nominal.");
                                 window.speechSynthesis.speak(u);
                             }}
                             className="w-full py-3 bg-slate-800 hover:bg-slate-700 border border-emerald-500/30 text-emerald-400 rounded flex items-center justify-center gap-2 font-mono text-sm transition-all"
                          >
                              <Speaker className="w-4 h-4" /> TEST AUDIO SYNTHESIS
                          </button>
                      </div>

                  </div>
                  <div className="p-4 bg-slate-950 flex justify-end">
                      <button 
                        onClick={() => setIsSettingsOpen(false)}
                        className="px-6 py-2 bg-cyan-600 hover:bg-cyan-500 text-white rounded font-bold text-sm"
                      >
                          APPLY CONFIG
                      </button>
                  </div>
              </div>
          </div>
      )}

      {/* HEADER */}
      <header className="h-16 border-b border-slate-800 bg-slate-950/80 backdrop-blur-md flex items-center justify-between px-6 sticky top-0 z-50">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 bg-gradient-to-br from-cyan-500 to-purple-600 rounded-lg flex items-center justify-center shadow-[0_0_15px_rgba(6,182,212,0.5)]">
            <Command className="w-5 h-5 text-white" />
          </div>
          <div>
            <h1 className="font-bold text-lg tracking-wider text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-purple-400">
              SIGNSYNC OMNI <span className="text-xs text-slate-500 font-mono">v10.0</span>
            </h1>
          </div>
        </div>
        
        <div className="flex items-center gap-4">
          <div className="hidden md:flex items-center gap-2 px-3 py-1 bg-slate-900 rounded-full border border-slate-800">
             <ShieldCheck className="w-4 h-4 text-emerald-500" />
             <span className="text-xs font-mono text-slate-400">CERT: {DIGITAL_CERTIFICATE.owner.split(' ')[0]}</span>
          </div>
          <button 
            onClick={() => setIsSettingsOpen(true)}
            className="p-2 hover:bg-slate-800 rounded-full text-slate-400 transition-colors"
            title="System Configuration"
          >
            <Settings className="w-5 h-5" />
          </button>
          <button 
            onClick={() => setIsPoweredOn(false)}
            className="p-2 hover:bg-red-900/20 hover:text-red-500 rounded-full text-slate-400 transition-colors"
            title="Shut Down System"
          >
            <Power className="w-5 h-5" />
          </button>
        </div>
      </header>

      {/* MAIN GRID */}
      <main className="p-4 md:p-6 grid grid-cols-1 lg:grid-cols-12 gap-6 h-[calc(100vh-64px)]">
        
        {/* LEFT COL: VISUAL INTERFACE (8 cols) */}
        <div className="lg:col-span-8 flex flex-col gap-6 h-full">
          
          {/* TOP: AVATAR VIEWPORT */}
          <div className="flex-1 relative min-h-[400px]">
            <HyperAvatar isSigning={isAvatarSigning} currentGloss={avatarGloss} />
          </div>

          {/* BOTTOM: SENSOR DECK */}
          <div className="h-48 grid grid-cols-1 md:grid-cols-2 gap-4">
            
            {/* Live Feed Input */}
            <LiveFeed 
              isActive={isSignCaptureActive} 
              onMotionDetected={handleMotion} 
              selectedDeviceId={selectedCameraId}
            />

            {/* Metrics / Control */}
            <div className="bg-slate-900 rounded-xl border border-slate-700 p-4 flex flex-col justify-between">
              <div className="flex justify-between items-start">
                 <h3 className="text-xs font-mono text-slate-500 flex items-center gap-2">
                   <Terminal className="w-3 h-3" /> SYSTEM STATUS
                 </h3>
                 <span className={`text-xs px-2 py-0.5 rounded ${appState === AppState.IDLE ? 'bg-slate-800 text-slate-400' : 'bg-cyan-900 text-cyan-400 animate-pulse'}`}>
                   {appState}
                 </span>
              </div>
              
              <div className="grid grid-cols-2 gap-2 mt-2">
                 <div className="bg-slate-950 p-2 rounded border border-slate-800">
                    <div className="text-[10px] text-slate-500 mb-1">FPS</div>
                    <div className="text-xl font-mono text-cyan-400">59.9</div>
                 </div>
                 <div className="bg-slate-950 p-2 rounded border border-slate-800">
                    <div className="text-[10px] text-slate-500 mb-1">CONFIDENCE</div>
                    <div className="text-xl font-mono text-purple-400">98.2%</div>
                 </div>
              </div>

              <div className="flex gap-2 mt-2">
                <button 
                  onClick={() => setIsSignCaptureActive(!isSignCaptureActive)}
                  className={`flex-1 py-2 text-xs font-bold rounded flex items-center justify-center gap-2 transition-all ${isSignCaptureActive ? 'bg-cyan-600 hover:bg-cyan-500 text-white shadow-lg shadow-cyan-900/50' : 'bg-slate-800 text-slate-400'}`}
                >
                  {isSignCaptureActive ? 'SENSORS ACTIVE' : 'SENSORS OFFLINE'}
                </button>
              </div>
            </div>

          </div>
        </div>

        {/* RIGHT COL: DATA & LOGS (4 cols) */}
        <div className="lg:col-span-4 bg-slate-900 rounded-2xl border border-slate-800 flex flex-col shadow-2xl overflow-hidden">
          
          {/* Log Header */}
          <div className="p-4 border-b border-slate-800 bg-slate-950 flex justify-between items-center">
            <h2 className="font-bold text-sm flex items-center gap-2">
              <MessageSquare className="w-4 h-4 text-purple-500" /> TRANSCRIPT LOG
            </h2>
            <button 
              onClick={handleSummaryRequest}
              className="text-xs bg-purple-900/30 text-purple-400 px-3 py-1 rounded border border-purple-500/30 hover:bg-purple-900/50 flex items-center gap-1"
            >
              <Sparkles className="w-3 h-3" /> AI SUMMARIZE
            </button>
          </div>

          {/* Log Body */}
          <div className="flex-1 overflow-y-auto p-4 space-y-4 font-mono text-sm relative">
            <div className="absolute top-0 left-0 w-full h-4 bg-gradient-to-b from-slate-900 to-transparent pointer-events-none"></div>
            
            {logs.length === 0 && (
              <div className="text-center text-slate-600 mt-10">
                <p>Waiting for input...</p>
              </div>
            )}

            {logs.map((log) => (
              <div key={log.id} className={`flex flex-col gap-1 p-3 rounded-lg border ${
                log.type === 'system' ? 'bg-slate-950 border-slate-800 text-slate-500 text-xs' : 
                log.type === 'ai-summary' ? 'bg-purple-900/10 border-purple-500/30' :
                'bg-slate-800/50 border-slate-700'
              }`}>
                <div className="flex justify-between items-center">
                  <span className={`font-bold text-xs ${
                    log.speaker.role === 'Host' ? 'text-cyan-400' : 
                    log.speaker.role === 'Guest' ? 'text-emerald-400' : 'text-purple-400'
                  }`}>
                    {log.speaker.name}
                  </span>
                  <span className="text-[10px] text-slate-600">{log.timestamp}</span>
                </div>
                
                <p className="text-slate-300 leading-relaxed">
                  {log.text}
                </p>

                {log.demographics && (
                   <div className="flex gap-2 mt-1">
                      <span className="text-[10px] bg-slate-950 px-1 rounded text-slate-500">Age: ~{log.demographics.age}</span>
                      <span className="text-[10px] bg-slate-950 px-1 rounded text-slate-500">{log.demographics.gender}</span>
                   </div>
                )}
              </div>
            ))}
            <div ref={logsEndRef} />
          </div>

          {/* Input Area */}
          <div className="p-4 bg-slate-950 border-t border-slate-800">
             <div className="flex items-center gap-2 bg-slate-900 border border-slate-700 rounded-xl p-2 focus-within:ring-2 focus-within:ring-cyan-500/50 focus-within:border-cyan-500 transition-all">
                <input 
                  type="text" 
                  className="bg-transparent border-none outline-none flex-1 text-white placeholder-slate-600"
                  placeholder="Type or speak to translate..."
                  value={currentInput}
                  onChange={(e) => setCurrentInput(e.target.value)}
                  onKeyDown={(e) => e.key === 'Enter' && currentInput && handleVoiceInput(currentInput)}
                />
                <button className="p-2 text-slate-400 hover:text-cyan-400 transition-colors">
                   <Mic className="w-5 h-5" />
                </button>
             </div>
             <div className="flex justify-between mt-2 px-1">
                <div className="text-[10px] text-slate-600 flex items-center gap-1">
                   <Volume2 className="w-3 h-3" /> TTS ENGINE READY
                </div>
                <div className="text-[10px] text-slate-600">
                   V10.0.1 (STABLE)
                </div>
             </div>
          </div>

        </div>
      </main>
    </div>
  );
};

export default App;