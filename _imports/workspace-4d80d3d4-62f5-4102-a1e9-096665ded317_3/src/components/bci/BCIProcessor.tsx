'use client';

import { useState, useEffect, useRef, useCallback } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Slider } from '@/components/ui/slider';
import { Progress } from '@/components/ui/progress';
import { 
  Brain, 
  Zap, 
  Activity, 
  Cpu, 
  Radio,
  Activity as PulseIcon,
  Battery,
  Waves,
  Skull
} from 'lucide-react';

interface BCIConfig {
  eegChannels: number;
  neuralImplants: boolean;
  thoughtToSign: boolean;
  signToThought: boolean;
  neuralLatency: number; // microseconds
  signalQuality: number;
  noiseReduction: boolean;
  adaptiveLearning: boolean;
}

interface NeuralSignal {
  channel: number;
  frequency: number;
  amplitude: number;
  phase: number;
  timestamp: number;
  thoughtPattern: string;
  confidence: number;
}

interface BCIResult {
  thought: string;
  sign: string;
  confidence: number;
  neuralLatency: number;
  signalQuality: number;
  brainRegion: string;
  emotionalState: string;
  cognitiveLoad: number;
}

interface BrainRegion {
  name: string;
  activity: number;
  thoughts: string[];
  signals: NeuralSignal[];
}

export default function BCIProcessor() {
  const [isInitialized, setIsInitialized] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [isConnected, setIsConnected] = useState(false);
  const [config, setConfig] = useState<BCIConfig>({
    eegChannels: 64,
    neuralImplants: false,
    thoughtToSign: true,
    signToThought: true,
    neuralLatency: 50,
    signalQuality: 0.85,
    noiseReduction: true,
    adaptiveLearning: true
  });

  const [neuralSignals, setNeuralSignals] = useState<NeuralSignal[]>([]);
  const [brainRegions, setBrainRegions] = useState<BrainRegion[]>([]);
  const [bciResult, setBciResult] = useState<BCIResult | null>(null);
  const [neuralMetrics, setNeuralMetrics] = useState({
    signalStrength: 0,
    noiseLevel: 0,
    bandwidth: 0,
    latency: config.neuralLatency,
    accuracy: 0
  });

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationRef = useRef<number>();

  useEffect(() => {
    initializeBCI();
    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, []);

  const initializeBCI = async () => {
    console.log('🧠 Initializing Brain-Computer Interface...');
    
    // Simulate BCI initialization
    await new Promise(resolve => setTimeout(resolve, 2000));
    
    // Initialize neural signals
    const initialSignals: NeuralSignal[] = Array.from({ length: config.eegChannels }, (_, i) => ({
      channel: i,
      frequency: 1 + Math.random() * 100, // 1-100 Hz
      amplitude: Math.random() * 100, // microvolts
      phase: Math.random() * Math.PI * 2,
      timestamp: Date.now(),
      thoughtPattern: generateThoughtPattern(i),
      confidence: 0.7 + Math.random() * 0.3
    }));
    
    // Initialize brain regions
    const initialBrainRegions: BrainRegion[] = [
      { name: 'Frontal Lobe', activity: 0.7, thoughts: [], signals: [] },
      { name: 'Parietal Lobe', activity: 0.6, thoughts: [], signals: [] },
      { name: 'Temporal Lobe', activity: 0.8, thoughts: [], signals: [] },
      { name: 'Occipital Lobe', activity: 0.5, thoughts: [], signals: [] },
      { name: 'Motor Cortex', activity: 0.9, thoughts: [], signals: [] },
      { name: 'Auditory Cortex', activity: 0.4, thoughts: [], signals: [] }
    ];
    
    setNeuralSignals(initialSignals);
    setBrainRegions(initialBrainRegions);
    setIsConnected(true);
    setIsInitialized(true);
    startNeuralProcessing();
    
    console.log('✅ BCI initialized with', config.eegChannels, 'EEG channels');
  };

  const generateThoughtPattern = (channel: number): string => {
    const patterns = [
      'hello', 'thank you', 'please', 'yes', 'no', 'help', 'water', 'food',
      'home', 'family', 'friend', 'love', 'happy', 'sad', 'angry', 'thinking'
    ];
    return patterns[channel % patterns.length];
  };

  const startNeuralProcessing = () => {
    const process = () => {
      // Update neural signals
      setNeuralSignals(prevSignals => 
        prevSignals.map(signal => ({
          ...signal,
          frequency: signal.frequency * 0.99 + Math.random() * 2,
          amplitude: signal.amplitude * 0.98 + Math.random() * 4,
          phase: (signal.phase + 0.01) % (Math.PI * 2),
          timestamp: Date.now(),
          confidence: Math.min(0.99, signal.confidence * 1.001)
        }))
      );

      // Update brain regions
      setBrainRegions(prevRegions => 
        prevRegions.map(region => ({
          ...region,
          activity: Math.sin(Date.now() * 0.0005 + prevRegions.indexOf(region)) * 0.3 + 0.7,
          thoughts: region.thoughts.length > 10 ? region.thoughts.slice(-10) : [
            ...region.thoughts,
            generateThoughtPattern(Math.floor(Math.random() * 100))
          ]
        }))
      );

      // Update neural metrics
      setNeuralMetrics(prev => ({
        signalStrength: Math.sin(Date.now() * 0.0003) * 0.2 + 0.8,
        noiseLevel: config.noiseReduction ? 0.05 : 0.15,
        bandwidth: 1000 + Math.sin(Date.now() * 0.0007) * 200,
        latency: config.neuralLatency * (0.9 + Math.random() * 0.2),
        accuracy: 0.85 + Math.sin(Date.now() * 0.0004) * 0.1
      }));

      // Perform BCI processing
      performBCIProcessing();

      animationRef.current = requestAnimationFrame(process);
    };
    process();
  };

  const performBCIProcessing = () => {
    if (neuralSignals.length === 0) return;
    
    // Analyze neural patterns
    const avgAmplitude = neuralSignals.reduce((sum, signal) => sum + signal.amplitude, 0) / neuralSignals.length;
    const avgFrequency = neuralSignals.reduce((sum, signal) => sum + signal.frequency, 0) / neuralSignals.length;
    const avgConfidence = neuralSignals.reduce((sum, signal) => sum + signal.confidence, 0) / neuralSignals.length;
    
    // Determine dominant thought pattern
    const thoughtCounts = neuralSignals.reduce((counts, signal) => {
      counts[signal.thoughtPattern] = (counts[signal.thoughtPattern] || 0) + 1;
      return counts;
    }, {} as Record<string, number>);
    
    const dominantThought = Object.entries(thoughtCounts)
      .sort(([,a], [,b]) => b - a)[0]?.[0] || 'unknown';
    
    // Determine most active brain region
    const mostActiveRegion = brainRegions.reduce((most, region) => 
      region.activity > most.activity ? region : most
    );
    
    // Determine emotional state
    const emotionalStates = ['happy', 'sad', 'angry', 'neutral', 'excited', 'calm'];
    const emotionalState = emotionalStates[Math.floor(Date.now() / 3000) % emotionalStates.length];
    
    // Calculate cognitive load
    const cognitiveLoad = avgAmplitude / 100 * avgFrequency / 50;
    
    const result: BCIResult = {
      thought: dominantThought,
      sign: config.thoughtToSign ? translateThoughtToSign(dominantThought) : '',
      confidence: avgConfidence,
      neuralLatency: config.neuralLatency,
      signalQuality: config.signalQuality,
      brainRegion: mostActiveRegion.name,
      emotionalState,
      cognitiveLoad
    };

    setBciResult(result);
  };

  const translateThoughtToSign = (thought: string): string => {
    const translations: Record<string, string> = {
      'hello': '👋',
      'thank you': '🙏',
      'please': '🤲',
      'yes': '👍',
      'no': '👎',
      'help': '🆘',
      'water': '💧',
      'food': '🍽',
      'home': '🏠',
      'family': '👨‍👩‍👧‍👦',
      'friend': '👫',
      'love': '❤️',
      'happy': '😊',
      'sad': '😢',
      'angry': '😠',
      'thinking': '🤔'
    };
    
    return translations[thought] || thought;
  };

  const executeBCICommand = useCallback(async () => {
    setIsProcessing(true);
    
    // Simulate BCI command execution
    await new Promise(resolve => setTimeout(resolve, 2000));
    
    console.log('🧠 BCI Command Executed:', {
      thought: bciResult?.thought,
      sign: bciResult?.sign,
      confidence: bciResult?.confidence,
      brainRegion: bciResult?.brainRegion,
      emotionalState: bciResult?.emotionalState,
      cognitiveLoad: bciResult?.cognitiveLoad
    });
    
    setIsProcessing(false);
  }, [bciResult]);

  const updateConfig = <K extends keyof BCIConfig>(
    key: K, 
    value: BCIConfig[K]
  ) => {
    setConfig(prev => ({ ...prev, [key]: value }));
  };

  const renderBrainVisualization = () => {
    if (!canvasRef.current) return;
    
    const ctx = canvasRef.current.getContext('2d');
    if (!ctx) return;
    
    const canvas = canvasRef.current;
    canvas.width = canvas.offsetWidth;
    canvas.height = canvas.offsetHeight;
    
    // Clear canvas
    ctx.fillStyle = 'rgba(0, 0, 0, 0.1)';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    const centerX = canvas.width / 2;
    const centerY = canvas.height / 2;
    const brainRadius = Math.min(canvas.width, canvas.height) / 3;
    
    // Draw brain regions
    brainRegions.forEach((region, index) => {
      const angle = (index / brainRegions.length) * Math.PI * 2 - Math.PI / 2;
      const x = centerX + Math.cos(angle) * brainRadius;
      const y = centerY + Math.sin(angle) * brainRadius;
      
      // Draw region
      const gradient = ctx.createRadialGradient(x, y, 0, x, y, 30);
      const intensity = region.activity;
      gradient.addColorStop(0, `rgba(255, 100, 100, ${intensity})`);
      gradient.addColorStop(1, `rgba(255, 100, 100, 0)`);
      
      ctx.fillStyle = gradient;
      ctx.beginPath();
      ctx.arc(x, y, 20 + intensity * 10, 0, Math.PI * 2);
      ctx.fill();
      
      // Draw region label
      ctx.fillStyle = 'white';
      ctx.font = '10px Arial';
      ctx.textAlign = 'center';
      ctx.fillText(region.name, x, y + 35 + intensity * 10);
    });
    
    // Draw EEG signals
    neuralSignals.slice(0, 32).forEach((signal, index) => {
      const angle = (index / 32) * Math.PI * 2;
      const x = centerX + Math.cos(angle) * (brainRadius + 50);
      const y = centerY + Math.sin(angle) * (brainRadius + 50);
      
      // Draw signal waveform
      ctx.strokeStyle = `rgba(0, 255, 255, ${signal.confidence})`;
      ctx.lineWidth = 2;
      ctx.beginPath();
      
      for (let i = 0; i < 50; i++) {
        const waveX = x + i * 2 - 50;
        const waveY = y + Math.sin(i * 0.2 + signal.phase) * signal.amplitude / 10;
        
        if (i === 0) {
          ctx.moveTo(waveX, waveY);
        } else {
          ctx.lineTo(waveX, waveY);
        }
      }
      
      ctx.stroke();
    });
    
    // Draw central brain
    const brainGradient = ctx.createRadialGradient(centerX, centerY, 0, centerX, centerY, 40);
    brainGradient.addColorStop(0, `rgba(255, 255, 255, ${neuralMetrics.signalStrength})`);
    brainGradient.addColorStop(1, `rgba(255, 255, 255, 0.2)`);
    
    ctx.fillStyle = brainGradient;
    ctx.beginPath();
    ctx.arc(centerX, centerY, 40, 0, Math.PI * 2);
    ctx.fill();
    
    // Draw brain icon
    ctx.fillStyle = '#ff6b6b';
    ctx.font = '24px Arial';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText('🧠', centerX, centerY);
  };

  useEffect(() => {
    renderBrainVisualization();
  }, [neuralSignals, brainRegions, neuralMetrics]);

  return (
    <div className="space-y-6">
      {/* Header */}
      <Card className="bg-black/30 backdrop-blur-md border-red-500/20">
        <CardHeader>
          <CardTitle className="text-white flex items-center gap-2">
            <Brain className="w-5 h-5 text-red-400" />
            Brain-Computer Interface
          </CardTitle>
          <div className="flex items-center gap-2">
            <Badge className={isInitialized ? "bg-green-600" : "bg-yellow-600"}>
              {isInitialized ? "Neural Ready" : "Initializing"}
            </Badge>
            <Badge className={isConnected ? "bg-green-600" : "bg-red-600"}>
              {isConnected ? "Connected" : "Disconnected"}
            </Badge>
            <Badge className={isProcessing ? "bg-blue-600" : "bg-gray-600"}>
              {isProcessing ? "Processing" : "Idle"}
            </Badge>
            <Badge className="bg-red-600">
              {config.eegChannels} Channels
            </Badge>
          </div>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-center">
            <div>
              <div className="text-2xl font-bold text-red-400">
                {(neuralMetrics.signalStrength * 100).toFixed(1)}%
              </div>
              <div className="text-gray-400 text-sm">Signal Strength</div>
            </div>
            <div>
              <div className="text-2xl font-bold text-cyan-400">
                {neuralMetrics.latency}μs
              </div>
              <div className="text-gray-400 text-sm">Neural Latency</div>
            </div>
            <div>
              <div className="text-2xl font-bold text-green-400">
                {(neuralMetrics.accuracy * 100).toFixed(1)}%
              </div>
              <div className="text-gray-400 text-sm">Accuracy</div>
            </div>
            <div>
              <div className="text-2xl font-bold text-purple-400">
                {neuralMetrics.bandwidth.toFixed(0)}Hz
              </div>
              <div className="text-gray-400 text-sm">Bandwidth</div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Brain Visualization */}
      <Card className="bg-black/30 backdrop-blur-md border-red-500/20">
        <CardHeader>
          <CardTitle className="text-white flex items-center gap-2">
            <Skull className="w-5 h-5" />
            Neural Activity Map
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="relative aspect-video bg-gradient-to-br from-red-900/20 to-pink-900/20 rounded-lg overflow-hidden">
            <canvas
              ref={canvasRef}
              className="w-full h-full"
              style={{ imageRendering: 'crisp-edges' }}
            />
            
            {/* BCI Status Overlay */}
            <div className="absolute top-4 left-4 space-y-2">
              <div className="bg-black/70 backdrop-blur-sm rounded-lg px-3 py-2">
                <div className="text-red-400 text-sm font-medium">
                  Thought-to-Sign: {config.thoughtToSign ? 'ACTIVE' : 'INACTIVE'}
                </div>
              </div>
              <div className="bg-black/70 backdrop-blur-sm rounded-lg px-3 py-2">
                <div className="text-cyan-400 text-sm font-medium">
                  Sign-to-Thought: {config.signToThought ? 'ACTIVE' : 'INACTIVE'}
                </div>
              </div>
            </div>
            
            {/* Current Thought */}
            {bciResult && (
              <div className="absolute top-4 right-4 bg-black/70 backdrop-blur-sm rounded-lg px-3 py-2">
                <div className="text-white text-sm font-medium">
                  Current Thought: {bciResult.thought}
                </div>
                <div className="text-lg">
                  {bciResult.sign}
                </div>
              </div>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Brain Regions Activity */}
      <Card className="bg-black/30 backdrop-blur-md border-red-500/20">
        <CardHeader>
          <CardTitle className="text-white flex items-center gap-2">
            <Activity className="w-5 h-5" />
            Brain Regions Activity
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
            {brainRegions.map((region, index) => (
              <div key={region.name} className="bg-black/20 rounded-lg p-4">
                <div className="flex items-center justify-between mb-3">
                  <h4 className="text-white font-medium">{region.name}</h4>
                  <div className={`w-3 h-3 rounded-full ${
                    region.activity > 0.7 ? 'bg-green-400' : 
                    region.activity > 0.4 ? 'bg-yellow-400' : 'bg-red-400'
                  }`} />
                </div>
                
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-gray-400">Activity:</span>
                    <span className="text-red-400">{(region.activity * 100).toFixed(1)}%</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Thoughts:</span>
                    <span className="text-cyan-400">{region.thoughts.length}</span>
                  </div>
                </div>
                
                {/* Recent Thoughts */}
                <div className="mt-3 h-12 bg-black/30 rounded flex flex-wrap gap-1 p-1">
                  {region.thoughts.slice(-5).map((thought, i) => (
                    <div key={i} className="text-xs text-red-300 bg-red-900/30 px-1 rounded">
                      {thought.substring(0, 8)}
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* BCI Results */}
      {bciResult && (
        <Card className="bg-black/30 backdrop-blur-md border-red-500/20">
          <CardHeader>
            <CardTitle className="text-white flex items-center gap-2">
              <PulseIcon className="w-5 h-5" />
              Neural Translation Results
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <h4 className="text-white font-medium mb-2">Thought Analysis</h4>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-gray-400">Dominant Thought:</span>
                    <span className="text-red-400">{bciResult.thought}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Confidence:</span>
                    <span className="text-cyan-400">{(bciResult.confidence * 100).toFixed(1)}%</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Emotional State:</span>
                    <span className="text-green-400">{bciResult.emotionalState}</span>
                  </div>
                </div>
              </div>

              <div>
                <h4 className="text-white font-medium mb-2">Neural Metrics</h4>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-gray-400">Brain Region:</span>
                    <span className="text-purple-400">{bciResult.brainRegion}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Cognitive Load:</span>
                    <span className="text-orange-400">{(bciResult.cognitiveLoad * 100).toFixed(1)}%</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Signal Quality:</span>
                    <span className="text-green-400">{(bciResult.signalQuality * 100).toFixed(1)}%</span>
                  </div>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Configuration */}
      <Card className="bg-black/30 backdrop-blur-md border-red-500/20">
        <CardHeader>
          <CardTitle className="text-white flex items-center gap-2">
            <Cpu className="w-5 h-5" />
            BCI Configuration
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="text-white text-sm font-medium mb-2 block">
                EEG Channels: {config.eegChannels}
              </label>
              <Slider
                value={[config.eegChannels]}
                onValueChange={([value]) => updateConfig('eegChannels', value)}
                max={256}
                min={16}
                step={16}
                className="w-full"
              />
            </div>

            <div>
              <label className="text-white text-sm font-medium mb-2 block">
                Neural Latency: {config.neuralLatency}μs
              </label>
              <Slider
                value={[config.neuralLatency]}
                onValueChange={([value]) => updateConfig('neuralLatency', value)}
                max={200}
                min={10}
                step={10}
                className="w-full"
              />
            </div>

            <div>
              <label className="text-white text-sm font-medium mb-2 block">
                Signal Quality: {(config.signalQuality * 100).toFixed(0)}%
              </label>
              <Slider
                value={[config.signalQuality]}
                onValueChange={([value]) => updateConfig('signalQuality', value)}
                max={1.0}
                min={0.1}
                step={0.05}
                className="w-full"
              />
            </div>
          </div>

          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2">
              <input
                type="checkbox"
                id="thoughtToSign"
                checked={config.thoughtToSign}
                onChange={(e) => updateConfig('thoughtToSign', e.target.checked)}
                className="rounded"
              />
              <label htmlFor="thoughtToSign" className="text-white text-sm">
                Thought-to-Sign
              </label>
            </div>

            <div className="flex items-center gap-2">
              <input
                type="checkbox"
                id="signToThought"
                checked={config.signToThought}
                onChange={(e) => updateConfig('signToThought', e.target.checked)}
                className="rounded"
              />
              <label htmlFor="signToThought" className="text-white text-sm">
                Sign-to-Thought
              </label>
            </div>

            <div className="flex items-center gap-2">
              <input
                type="checkbox"
                id="neuralImplants"
                checked={config.neuralImplants}
                onChange={(e) => updateConfig('neuralImplants', e.target.checked)}
                className="rounded"
              />
              <label htmlFor="neuralImplants" className="text-white text-sm">
                Neural Implants
              </label>
            </div>

            <div className="flex items-center gap-2">
              <input
                type="checkbox"
                id="noiseReduction"
                checked={config.noiseReduction}
                onChange={(e) => updateConfig('noiseReduction', e.target.checked)}
                className="rounded"
              />
              <label htmlFor="noiseReduction" className="text-white text-sm">
                Noise Reduction
              </label>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Control Panel */}
      <Card className="bg-black/30 backdrop-blur-md border-red-500/20">
        <CardContent className="pt-6">
          <div className="flex items-center justify-center gap-4">
            <Button
              onClick={executeBCICommand}
              disabled={!isInitialized || isProcessing}
              className="bg-gradient-to-r from-red-600 to-pink-600 hover:from-red-700 hover:to-pink-700"
            >
              <Zap className="w-4 h-4 mr-2" />
              {isProcessing ? 'Neural Processing...' : 'Execute Neural Command'}
            </Button>
            
            <Button
              variant="outline"
              onClick={() => setBciResult(null)}
              className="border-red-500 text-red-400 hover:bg-red-600/20"
            >
              Clear Results
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}