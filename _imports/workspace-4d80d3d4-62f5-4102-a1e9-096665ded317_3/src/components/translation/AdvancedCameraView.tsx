'use client';

import { useState, useEffect, useRef } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Slider } from '@/components/ui/slider';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { 
  Camera, 
  CameraOff, 
  Mic, 
  MicOff, 
  Volume2, 
  Settings, 
  History, 
  User, 
  Users,
  Play,
  Pause,
  Download,
  Share,
  Brain,
  Eye,
  Smile
} from 'lucide-react';
import { 
  PERFORMANCE_THRESHOLDS, 
  VOICE_OPTIONS, 
  EMOTION_CATEGORIES,
  AGE_GROUPS 
} from '@/constants';
import { 
  PerformanceMetrics, 
  TranslationResult, 
  FaceData, 
  VoiceSettings,
  UserSettings 
} from '@/types';
import { io, Socket } from 'socket.io-client';

interface AdvancedCameraViewProps {
  isActive: boolean;
  onToggle: () => void;
  settings: UserSettings;
  onSettingsChange: (settings: UserSettings) => void;
}

export default function AdvancedCameraView({ 
  isActive, 
  onToggle, 
  settings, 
  onSettingsChange 
}: AdvancedCameraViewProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const audioRef = useRef<HTMLAudioElement>(null);
  const socketRef = useRef<Socket | null>(null);
  const frameIntervalRef = useRef<NodeJS.Timeout | null>(null);
  
  const [performance, setPerformance] = useState<PerformanceMetrics>({
    fps: 0,
    latency: 0,
    accuracy: 0,
    processingTime: 0,
    gpuUsage: 0,
    memoryUsage: 0
  });
  
  const [currentTranslation, setCurrentTranslation] = useState<string>("");
  const [confidence, setConfidence] = useState<number>(0);
  const [isRecording, setIsRecording] = useState<boolean>(false);
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const [translationHistory, setTranslationHistory] = useState<TranslationResult[]>([]);
  const [connectionStatus, setConnectionStatus] = useState<'disconnected' | 'connecting' | 'connected'>('disconnected');
  const [activeConnections, setActiveConnections] = useState<number>(0);
  
  // Face recognition states
  const [detectedFaces, setDetectedFaces] = useState<FaceData[]>([]);
  const [faceRecognitionEnabled, setFaceRecognitionEnabled] = useState(settings.faceRecognition.enabled);
  const [emotionDetectionEnabled, setEmotionDetectionEnabled] = useState(true);
  
  // Voice synthesis states
  const [voiceEnabled, setVoiceEnabled] = useState(settings.voiceSettings.enabled);
  const [currentVoice, setCurrentVoice] = useState(settings.voiceSettings.voice);
  const [isSpeaking, setIsSpeaking] = useState(false);
  
  // Advanced features
  const [showFaceOverlays, setShowFaceOverlays] = useState(true);
  const [showEmotionIndicators, setShowEmotionIndicators] = useState(true);
  const [showLandmarks, setShowLandmarks] = useState(false);

  // WebSocket connection
  useEffect(() => {
    if (isActive) {
      connectWebSocket();
    } else {
      disconnectWebSocket();
    }

    return () => disconnectWebSocket();
  }, [isActive]);

  const connectWebSocket = () => {
    setConnectionStatus('connecting');
    
    socketRef.current = io("/?XTransformPort=3002", {
      transports: ['websocket'],
      upgrade: false
    });

    socketRef.current.on('connect', () => {
      console.log('Connected to advanced WebSocket service');
      setConnectionStatus('connected');
    });

    socketRef.current.on('disconnect', () => {
      console.log('Disconnected from WebSocket service');
      setConnectionStatus('disconnected');
    });

    socketRef.current.on('translation_result', (result: TranslationResult) => {
      handleTranslationResult(result);
    });

    socketRef.current.on('face_recognition_result', (faces: FaceData[]) => {
      setDetectedFaces(faces);
    });

    socketRef.current.on('voice_synthesis_result', (audioData: any) => {
      handleVoiceSynthesis(audioData);
    });

    socketRef.current.on('performance_metrics', (metrics: any) => {
      setPerformance(prev => ({ ...prev, ...metrics }));
      setActiveConnections(metrics.activeConnections || 0);
    });

    socketRef.current.on('error', (error: any) => {
      console.error('WebSocket error:', error);
    });
  };

  const disconnectWebSocket = () => {
    if (socketRef.current) {
      socketRef.current.disconnect();
      socketRef.current = null;
    }
    setConnectionStatus('disconnected');
    
    if (frameIntervalRef.current) {
      clearInterval(frameIntervalRef.current);
      frameIntervalRef.current = null;
    }
  };

  const handleTranslationResult = (result: TranslationResult) => {
    setCurrentTranslation(result.translatedText);
    setConfidence(result.confidence || 0.8);
    
    if (result.performance) {
      setPerformance(prev => ({ ...prev, ...result.performance }));
    }

    // Add to history
    setTranslationHistory(prev => [result, ...prev.slice(0, 9)]);

    // Trigger voice synthesis if enabled
    if (voiceEnabled && result.translatedText) {
      synthesizeSpeech(result.translatedText);
    }
  };

  const handleVoiceSynthesis = (audioData: any) => {
    if (audioRef.current && audioData.audioUrl) {
      audioRef.current.src = audioData.audioUrl;
      audioRef.current.play().catch(error => {
        console.warn('Audio play failed:', error);
        setIsSpeaking(false);
      });
      setIsSpeaking(true);
      
      audioRef.current.onended = () => {
        setIsSpeaking(false);
      };
    }
  };

  const synthesizeSpeech = async (text: string) => {
    try {
      const response = await fetch('/api/voice-synthesis', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text,
          voice: currentVoice,
          speed: settings.voiceSettings.speed,
          pitch: settings.voiceSettings.pitch,
          volume: settings.voiceSettings.volume,
          language: settings.voiceSettings.language
        })
      });
      
      const audioData = await response.json();
      handleVoiceSynthesis(audioData);
    } catch (error) {
      console.error('Voice synthesis failed:', error);
    }
  };

  useEffect(() => {
    if (isActive && videoRef.current) {
      startCamera();
    } else {
      stopCamera();
    }
    return () => stopCamera();
  }, [isActive, connectionStatus]);

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { 
          width: { ideal: 1280 },
          height: { ideal: 720 },
          frameRate: { ideal: 30 }
        } 
      });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.play().catch(error => {
          console.warn('Video play failed:', error);
        });
        
        if (connectionStatus === 'connected') {
          startFrameCapture();
        }
      }
    } catch (error) {
      console.error('Error accessing camera:', error);
    }
  };

  const stopCamera = () => {
    if (videoRef.current && videoRef.current.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream;
      stream.getTracks().forEach(track => track.stop());
      videoRef.current.srcObject = null;
    }
    stopFrameCapture();
  };

  const startFrameCapture = () => {
    if (frameIntervalRef.current) return;
    
    frameIntervalRef.current = setInterval(() => {
      if (videoRef.current && socketRef.current && connectionStatus === 'connected') {
        const canvas = document.createElement('canvas');
        canvas.width = 640;
        canvas.height = 480;
        const ctx = canvas.getContext('2d');
        if (ctx) {
          ctx.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);
          const imageData = canvas.toDataURL('image/jpeg', 0.8);
          
          socketRef.current.emit('advanced_frame', {
            image: imageData,
            language: settings.preferredLanguage,
            settings: {
              ...settings,
              faceRecognition: faceRecognitionEnabled,
              emotionDetection: emotionDetectionEnabled,
              voiceSynthesis: voiceEnabled
            }
          });
        }
      }
    }, 1000 / 15); // 15 FPS for advanced processing
  };

  const stopFrameCapture = () => {
    if (frameIntervalRef.current) {
      clearInterval(frameIntervalRef.current);
      frameIntervalRef.current = null;
    }
  };

  const drawFaceOverlays = () => {
    if (!canvasRef.current || !showFaceOverlays) return;
    
    const ctx = canvasRef.current.getContext('2d');
    if (!ctx) return;

    ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);

    detectedFaces.forEach((face, index) => {
      const { boundingBox, gender, age, emotion, confidence } = face;
      
      // Draw bounding box
      ctx.strokeStyle = gender === 'male' ? '#3B82F6' : '#EC4899';
      ctx.lineWidth = 2;
      ctx.strokeRect(boundingBox.x, boundingBox.y, boundingBox.width, boundingBox.height);
      
      // Draw info background
      ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
      ctx.fillRect(boundingBox.x, boundingBox.y - 60, boundingBox.width, 60);
      
      // Draw text info
      ctx.fillStyle = '#FFFFFF';
      ctx.font = '14px Arial';
      ctx.fillText(`${gender === 'male' ? '👨' : '👩'} ${age}y`, boundingBox.x + 5, boundingBox.y - 35);
      ctx.fillText(`${(confidence * 100).toFixed(1)}%`, boundingBox.x + 5, boundingBox.y - 15);
      
      // Draw emotion if enabled
      if (showEmotionIndicators && emotion) {
        const emotionData = EMOTION_CATEGORIES.find(e => e.key === emotion.primary);
        if (emotionData) {
          ctx.fillText(`${emotionData.icon} ${emotionData.label}`, boundingBox.x + 5, boundingBox.y + 20);
        }
      }
      
      // Draw landmarks if enabled
      if (showLandmarks && face.landmarks) {
        ctx.fillStyle = '#10B981';
        face.landmarks.forEach(point => {
          ctx.beginPath();
          ctx.arc(point[0], point[1], 2, 0, 2 * Math.PI);
          ctx.fill();
        });
      }
    });
  };

  useEffect(() => {
    drawFaceOverlays();
  }, [detectedFaces, showFaceOverlays, showEmotionIndicators, showLandmarks]);

  const getPerformanceColor = (value: number, threshold: number, isHigherBetter: boolean = true) => {
    const ratio = isHigherBetter ? value / threshold : threshold / value;
    if (ratio >= 0.9) return 'text-green-400';
    if (ratio >= 0.7) return 'text-yellow-400';
    return 'text-red-400';
  };

  const getEmotionColor = (emotion: string) => {
    const emotionData = EMOTION_CATEGORIES.find(e => e.key === emotion);
    return emotionData?.color || '#6B7280';
  };

  return (
    <div className="space-y-6">
      {/* Advanced Camera View */}
      <Card className="bg-black/30 backdrop-blur-md border-purple-500/20">
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle className="text-white flex items-center gap-2">
              <Camera className="w-5 h-5" />
              Advanced Neural Camera
            </CardTitle>
            <div className="flex items-center gap-2">
              <Badge className={
                connectionStatus === 'connected' ? "bg-green-600" : 
                connectionStatus === 'connecting' ? "bg-yellow-600" : "bg-red-600"
              }>
                {connectionStatus === 'connected' ? "Connected" : 
                 connectionStatus === 'connecting' ? "Connecting..." : "Disconnected"}
              </Badge>
              <Badge className={isActive ? "bg-green-600" : "bg-red-600"}>
                {isActive ? "Active" : "Inactive"}
              </Badge>
              <Button
                variant={isActive ? "destructive" : "default"}
                size="sm"
                onClick={onToggle}
              >
                {isActive ? <CameraOff className="w-4 h-4" /> : <Camera className="w-4 h-4" />}
              </Button>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <div className="relative aspect-video bg-black rounded-lg overflow-hidden">
            <video
              ref={videoRef}
              className="w-full h-full object-cover"
              autoPlay
              playsInline
              muted
            />
            <canvas
              ref={canvasRef}
              width={640}
              height={480}
              className="absolute inset-0 w-full h-full object-cover pointer-events-none"
            />
            
            {/* Face Count Overlay */}
            {detectedFaces.length > 0 && (
              <div className="absolute top-4 left-4 space-y-2">
                <div className="bg-black/70 backdrop-blur-sm rounded-lg px-3 py-2">
                  <div className="flex items-center gap-2 text-white">
                    <Users className="w-4 h-4" />
                    <span className="text-sm font-medium">{detectedFaces.length} Face{detectedFaces.length > 1 ? 's' : ''} Detected</span>
                  </div>
                </div>
              </div>
            )}

            {/* Voice Status */}
            {isSpeaking && (
              <div className="absolute top-4 right-4">
                <div className="bg-black/70 backdrop-blur-sm rounded-lg px-3 py-2">
                  <div className="flex items-center gap-2 text-green-400">
                    <Volume2 className="w-4 h-4 animate-pulse" />
                    <span className="text-sm font-medium">Speaking</span>
                  </div>
                </div>
              </div>
            )}

            {/* Translation Overlay */}
            <div className="absolute bottom-4 left-4 right-4">
              <div className="bg-black/70 backdrop-blur-sm rounded-lg px-4 py-3">
                <div className="flex items-center justify-between">
                  <div className="flex-1">
                    <div className="text-white text-sm mb-1">Translation</div>
                    <div className="text-lg font-medium text-cyan-400">
                      {currentTranslation || "Waiting for sign..."}
                    </div>
                  </div>
                  <div className="ml-4 text-right">
                    <div className="text-white text-sm mb-1">Confidence</div>
                    <div className={`text-sm font-medium ${getPerformanceColor(confidence, 0.85)}`}>
                      {(confidence * 100).toFixed(1)}%
                    </div>
                  </div>
                </div>
                <Progress value={confidence * 100} className="mt-2 h-2" />
              </div>
            </div>
          </div>

          {/* Advanced Controls */}
          <div className="flex items-center justify-center gap-4 mt-4">
            <Button
              variant={isRecording ? "destructive" : "outline"}
              size="sm"
              onClick={() => setIsRecording(!isRecording)}
            >
              <Mic className={`w-4 h-4 ${isRecording ? 'animate-pulse' : ''}`} />
            </Button>
            
            <Button
              variant={isPlaying ? "destructive" : "outline"}
              size="sm"
              onClick={() => setIsPlaying(!isPlaying)}
            >
              {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
            </Button>
            
            <Button
              variant={voiceEnabled ? "default" : "outline"}
              size="sm"
              onClick={() => setVoiceEnabled(!voiceEnabled)}
            >
              <Volume2 className="w-4 h-4" />
            </Button>
            
            <Button
              variant={faceRecognitionEnabled ? "default" : "outline"}
              size="sm"
              onClick={() => setFaceRecognitionEnabled(!faceRecognitionEnabled)}
            >
              <Eye className="w-4 h-4" />
            </Button>
            
            <Button
              variant={emotionDetectionEnabled ? "default" : "outline"}
              size="sm"
              onClick={() => setEmotionDetectionEnabled(!emotionDetectionEnabled)}
            >
              <Smile className="w-4 h-4" />
            </Button>
            
            <Button variant="outline" size="sm">
              <Settings className="w-4 h-4" />
            </Button>
            
            <Button variant="outline" size="sm">
              <History className="w-4 h-4" />
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Voice Settings */}
      <Card className="bg-black/30 backdrop-blur-md border-purple-500/20">
        <CardHeader>
          <CardTitle className="text-white flex items-center gap-2">
            <Volume2 className="w-5 h-5" />
            Voice Synthesis Settings
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="text-white text-sm font-medium mb-2 block">Voice Selection</label>
              <Select value={currentVoice} onValueChange={setCurrentVoice}>
                <SelectTrigger className="bg-black/20 border-gray-600 text-white">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent className="bg-black/90 border-gray-600">
                  {VOICE_OPTIONS.map((voice) => (
                    <SelectItem key={voice.id} value={voice.id} className="text-white hover:bg-purple-600/20">
                      <div className="flex items-center gap-2">
                        <span>{voice.name}</span>
                        <span className="text-gray-400 text-sm">({voice.accent})</span>
                      </div>
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            
            <div>
              <label className="text-white text-sm font-medium mb-2 block">
                Speed: {settings.voiceSettings.speed.toFixed(1)}x
              </label>
              <Slider
                value={[settings.voiceSettings.speed]}
                onValueChange={([value]) => onSettingsChange({
                  ...settings,
                  voiceSettings: { ...settings.voiceSettings, speed: value }
                })}
                max={2.0}
                min={0.5}
                step={0.1}
                className="w-full"
              />
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Face Recognition Results */}
      {detectedFaces.length > 0 && (
        <Card className="bg-black/30 backdrop-blur-md border-purple-500/20">
          <CardHeader>
            <CardTitle className="text-white flex items-center gap-2">
              <Users className="w-5 h-5" />
              Face Recognition Results
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {detectedFaces.map((face, index) => (
                <div key={face.id} className="bg-black/20 rounded-lg p-4 border border-gray-600">
                  <div className="flex items-center justify-between mb-3">
                    <div className="flex items-center gap-2">
                      <div className={`w-8 h-8 rounded-full flex items-center justify-center ${
                        face.gender === 'male' ? 'bg-blue-500' : 
                        face.gender === 'female' ? 'bg-pink-500' : 'bg-gray-500'
                      }`}>
                        {face.gender === 'male' ? '👨' : face.gender === 'female' ? '👩' : '❓'}
                      </div>
                      <div>
                        <div className="text-white font-medium">
                          {face.name || `Person ${index + 1}`}
                        </div>
                        <div className="text-gray-400 text-sm">
                          {face.gender === 'male' ? 'Male' : face.gender === 'female' ? 'Female' : 'Unknown'} • {face.age} years
                        </div>
                      </div>
                    </div>
                    <Badge className={face.isKnown ? "bg-green-600" : "bg-yellow-600"}>
                      {face.isKnown ? "Known" : "Unknown"}
                    </Badge>
                  </div>
                  
                  {face.emotion && (
                    <div className="mb-3">
                      <div className="flex items-center gap-2 mb-2">
                        <span className="text-white text-sm">Emotion:</span>
                        <span 
                          className="text-sm font-medium px-2 py-1 rounded"
                          style={{ backgroundColor: getEmotionColor(face.emotion.primary) + '20', color: getEmotionColor(face.emotion.primary) }}
                        >
                          {EMOTION_CATEGORIES.find(e => e.key === face.emotion.primary)?.icon} {face.emotion.primary}
                        </span>
                        <span className="text-gray-400 text-sm">
                          ({(face.emotion.confidence * 100).toFixed(1)}%)
                        </span>
                      </div>
                      <div className="grid grid-cols-4 gap-1 text-xs">
                        {Object.entries(face.emotion.all).map(([emotion, value]) => (
                          <div key={emotion} className="text-center">
                            <div className="text-gray-400 capitalize">{emotion}</div>
                            <div className="text-white font-medium">{(value * 100).toFixed(0)}%</div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                  
                  <div className="text-center">
                    <div className="text-gray-400 text-sm">Confidence</div>
                    <div className="text-white font-medium">{(face.confidence * 100).toFixed(1)}%</div>
                    <Progress value={face.confidence * 100} className="mt-1 h-1" />
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Performance Metrics */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <Card className="bg-black/30 backdrop-blur-md border-purple-500/20">
          <CardContent className="p-4 text-center">
            <div className={`text-2xl font-bold ${getPerformanceColor(performance.fps, PERFORMANCE_THRESHOLDS.TARGET_FPS)}`}>
              {performance.fps.toFixed(1)}
            </div>
            <div className="text-gray-400 text-sm">FPS</div>
          </CardContent>
        </Card>

        <Card className="bg-black/30 backdrop-blur-md border-purple-500/20">
          <CardContent className="p-4 text-center">
            <div className={`text-2xl font-bold ${getPerformanceColor(performance.latency, PERFORMANCE_THRESHOLDS.MAX_LATENCY, false)}`}>
              {performance.latency.toFixed(0)}ms
            </div>
            <div className="text-gray-400 text-sm">Latency</div>
          </CardContent>
        </Card>

        <Card className="bg-black/30 backdrop-blur-md border-purple-500/20">
          <CardContent className="p-4 text-center">
            <div className={`text-2xl font-bold ${getPerformanceColor(performance.accuracy, PERFORMANCE_THRESHOLDS.TARGET_ACCURACY)}`}>
              {(performance.accuracy * 100).toFixed(1)}%
            </div>
            <div className="text-gray-400 text-sm">Accuracy</div>
          </CardContent>
        </Card>

        <Card className="bg-black/30 backdrop-blur-md border-purple-500/20">
          <CardContent className="p-4 text-center">
            <div className="text-2xl font-bold text-cyan-400">
              {detectedFaces.length}
            </div>
            <div className="text-gray-400 text-sm">Faces</div>
          </CardContent>
        </Card>
      </div>

      {/* Hidden audio element for voice synthesis */}
      <audio ref={audioRef} className="hidden" />
    </div>
  );
}