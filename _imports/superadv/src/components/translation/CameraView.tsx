'use client';

import { useState, useEffect, useRef } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Camera, CameraOff, Mic, MicOff, Volume2, Settings, History } from 'lucide-react';
import { PERFORMANCE_THRESHOLDS } from '@/constants';
import { PerformanceMetrics, TranslationResult } from '@/types';
import { io, Socket } from 'socket.io-client';

interface CameraViewProps {
  isActive: boolean;
  onToggle: () => void;
}

export default function CameraView({ isActive, onToggle }: CameraViewProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const socketRef = useRef<Socket | null>(null);
  const frameIntervalRef = useRef<NodeJS.Timeout | null>(null);
  
  const [performance, setPerformance] = useState<PerformanceMetrics>({
    fps: 0,
    latency: 0,
    accuracy: 0,
    processingTime: 0
  });
  const [currentTranslation, setCurrentTranslation] = useState<string>("");
  const [confidence, setConfidence] = useState<number>(0);
  const [isRecording, setIsRecording] = useState<boolean>(false);
  const [isMuted, setIsMuted] = useState<boolean>(false);
  const [translationHistory, setTranslationHistory] = useState<TranslationResult[]>([]);
  const [connectionStatus, setConnectionStatus] = useState<'disconnected' | 'connecting' | 'connected'>('disconnected');
  const [activeConnections, setActiveConnections] = useState<number>(0);

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
    
    // Connect to WebSocket service
    socketRef.current = io("/?XTransformPort=3002", {
      transports: ['websocket'],
      upgrade: false
    });

    socketRef.current.on('connect', () => {
      console.log('Connected to WebSocket service');
      setConnectionStatus('connected');
    });

    socketRef.current.on('disconnect', () => {
      console.log('Disconnected from WebSocket service');
      setConnectionStatus('disconnected');
    });

    socketRef.current.on('translation_result', (result: TranslationResult) => {
      setCurrentTranslation(result.translatedText);
      setConfidence(result.confidence || 0.8);
      
      if (result.performance) {
        setPerformance({
          fps: result.performance.fps || 30,
          latency: result.performance.processingTime || 50,
          accuracy: result.confidence || 0.85,
          processingTime: result.performance.processingTime || 30
        });
      }

      // Add to history
      setTranslationHistory(prev => [result, ...prev.slice(0, 9)]);
    });

    socketRef.current.on('performance_metrics', (metrics: any) => {
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
        videoRef.current.play();
        
        // Start frame capture when WebSocket is connected
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
        // Capture frame from video
        const canvas = document.createElement('canvas');
        canvas.width = 640;
        canvas.height = 480;
        const ctx = canvas.getContext('2d');
        if (ctx) {
          ctx.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);
          const imageData = canvas.toDataURL('image/jpeg', 0.8);
          
          // Send frame to WebSocket service
          socketRef.current.emit('camera_frame', {
            image: imageData,
            language: 'ASL',
            settings: {
              quality: 0.8,
              frameRate: 30
            }
          });
        }
      }
    }, 1000 / 10); // Send 10 frames per second to manage load
  };

  const stopFrameCapture = () => {
    if (frameIntervalRef.current) {
      clearInterval(frameIntervalRef.current);
      frameIntervalRef.current = null;
    }
  };

  const getPerformanceColor = (value: number, threshold: number, isHigherBetter: boolean = true) => {
    const ratio = isHigherBetter ? value / threshold : threshold / value;
    if (ratio >= 0.9) return 'text-green-400';
    if (ratio >= 0.7) return 'text-yellow-400';
    return 'text-red-400';
  };

  return (
    <div className="space-y-6">
      {/* Camera View */}
      <Card className="bg-black/30 backdrop-blur-md border-purple-500/20">
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle className="text-white flex items-center gap-2">
              <Camera className="w-5 h-5" />
              Neural Camera Interface
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
              className="absolute inset-0 w-full h-full object-cover"
            />
            
            {/* Overlay Info */}
            <div className="absolute top-4 left-4 space-y-2">
              <div className="bg-black/70 backdrop-blur-sm rounded-lg px-3 py-2">
                <div className="text-green-400 text-sm font-medium">● LIVE</div>
                <div className="text-white text-xs">Neural Processing</div>
              </div>
            </div>

            <div className="absolute top-4 right-4 space-y-2">
              <div className="bg-black/70 backdrop-blur-sm rounded-lg px-3 py-2">
                <div className={`text-sm font-medium ${getPerformanceColor(performance.fps, PERFORMANCE_THRESHOLDS.TARGET_FPS)}`}>
                  {performance.fps.toFixed(1)} FPS
                </div>
                <div className="text-white text-xs">{performance.latency.toFixed(0)}ms</div>
              </div>
            </div>

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

          {/* Control Buttons */}
          <div className="flex items-center justify-center gap-4 mt-4">
            <Button
              variant={isRecording ? "destructive" : "outline"}
              size="sm"
              onClick={() => setIsRecording(!isRecording)}
            >
              <Mic className={`w-4 h-4 ${isRecording ? 'animate-pulse' : ''}`} />
            </Button>
            <Button
              variant={isMuted ? "destructive" : "outline"}
              size="sm"
              onClick={() => setIsMuted(!isMuted)}
            >
              <Volume2 className="w-4 h-4" />
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
              {performance.processingTime.toFixed(0)}ms
            </div>
            <div className="text-gray-400 text-sm">Processing</div>
          </CardContent>
        </Card>
      </div>

      {/* Translation History */}
      <Card className="bg-black/30 backdrop-blur-md border-purple-500/20">
        <CardHeader>
          <CardTitle className="text-white flex items-center gap-2">
            <History className="w-5 h-5" />
            Recent Translations
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-2 max-h-64 overflow-y-auto">
            {translationHistory.length === 0 ? (
              <div className="text-center text-gray-400 py-8">
                No translations yet. Start the camera to begin.
              </div>
            ) : (
              translationHistory.map((result) => (
                <div key={result.id} className="flex items-center justify-between p-3 bg-black/20 rounded-lg border border-gray-600">
                  <div className="flex-1">
                    <div className="text-white font-medium">{result.translatedText}</div>
                    <div className="text-gray-400 text-xs">{result.timestamp.toLocaleTimeString()}</div>
                  </div>
                  <Badge className="ml-2">
                    {(result.confidence * 100).toFixed(0)}%
                  </Badge>
                </div>
              ))
            )}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}