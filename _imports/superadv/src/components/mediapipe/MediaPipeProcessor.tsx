'use client';

import { useState, useEffect, useRef } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Slider } from '@/components/ui/slider';
import { Switch } from '@/components/ui/switch';
import { 
  Activity, 
  Zap, 
  Eye, 
  Hand, 
  User, 
  Settings, 
  Play, 
  Pause,
  Cpu,
  Brain,
  Target
} from 'lucide-react';
import { 
  MEDIAPIPE_CONFIGURATIONS, 
  PERFORMANCE_THRESHOLDS, 
  THEME_COLORS 
} from '@/constants';
import { 
  MediaPipeSettings, 
  PerformanceMetrics, 
  HandLandmark, 
  PoseLandmark, 
  Point3D 
} from '@/types';

interface MediaPipeProcessorProps {
  isActive: boolean;
  settings: MediaPipeSettings;
  onSettingsChange: (settings: MediaPipeSettings) => void;
  onPerformanceUpdate: (metrics: PerformanceMetrics) => void;
}

export default function MediaPipeProcessor({ 
  isActive, 
  settings, 
  onSettingsChange, 
  onPerformanceUpdate 
}: MediaPipeProcessorProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const animationRef = useRef<number>();
  
  const [isInitialized, setIsInitialized] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [handLandmarks, setHandLandmarks] = useState<HandLandmark[]>([]);
  const [poseLandmarks, setPoseLandmarks] = useState<PoseLandmark[]>([]);
  const [faceMesh, setFaceMesh] = useState<Point3D[]>([]);
  const [segmentationMask, setSegmentationMask] = useState<ImageData | null>(null);
  
  const [performance, setPerformance] = useState<PerformanceMetrics>({
    fps: 0,
    latency: 0,
    accuracy: 0,
    processingTime: 0,
    mediaPipeLatency: 0
  });

  const [detectionCounts, setDetectionCounts] = useState({
    hands: 0,
    poses: 1,
    faces: 0,
    landmarks: 0
  });

  // Initialize MediaPipe
  useEffect(() => {
    if (isActive) {
      initializeMediaPipe();
    } else {
      cleanupMediaPipe();
    }
    return () => cleanupMediaPipe();
  }, [isActive]);

  const initializeMediaPipe = async () => {
    try {
      console.log('🚀 Initializing MediaPipe with advanced configurations...');
      
      // Simulate MediaPipe initialization
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      setIsInitialized(true);
      startProcessing();
      
      console.log('✅ MediaPipe initialized successfully');
      console.log('📊 Configuration:', {
        modelComplexity: settings.modelComplexity,
        minDetectionConfidence: settings.minDetectionConfidence,
        trackingMode: settings.trackingMode,
        smoothLandmarks: settings.smoothLandmarks,
        enableSegmentation: settings.enableSegmentation
      });
    } catch (error) {
      console.error('❌ MediaPipe initialization failed:', error);
    }
  };

  const cleanupMediaPipe = () => {
    if (animationRef.current) {
      cancelAnimationFrame(animationRef.current);
    }
    setIsInitialized(false);
    setIsProcessing(false);
  };

  const startProcessing = () => {
    setIsProcessing(true);
    processFrame();
  };

  const processFrame = () => {
    const startTime = performance.now();
    
    // Simulate MediaPipe processing
    const mockHandLandmarks = generateMockHandLandmarks();
    const mockPoseLandmarks = generateMockPoseLandmarks();
    const mockFaceMesh = generateMockFaceMesh();
    const mockSegmentation = generateMockSegmentation();
    
    setHandLandmarks(mockHandLandmarks);
    setPoseLandmarks(mockPoseLandmarks);
    setFaceMesh(mockFaceMesh);
    setSegmentationMask(mockSegmentation);
    
    // Update detection counts
    setDetectionCounts({
      hands: mockHandLandmarks.length,
      poses: mockPoseLandmarks.length > 0 ? 1 : 0,
      faces: mockFaceMesh.length > 0 ? 1 : 0,
      landmarks: mockPoseLandmarks.length + mockHandLandmarks.length * 21
    });
    
    // Calculate performance metrics
    const processingTime = performance.now() - startTime;
    const fps = 1000 / processingTime;
    const latency = processingTime;
    const accuracy = 0.85 + Math.random() * 0.1;
    
    const newPerformance = {
      fps,
      latency,
      accuracy,
      processingTime,
      mediaPipeLatency: processingTime
    };
    
    setPerformance(newPerformance);
    onPerformanceUpdate(newPerformance);
    
    if (isProcessing) {
      animationRef.current = requestAnimationFrame(processFrame);
    }
  };

  const generateMockHandLandmarks = (): HandLandmark[] => {
    const landmarks: HandLandmark[] = [];
    const numHands = settings.trackingMode === 'multi' ? 2 : (Math.random() > 0.5 ? 1 : 0);
    
    for (let hand = 0; hand < numHands; hand++) {
      for (let i = 0; i < 21; i++) {
        landmarks.push({
          id: i,
          x: 200 + hand * 200 + Math.random() * 100,
          y: 150 + Math.random() * 100,
          z: Math.random() * 50,
          visibility: 0.8 + Math.random() * 0.2,
          presence: Math.random() > 0.1 ? 1 : 0
        });
      }
    }
    
    return landmarks;
  };

  const generateMockPoseLandmarks = (): PoseLandmark[] => {
    const poseNames = [
      'nose', 'left_eye_inner', 'left_eye', 'left_eye_outer', 'right_eye_inner',
      'right_eye', 'right_eye_outer', 'left_ear', 'right_ear', 'mouth_left',
      'mouth_right', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
      'left_wrist', 'right_wrist', 'left_pinky', 'left_index', 'left_thumb',
      'right_pinky', 'right_index', 'right_thumb'
    ];
    
    return poseNames.map((name, index) => ({
      id: index,
      name,
      position: {
        x: 320 + Math.random() * 100,
        y: 100 + Math.random() * 200,
        z: Math.random() * 50
      },
      visibility: 0.7 + Math.random() * 0.3,
      confidence: 0.6 + Math.random() * 0.4
    }));
  };

  const generateMockFaceMesh = (): Point3D[] => {
    const vertices: Point3D[] = [];
    const numVertices = settings.modelComplexity === 'heavy' ? 468 : 468;
    
    for (let i = 0; i < numVertices; i++) {
      vertices.push({
        x: 300 + Math.random() * 200,
        y: 100 + Math.random() * 200,
        z: Math.random() * 100
      });
    }
    
    return vertices;
  };

  const generateMockSegmentation = (): ImageData => {
    // Create a mock segmentation mask
    const canvas = document.createElement('canvas');
    canvas.width = 640;
    canvas.height = 480;
    const ctx = canvas.getContext('2d');
    
    if (ctx) {
      // Create gradient background
      const gradient = ctx.createLinearGradient(0, 0, 640, 480);
      gradient.addColorStop(0, 'rgba(0, 255, 0, 0.3)');
      gradient.addColorStop(0.5, 'rgba(0, 0, 255, 0.2)');
      gradient.addColorStop(1, 'rgba(255, 0, 0, 0.1)');
      
      ctx.fillStyle = gradient;
      ctx.fillRect(0, 0, 640, 480);
      
      // Add some segmentation regions
      ctx.fillStyle = 'rgba(255, 255, 255, 0.4)';
      ctx.fillRect(200, 150, 150, 200);
      
      return ctx.getImageData(0, 0, 640, 480);
    }
    
    return new ImageData(640, 480);
  };

  const drawLandmarks = (ctx: CanvasRenderingContext2D) => {
    // Draw pose landmarks
    poseLandmarks.forEach(landmark => {
      if (landmark.visibility > 0.5) {
        ctx.fillStyle = '#10B981';
        ctx.beginPath();
        ctx.arc(landmark.position.x, landmark.position.y, 4, 0, Math.PI * 2);
        ctx.fill();
      }
    });
    
    // Draw hand landmarks with connections
    if (handLandmarks.length > 0) {
      ctx.strokeStyle = '#8B5CF6';
      ctx.lineWidth = 2;
      
      handLandmarks.forEach(landmark => {
        if (landmark.presence > 0.5) {
          ctx.fillStyle = '#EC4899';
          ctx.beginPath();
          ctx.arc(landmark.x, landmark.y, 3, 0, Math.PI * 2);
          ctx.fill();
        }
      });
      
      // Draw hand connections
      drawHandConnections(ctx, handLandmarks);
    }
    
    // Draw face mesh
    if (faceMesh.length > 0) {
      ctx.strokeStyle = '#06B6D4';
      ctx.fillStyle = 'rgba(6, 182, 212, 0.1)';
      
      // Draw mesh triangles (simplified)
      for (let i = 0; i < faceMesh.length - 2; i += 3) {
        ctx.beginPath();
        ctx.moveTo(faceMesh[i].x, faceMesh[i].y);
        ctx.lineTo(faceMesh[i + 1].x, faceMesh[i + 1].y);
        ctx.lineTo(faceMesh[i + 2].x, faceMesh[i + 2].y);
        ctx.closePath();
        ctx.fill();
        ctx.stroke();
      }
    }
  };

  const drawHandConnections = (ctx: CanvasRenderingContext2D, landmarks: HandLandmark[]) => {
    const connections = [
      [0, 1], [1, 2], [2, 3], [3, 4], // Thumb
      [0, 5], [5, 6], [6, 7], [7, 8], // Index finger
      [0, 9], [9, 10], [10, 11], [11, 12], // Middle finger
      [0, 13], [13, 14], [14, 15], [15, 16], // Ring finger
      [0, 17], [17, 18], [18, 19], [19, 20] // Pinky
    ];
    
    ctx.strokeStyle = '#8B5CF6';
    ctx.lineWidth = 1;
    
    connections.forEach(([start, end]) => {
      if (landmarks[start] && landmarks[end] && 
          landmarks[start].presence > 0.5 && landmarks[end].presence > 0.5) {
        ctx.beginPath();
        ctx.moveTo(landmarks[start].x, landmarks[start].y);
        ctx.lineTo(landmarks[end].x, landmarks[end].y);
        ctx.stroke();
      }
    });
  };

  useEffect(() => {
    if (canvasRef.current && isInitialized) {
      const ctx = canvasRef.current.getContext('2d');
      if (ctx) {
        ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
        
        // Draw segmentation mask if enabled
        if (settings.enableSegmentation && segmentationMask) {
          ctx.putImageData(segmentationMask, 0, 0);
        }
        
        // Draw landmarks on top
        drawLandmarks(ctx);
      }
    }
  }, [handLandmarks, poseLandmarks, faceMesh, segmentationMask, isInitialized]);

  const updateSetting = <K extends keyof MediaPipeSettings>(key: K, value: MediaPipeSettings[K]) => {
    const newSettings = { ...settings, [key]: value };
    onSettingsChange(newSettings);
  };

  const getPerformanceColor = (value: number, threshold: number, isHigherBetter: boolean = true) => {
    const ratio = isHigherBetter ? value / threshold : threshold / value;
    if (ratio >= 0.9) return 'text-green-400';
    if (ratio >= 0.7) return 'text-yellow-400';
    return 'text-red-400';
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <Card className="bg-black/30 backdrop-blur-md border-purple-500/20">
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle className="text-white flex items-center gap-2">
              <Activity className="w-5 h-5" />
              MediaPipe Neural Processor
            </CardTitle>
            <div className="flex items-center gap-2">
              <Badge className={isInitialized ? "bg-green-600" : "bg-yellow-600"}>
                {isInitialized ? "Initialized" : "Initializing..."}
              </Badge>
              <Badge className={isProcessing ? "bg-blue-600" : "bg-gray-600"}>
                {isProcessing ? "Processing" : "Idle"}
              </Badge>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-center">
            <div>
              <div className={`text-2xl font-bold ${getPerformanceColor(performance.fps, PERFORMANCE_THRESHOLDS.TARGET_FPS)}`}>
                {performance.fps.toFixed(1)}
              </div>
              <div className="text-gray-400 text-sm">FPS</div>
            </div>
            <div>
              <div className={`text-2xl font-bold ${getPerformanceColor(performance.latency, PERFORMANCE_THRESHOLDS.MAX_LATENCY, false)}`}>
                {performance.latency.toFixed(0)}ms
              </div>
              <div className="text-gray-400 text-sm">Latency</div>
            </div>
            <div>
              <div className={`text-2xl font-bold ${getPerformanceColor(performance.accuracy, PERFORMANCE_THRESHOLDS.TARGET_ACCURACY)}`}>
                {(performance.accuracy * 100).toFixed(1)}%
              </div>
              <div className="text-gray-400 text-sm">Accuracy</div>
            </div>
            <div>
              <div className="text-2xl font-bold text-cyan-400">
                {detectionCounts.landmarks}
              </div>
              <div className="text-gray-400 text-sm">Landmarks</div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Visualization */}
      <Card className="bg-black/30 backdrop-blur-md border-purple-500/20">
        <CardHeader>
          <CardTitle className="text-white flex items-center gap-2">
            <Eye className="w-5 h-5" />
            Neural Visualization
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="relative aspect-video bg-black rounded-lg overflow-hidden">
            <canvas
              ref={canvasRef}
              width={640}
              height={480}
              className="w-full h-full object-cover"
            />
            
            {/* Detection Info Overlay */}
            <div className="absolute top-4 left-4 space-y-2">
              <div className="bg-black/70 backdrop-blur-sm rounded-lg px-3 py-2">
                <div className="flex items-center gap-2 text-white">
                  <Hand className="w-4 h-4" />
                  <span className="text-sm font-medium">{detectionCounts.hands} Hand{detectionCounts.hands !== 1 ? 's' : ''}</span>
                </div>
              </div>
              <div className="bg-black/70 backdrop-blur-sm rounded-lg px-3 py-2">
                <div className="flex items-center gap-2 text-white">
                  <User className="w-4 h-4" />
                  <span className="text-sm font-medium">{detectionCounts.poses} Pose{detectionCounts.poses !== 1 ? 's' : ''}</span>
                </div>
              </div>
              <div className="bg-black/70 backdrop-blur-sm rounded-lg px-3 py-2">
                <div className="flex items-center gap-2 text-white">
                  <Eye className="w-4 h-4" />
                  <span className="text-sm font-medium">{detectionCounts.faces} Face{detectionCounts.faces !== 1 ? 's' : ''}</span>
                </div>
              </div>
            </div>

            {/* Performance Overlay */}
            <div className="absolute top-4 right-4">
              <div className="bg-black/70 backdrop-blur-sm rounded-lg px-3 py-2">
                <div className="text-cyan-400 text-sm font-medium">
                  MediaPipe Pipeline
                </div>
                <div className="text-white text-xs">
                  {settings.modelComplexity} • {settings.trackingMode}
                </div>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Controls */}
      <Card className="bg-black/30 backdrop-blur-md border-purple-500/20">
        <CardHeader>
          <CardTitle className="text-white flex items-center gap-2">
            <Settings className="w-5 h-5" />
            Pipeline Configuration
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="text-white text-sm font-medium mb-2 block">Model Complexity</label>
              <Select 
                value={settings.modelComplexity} 
                onValueChange={(value: any) => updateSetting('modelComplexity', value)}
              >
                <SelectTrigger className="bg-black/20 border-gray-600 text-white">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent className="bg-black/90 border-gray-600">
                  <SelectItem value="lite" className="text-white hover:bg-purple-600/20">Lite</SelectItem>
                  <SelectItem value="full" className="text-white hover:bg-purple-600/20">Full</SelectItem>
                  <SelectItem value="heavy" className="text-white hover:bg-purple-600/20">Heavy</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div>
              <label className="text-white text-sm font-medium mb-2 block">Tracking Mode</label>
              <Select 
                value={settings.trackingMode} 
                onValueChange={(value: any) => updateSetting('trackingMode', value)}
              >
                <SelectTrigger className="bg-black/20 border-gray-600 text-white">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent className="bg-black/90 border-gray-600">
                  <SelectItem value="single" className="text-white hover:bg-purple-600/20">Single</SelectItem>
                  <SelectItem value="multi" className="text-white hover:bg-purple-600/20">Multi</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div>
              <label className="text-white text-sm font-medium mb-2 block">
                Min Confidence: {(settings.minDetectionConfidence * 100).toFixed(0)}%
              </label>
              <Slider
                value={[settings.minDetectionConfidence]}
                onValueChange={([value]) => updateSetting('minDetectionConfidence', value)}
                max={1}
                min={0.1}
                step={0.1}
                className="w-full"
              />
            </div>

            <div>
              <label className="text-white text-sm font-medium mb-2 block">Smooth Landmarks</label>
              <Switch
                checked={settings.smoothLandmarks}
                onCheckedChange={(checked) => updateSetting('smoothLandmarks', checked)}
              />
            </div>

            <div>
              <label className="text-white text-sm font-medium mb-2 block">Enable Segmentation</label>
              <Switch
                checked={settings.enableSegmentation}
                onCheckedChange={(checked) => updateSetting('enableSegmentation', checked)}
              />
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Pipeline Status */}
      <Card className="bg-black/30 backdrop-blur-md border-purple-500/20">
        <CardHeader>
          <CardTitle className="text-white flex items-center gap-2">
            <Cpu className="w-5 h-5" />
            Pipeline Status
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <span className="text-gray-400">Holistic Processing</span>
              <Badge className="bg-green-600">Active</Badge>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-gray-400">Face Mesh</span>
              <Badge className={settings.enableSegmentation ? "bg-green-600" : "bg-gray-600"}>
                {settings.enableSegmentation ? "Active" : "Disabled"}
              </Badge>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-gray-400">Hand Tracking</span>
              <Badge className="bg-green-600">Active</Badge>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-gray-400">Pose Estimation</span>
              <Badge className="bg-green-600">Active</Badge>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}