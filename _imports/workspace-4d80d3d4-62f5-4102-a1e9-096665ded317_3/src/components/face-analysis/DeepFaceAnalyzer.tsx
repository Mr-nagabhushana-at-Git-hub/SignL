'use client';

import { useState, useEffect, useRef } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { 
  Eye, 
  User, 
  Zap, 
  Settings, 
  Camera, 
  Shield, 
  Fingerprint,
  Activity,
  CheckCircle,
  AlertTriangle,
  Info
} from 'lucide-react';
import { 
  DEEP_FACE_FEATURES, 
  PERFORMANCE_THRESHOLDS, 
  THEME_COLORS 
} from '@/constants';
import { 
  FaceData, 
  BiometricData, 
  Point3D, 
  BiometricAuthentication,
  PerformanceMetrics 
} from '@/types';

interface DeepFaceAnalyzerProps {
  isActive: boolean;
  onAuthenticationUpdate: (auth: BiometricAuthentication) => void;
  onPerformanceUpdate: (metrics: PerformanceMetrics) => void;
}

export default function DeepFaceAnalyzer({ 
  isActive, 
  onAuthenticationUpdate,
  onPerformanceUpdate 
}: DeepFaceAnalyzerProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const animationRef = useRef<number>();
  
  const [isInitialized, setIsInitialized] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [detectedFaces, setDetectedFaces] = useState<FaceData[]>([]);
  const [selectedFace, setSelectedFace] = useState<FaceData | null>(null);
  const [authenticationMode, setAuthenticationMode] = useState<'detection' | 'recognition' | 'authentication'>('detection');
  
  const [performance, setPerformance] = useState<PerformanceMetrics>({
    fps: 0,
    latency: 0,
    accuracy: 0,
    processingTime: 0
  });

  const [systemStatus, setSystemStatus] = useState({
    faceDatabase: 1250,
    processingQueue: 0,
    antiSpoofingActive: true,
    livenessDetection: true
  });

  // Initialize Deep Face Analyzer
  useEffect(() => {
    if (isActive) {
      initializeDeepFaceAnalyzer();
    } else {
      cleanupDeepFaceAnalyzer();
    }
    return () => cleanupDeepFaceAnalyzer();
  }, [isActive]);

  const initializeDeepFaceAnalyzer = async () => {
    try {
      console.log('🧠 Initializing Deep Face Analyzer with biometric analysis...');
      
      // Simulate initialization
      await new Promise(resolve => setTimeout(resolve, 2000));
      
      setIsInitialized(true);
      startFaceAnalysis();
      
      console.log('✅ Deep Face Analyzer initialized successfully');
      console.log('🔍 Features:', {
        biometricAnalysis: DEEP_FACE_FEATURES.biometricAnalysis,
        faceShapes: DEEP_FACE_FEATURES.faceShapes,
        skinTones: DEEP_FACE_FEATURES.skinTones,
        qualityMetrics: DEEP_FACE_FEATURES.qualityMetrics
      });
    } catch (error) {
      console.error('❌ Deep Face Analyzer initialization failed:', error);
    }
  };

  const cleanupDeepFaceAnalyzer = () => {
    if (animationRef.current) {
      cancelAnimationFrame(animationRef.current);
    }
    setIsProcessing(false);
    setIsInitialized(false);
  };

  const startFaceAnalysis = () => {
    setIsProcessing(true);
    analyzeFrame();
  };

  const analyzeFrame = () => {
    const startTime = performance.now();
    
    // Simulate deep face analysis
    const mockFaces = generateMockFaceData();
    setDetectedFaces(mockFaces);
    
    // Calculate performance metrics
    const processingTime = performance.now() - startTime;
    const fps = 1000 / processingTime;
    const accuracy = 0.92 + Math.random() * 0.06;
    
    const newPerformance = {
      fps,
      latency: processingTime,
      accuracy,
      processingTime
    };
    
    setPerformance(newPerformance);
    onPerformanceUpdate(newPerformance);
    
    if (isProcessing) {
      setTimeout(() => analyzeFrame(), 100);
    }
  };

  const generateMockFaceData = (): FaceData[] => {
    const faces: FaceData[] = [];
    const numFaces = Math.floor(Math.random() * 3) + 1; // 1-3 faces
    
    for (let i = 0; i < numFaces; i++) {
      const biometrics = generateBiometricData();
      const faceData: FaceData = {
        id: `face_${Date.now()}_${i}`,
        name: Math.random() > 0.7 ? `Person ${i + 1}` : undefined,
        gender: biometrics.gender,
        age: biometrics.age,
        ageRange: getAgeRange(biometrics.age),
        confidence: 0.85 + Math.random() * 0.12,
        boundingBox: {
          x: 150 + i * 200 + Math.random() * 50,
          y: 100 + Math.random() * 100,
          width: 120 + Math.random() * 40,
          height: 150 + Math.random() * 30
        },
        landmarks: generateFacialLandmarks(),
        mesh3D: generate3DMesh(),
        emotion: generateEmotionData(),
        biometrics,
        timestamp: new Date(),
        isKnown: Math.random() > 0.6,
        faceEmbedding: generateFaceEmbedding()
      };
      
      faces.push(faceData);
    }
    
    return faces;
  };

  const generateBiometricData = (): BiometricData => {
    const faceShapes = DEEP_FACE_FEATURES.faceShapes;
    const skinTones = DEEP_FACE_FEATURES.skinTones;
    const eyeColors = DEEP_FACE_FEATURES.eyeColors;
    const hairColors = DEEP_FACE_FEATURES.hairColors;
    const qualityMetrics = DEEP_FACE_FEATURES.qualityMetrics;
    
    return {
      faceShape: faceShapes[Math.floor(Math.random() * faceShapes.length)],
      skinTone: skinTones[Math.floor(Math.random() * skinTones.length)],
      eyeColor: eyeColors[Math.floor(Math.random() * eyeColors.length)],
      hairColor: hairColors[Math.floor(Math.random() * hairColors.length)],
      facialAsymmetry: Math.random() * 0.3,
      uniqueFeatures: generateUniqueFeatures(),
      faceQuality: qualityMetrics[Math.floor(Math.random() * qualityMetrics.length)],
      blurScore: Math.random() * 0.4,
      illuminationScore: 0.3 + Math.random() * 0.7
    };
  };

  const generateUniqueFeatures = (): string[] => {
    const features = [
      'dimples', 'freckles', 'mole', 'scar', 'beard', 'mustache', 
      'glasses', 'wrinkles', 'high_cheekbones', 'strong_jawline'
    ];
    const numFeatures = Math.floor(Math.random() * 3) + 1;
    const selectedFeatures: string[] = [];
    
    for (let i = 0; i < numFeatures; i++) {
      const feature = features[Math.floor(Math.random() * features.length)];
      if (!selectedFeatures.includes(feature)) {
        selectedFeatures.push(feature);
      }
    }
    
    return selectedFeatures;
  };

  const generateFacialLandmarks = (): number[][] => {
    const landmarks = [];
    for (let i = 0; i < 68; i++) {
      landmarks.push([
        200 + Math.random() * 200,
        100 + Math.random() * 150,
        Math.random() * 50
      ]);
    }
    return landmarks;
  };

  const generate3DMesh = (): Point3D[] => {
    const mesh: Point3D[] = [];
    const numVertices = 468; // Standard 3D face mesh
    
    for (let i = 0; i < numVertices; i++) {
      mesh.push({
        x: 200 + Math.random() * 200,
        y: 100 + Math.random() * 200,
        z: Math.random() * 100 - 50
      });
    }
    
    return mesh;
  };

  const generateEmotionData = () => {
    const emotions = ['happy', 'neutral', 'surprised', 'focused', 'thoughtful'];
    const primary = emotions[Math.floor(Math.random() * emotions.length)];
    
    return {
      primary,
      confidence: 0.7 + Math.random() * 0.25,
      all: {
        happy: primary === 'happy' ? 0.8 + Math.random() * 0.2 : Math.random() * 0.3,
        sad: primary === 'sad' ? 0.8 + Math.random() * 0.2 : Math.random() * 0.3,
        angry: primary === 'angry' ? 0.8 + Math.random() * 0.2 : Math.random() * 0.3,
        surprised: primary === 'surprised' ? 0.8 + Math.random() * 0.2 : Math.random() * 0.3,
        neutral: primary === 'neutral' ? 0.8 + Math.random() * 0.2 : Math.random() * 0.3,
        fear: Math.random() * 0.2,
        disgusted: Math.random() * 0.1
      },
      valence: primary === 'happy' ? 0.8 : primary === 'sad' ? -0.6 : 0,
      arousal: primary === 'surprised' ? 0.7 : 0.3,
      dominance: primary === 'angry' ? 0.6 : 0.4
    };
  };

  const generateFaceEmbedding = (): number[] => {
    const embedding = [];
    for (let i = 0; i < 512; i++) {
      embedding.push(Math.random() * 2 - 1);
    }
    return embedding;
  };

  const getAgeRange = (age: number): string => {
    if (age < 13) return '0-12';
    if (age < 20) return '13-19';
    if (age < 36) return '20-35';
    if (age < 56) return '36-55';
    return '56+';
  };

  const performAuthentication = (face: FaceData) => {
    const authData: BiometricAuthentication = {
      faceId: face.id,
      livenessScore: 0.85 + Math.random() * 0.1,
      antiSpoofingScore: 0.9 + Math.random() * 0.08,
      matchConfidence: face.isKnown ? 0.88 + Math.random() * 0.1 : 0.15 + Math.random() * 0.2,
      authenticationMethod: 'face',
      timestamp: new Date(),
      sessionToken: `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`
    };
    
    onAuthenticationUpdate(authData);
    console.log('🔐 Authentication performed:', authData);
  };

  const drawFaceAnalysis = (ctx: CanvasRenderingContext2D) => {
    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
    
    detectedFaces.forEach((face, index) => {
      // Draw bounding box
      ctx.strokeStyle = face.gender === 'male' ? '#3B82F6' : '#EC4899';
      ctx.lineWidth = 2;
      ctx.strokeRect(face.boundingBox.x, face.boundingBox.y, face.boundingBox.width, face.boundingBox.height);
      
      // Draw facial landmarks
      if (face.landmarks) {
        ctx.fillStyle = '#10B981';
        face.landmarks.forEach((landmark, i) => {
          if (i % 5 === 0) { // Draw every 5th landmark for clarity
            ctx.beginPath();
            ctx.arc(landmark[0], landmark[1], 2, 0, Math.PI * 2);
            ctx.fill();
          }
        });
      }
      
      // Draw 3D mesh points (simplified projection)
      if (face.mesh3D && face.mesh3D.length > 0) {
        ctx.fillStyle = 'rgba(6, 182, 212, 0.3)';
        face.mesh3D.forEach((point, i) => {
          if (i % 20 === 0) { // Draw every 20th point
            ctx.beginPath();
            ctx.arc(point.x, point.y, 1, 0, Math.PI * 2);
            ctx.fill();
          }
        });
      }
      
      // Draw biometric info
      ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
      ctx.fillRect(face.boundingBox.x, face.boundingBox.y - 60, face.boundingBox.width, 60);
      
      ctx.fillStyle = '#FFFFFF';
      ctx.font = '12px Arial';
      ctx.fillText(`${face.gender} • ${face.age}y`, face.boundingBox.x + 5, face.boundingBox.y - 45);
      
      if (face.biometrics) {
        ctx.font = '10px Arial';
        ctx.fillText(`${face.biometrics.faceShape} • ${face.biometrics.skinTone}`, face.boundingBox.x + 5, face.boundingBox.y - 30);
        ctx.fillText(`Quality: ${face.biometrics.faceQuality}`, face.boundingBox.x + 5, face.boundingBox.y - 15);
      }
      
      // Draw emotion if available
      if (face.emotion) {
        const emotionColors = {
          happy: '#10B981',
          sad: '#3B82F6',
          angry: '#EF4444',
          surprised: '#F59E0B',
          neutral: '#6B7280',
          fearful: '#8B5CF6',
          disgusted: '#84CC16'
        };
        
        ctx.fillStyle = emotionColors[face.emotion.primary] || '#6B7280';
        ctx.fillRect(face.boundingBox.x, face.boundingBox.y + face.boundingBox.height + 5, 100, 30);
        
        ctx.fillStyle = '#FFFFFF';
        ctx.font = '11px Arial';
        ctx.fillText(`${face.emotion.primary} (${(face.emotion.confidence * 100).toFixed(0)}%)`, face.boundingBox.x + 5, face.boundingBox.y + 25);
      }
      
      // Draw authentication status
      if (face.isKnown) {
        ctx.fillStyle = '#10B981';
        ctx.fillRect(face.boundingBox.x + face.boundingBox.width - 60, face.boundingBox.y, 60, 20);
        ctx.fillStyle = '#FFFFFF';
        ctx.font = '10px Arial';
        ctx.fillText('✓ Known', face.boundingBox.x + face.boundingBox.width - 50, face.boundingBox.y + 14);
      }
    });
  };

  useEffect(() => {
    if (canvasRef.current && isInitialized) {
      const ctx = canvasRef.current.getContext('2d');
      if (ctx) {
        drawFaceAnalysis(ctx);
      }
    }
  }, [detectedFaces, isInitialized]);

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
              <Eye className="w-5 h-5" />
              Deep Face Analyzer
            </CardTitle>
            <div className="flex items-center gap-2">
              <Badge className={isInitialized ? "bg-green-600" : "bg-yellow-600"}>
                {isInitialized ? "Active" : "Initializing..."}
              </Badge>
              <Badge className={isProcessing ? "bg-blue-600" : "bg-gray-600"}>
                {isProcessing ? "Analyzing" : "Idle"}
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
              <div className={`text-2xl font-bold ${getPerformanceColor(performance.accuracy, PERFORMANCE_THRESHOLDS.TARGET_ACCURACY)}`}>
                {(performance.accuracy * 100).toFixed(1)}%
              </div>
              <div className="text-gray-400 text-sm">Accuracy</div>
            </div>
            <div>
              <div className="text-2xl font-bold text-cyan-400">
                {detectedFaces.length}
              </div>
              <div className="text-gray-400 text-sm">Faces</div>
            </div>
            <div>
              <div className="text-2xl font-bold text-purple-400">
                {systemStatus.faceDatabase}
              </div>
              <div className="text-gray-400 text-sm">Database</div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Face Visualization */}
      <Card className="bg-black/30 backdrop-blur-md border-purple-500/20">
        <CardHeader>
          <CardTitle className="text-white flex items-center gap-2">
            <Camera className="w-5 h-5" />
            Biometric Analysis
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
            
            {/* Analysis Mode Toggle */}
            <div className="absolute top-4 right-4">
              <div className="bg-black/70 backdrop-blur-sm rounded-lg px-3 py-2">
                <div className="flex flex-col gap-2">
                  <Button 
                    size="sm" 
                    variant={authenticationMode === 'detection' ? "default" : "outline"}
                    onClick={() => setAuthenticationMode('detection')}
                  >
                    <Eye className="w-3 h-3 mr-1" />
                    Detection
                  </Button>
                  <Button 
                    size="sm" 
                    variant={authenticationMode === 'recognition' ? "default" : "outline"}
                    onClick={() => setAuthenticationMode('recognition')}
                  >
                    <User className="w-3 h-3 mr-1" />
                    Recognition
                  </Button>
                  <Button 
                    size="sm" 
                    variant={authenticationMode === 'authentication' ? "default" : "outline"}
                    onClick={() => setAuthenticationMode('authentication')}
                  >
                    <Fingerprint className="w-3 h-3 mr-1" />
                    Auth
                  </Button>
                </div>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Detailed Face Analysis */}
      {selectedFace && (
        <Card className="bg-black/30 backdrop-blur-md border-purple-500/20">
          <CardHeader>
            <CardTitle className="text-white flex items-center gap-2">
              <User className="w-5 h-5" />
              Detailed Analysis
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="space-y-4">
                <div>
                  <h4 className="text-white font-medium mb-2">Biometric Data</h4>
                  <div className="space-y-2 text-sm">
                    <div className="flex justify-between">
                      <span className="text-gray-400">Face Shape:</span>
                      <span className="text-white">{selectedFace.biometrics?.faceShape}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Skin Tone:</span>
                      <span className="text-white">{selectedFace.biometrics?.skinTone}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Eye Color:</span>
                      <span className="text-white">{selectedFace.biometrics?.eyeColor}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Hair Color:</span>
                      <span className="text-white">{selectedFace.biometrics?.hairColor}</span>
                    </div>
                  </div>
                </div>
                
                <div>
                  <h4 className="text-white font-medium mb-2">Quality Metrics</h4>
                  <div className="space-y-2">
                    <div className="flex justify-between">
                      <span className="text-gray-400">Face Quality:</span>
                      <span className="text-white">{selectedFace.biometrics?.faceQuality}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Blur Score:</span>
                      <span className="text-white">{(selectedFace.biometrics?.blurScore * 100).toFixed(1)}%</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Illumination:</span>
                      <span className="text-white">{(selectedFace.biometrics?.illuminationScore * 100).toFixed(1)}%</span>
                    </div>
                  </div>
                </div>
              </div>
              
              <div className="space-y-4">
                <div>
                  <h4 className="text-white font-medium mb-2">Unique Features</h4>
                  <div className="flex flex-wrap gap-2">
                    {selectedFace.biometrics?.uniqueFeatures.map((feature, index) => (
                      <Badge key={index} className="bg-purple-600">
                        {feature}
                      </Badge>
                    ))}
                  </div>
                </div>
                
                <div>
                  <h4 className="text-white font-medium mb-2">Emotion Analysis</h4>
                  {selectedFace.emotion && (
                    <div className="space-y-2">
                      <div className="flex items-center gap-2">
                        <span className="text-gray-400">Primary:</span>
                        <Badge className="bg-blue-600">
                          {selectedFace.emotion.primary}
                        </Badge>
                        <span className="text-white text-sm">
                          ({(selectedFace.emotion.confidence * 100).toFixed(0)}%)
                        </span>
                      </div>
                      <div className="flex items-center gap-2">
                        <span className="text-gray-400">Valence:</span>
                        <div className="w-24 bg-gray-700 rounded-full h-2">
                          <div 
                            className="bg-blue-500 h-2 rounded-full" 
                            style={{ width: `${((selectedFace.emotion.valence || 0) + 1) * 50}%` }}
                          />
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </div>
            
            <div className="mt-6 pt-6 border-t border-gray-700">
              <div className="flex items-center justify-between">
                <div>
                  <h4 className="text-white font-medium">Authentication Status</h4>
                  <p className="text-gray-400 text-sm">
                    {selectedFace.isKnown ? 'Known face detected' : 'Unknown face'}
                  </p>
                </div>
                <Button 
                  onClick={() => performAuthentication(selectedFace)}
                  disabled={selectedFace.isKnown}
                  className="bg-green-600 hover:bg-green-700"
                >
                  <Shield className="w-4 h-4 mr-2" />
                  Authenticate
                </Button>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* System Status */}
      <Card className="bg-black/30 backdrop-blur-md border-purple-500/20">
        <CardHeader>
          <CardTitle className="text-white flex items-center gap-2">
            <Settings className="w-5 h-5" />
            System Status
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-gray-400 text-sm">Face Database</span>
                <span className="text-white font-medium">{systemStatus.faceDatabase} faces</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-gray-400 text-sm">Processing Queue</span>
                <span className="text-white font-medium">{systemStatus.processingQueue}</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-gray-400 text-sm">Anti-Spoofing</span>
                <Badge className={systemStatus.antiSpoofingActive ? "bg-green-600" : "bg-gray-600"}>
                  {systemStatus.antiSpoofingActive ? "Active" : "Disabled"}
                </Badge>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-gray-400 text-sm">Liveness Detection</span>
                <Badge className={systemStatus.livenessDetection ? "bg-green-600" : "bg-gray-600"}>
                  {systemStatus.livenessDetection ? "Active" : "Disabled"}
                </Badge>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}