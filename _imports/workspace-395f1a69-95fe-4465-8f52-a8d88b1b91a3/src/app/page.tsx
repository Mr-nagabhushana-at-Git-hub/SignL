'use client'

import { useState, useRef, useEffect, useCallback } from 'react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Badge } from '@/components/ui/badge'
import { Progress } from '@/components/ui/progress'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { Separator } from '@/components/ui/separator'
import { useToast } from '@/hooks/use-toast'
import { useMediaPipe } from '@/hooks/use-mediapipe'
import { 
  Camera, 
  CameraOff, 
  Play, 
  Square, 
  RotateCcw,
  Video,
  Brain,
  Database,
  Globe,
  Hand,
  Activity,
  CheckCircle,
  AlertCircle,
  Zap,
  Loader2
} from 'lucide-react'

interface RecognitionResult {
  sign: string
  confidence: number
  timestamp: number
}

interface ModelInfo {
  name: string
  type: string
  accuracy: number
  speed: number
  description: string
}

interface DatasetInfo {
  name: string
  language: string
  size: string
  vocabulary: number
  type: string
}

const models: ModelInfo[] = [
  {
    name: 'SPOTER',
    type: 'Pose-based Transformer',
    accuracy: 85,
    speed: 95,
    description: 'Lightweight transformer for real-time pose-based sign recognition'
  },
  {
    name: 'ViT ISL Classifier',
    type: 'Vision Transformer',
    accuracy: 88,
    speed: 70,
    description: 'Indian Sign Language classification using ViT architecture'
  },
  {
    name: 'SignVLM',
    type: 'Vision-Language Model',
    accuracy: 91,
    speed: 60,
    description: 'Multi-sign language model with zero-shot capabilities'
  },
  {
    name: 'MediaPipe + NN',
    type: 'Hybrid Approach',
    accuracy: 82,
    speed: 90,
    description: 'MediaPipe landmarks with neural network classification'
  }
]

const datasets: DatasetInfo[] = [
  {
    name: 'WLASL2000',
    language: 'ASL (American)',
    size: '21,083 videos',
    vocabulary: 2000,
    type: 'Isolated signs'
  },
  {
    name: 'iSign',
    language: 'ISL (Indian)',
    size: '31,000+ pairs',
    vocabulary: 0,
    type: 'Continuous sentences'
  },
  {
    name: 'CISLR',
    language: 'ISL (Indian)',
    size: '50,000+ videos',
    vocabulary: 4700,
    type: 'Isolated signs'
  },
  {
    name: 'Boston ASLLVD',
    language: 'ASL (American)',
    size: '3,000+ videos',
    vocabulary: 161,
    type: 'Isolated signs'
  }
]

export default function SignLanguageRecognition() {
  const [isStreaming, setIsStreaming] = useState(false)
  const [selectedModel, setSelectedModel] = useState('SPOTER')
  const [selectedLanguage, setSelectedLanguage] = useState('ASL')
  const [recognitionResults, setRecognitionResults] = useState<RecognitionResult[]>([])
  const [isProcessing, setIsProcessing] = useState(false)
  const [fps, setFps] = useState(0)
  const [landmarksDetected, setLandmarksDetected] = useState(false)
  const [isLoading, setIsLoading] = useState(false)
  const [apiModels, setApiModels] = useState<any>({})
  const [apiDatasets, setApiDatasets] = useState<any>({})
  
  const { toast } = useToast()
  const streamRef = useRef<MediaStream | null>(null)
  const lastFrameTime = useRef<number>(0)
  const recognitionInterval = useRef<NodeJS.Timeout | null>(null)

  // MediaPipe integration
  const {
    videoRef,
    canvasRef,
    startProcessing,
    stopProcessing,
    isInitialized: mediaPipeInitialized
  } = useMediaPipe({
    onLandmarks: (landmarks) => {
      setLandmarksDetected(true)
      // Send landmarks to API for recognition if processing
      if (isProcessing) {
        recognizeSign(landmarks)
      }
    },
    onNoLandmarks: () => {
      setLandmarksDetected(false)
    }
  })

  // Load API data on mount
  useEffect(() => {
    loadApiData()
  }, [])

  const loadApiData = async () => {
    try {
      const response = await fetch('/api/sign-language')
      const data = await response.json()
      setApiModels(data.models || {})
      setApiDatasets(data.datasets || {})
    } catch (error) {
      console.error('Failed to load API data:', error)
    }
  }

  const recognizeSign = async (landmarks: any) => {
    try {
      const response = await fetch('/api/sign-language', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          action: 'recognize',
          model: selectedModel,
          language: selectedLanguage,
          landmarks: landmarks
        })
      })

      const result = await response.json()
      
      if (result.sign) {
        const newResult: RecognitionResult = {
          sign: result.sign,
          confidence: result.confidence || 0,
          timestamp: Date.now()
        }
        
        setRecognitionResults(prev => [newResult, ...prev.slice(0, 9)])
        
        toast({
          title: "Sign Detected",
          description: `${result.sign} (${result.confidence}% confidence)`,
        })
      }
    } catch (error) {
      console.error('Recognition error:', error)
    }
  }

  const startCamera = async () => {
    try {
      setIsLoading(true)
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { 
          width: 640, 
          height: 480,
          facingMode: 'user'
        } 
      })
      
      if (videoRef.current) {
        videoRef.current.srcObject = stream
        streamRef.current = stream
        setIsStreaming(true)
        
        // Start MediaPipe processing
        setTimeout(() => {
          startProcessing()
        }, 1000)
      }
    } catch (error) {
      console.error('Error accessing camera:', error)
      toast({
        title: "Camera Error",
        description: "Unable to access camera. Please check permissions.",
        variant: "destructive"
      })
    } finally {
      setIsLoading(false)
    }
  }

  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop())
      streamRef.current = null
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null
    }
    
    stopProcessing()
    setIsStreaming(false)
    setLandmarksDetected(false)
    setFps(0)
  }

  const startRecognition = () => {
    setIsProcessing(true)
    toast({
      title: "Recognition Started",
      description: `Using ${selectedModel} for ${selectedLanguage} sign language`,
    })
  }

  const stopRecognition = () => {
    setIsProcessing(false)
    if (recognitionInterval.current) {
      clearInterval(recognitionInterval.current)
      recognitionInterval.current = null
    }
  }

  const resetRecognition = () => {
    setRecognitionResults([])
    stopRecognition()
  }

  // Calculate FPS
  useEffect(() => {
    let frameCount = 0
    let lastTime = performance.now()
    
    const calculateFPS = () => {
      frameCount++
      const currentTime = performance.now()
      
      if (currentTime - lastTime >= 1000) {
        setFps(frameCount)
        frameCount = 0
        lastTime = currentTime
      }
      
      if (isStreaming) {
        requestAnimationFrame(calculateFPS)
      }
    }
    
    if (isStreaming) {
      calculateFPS()
    }
  }, [isStreaming])

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 p-4">
      <div className="max-w-7xl mx-auto space-y-6">
        {/* Header */}
        <div className="text-center space-y-2">
          <h1 className="text-4xl font-bold text-gray-900 flex items-center justify-center gap-2">
            <Hand className="w-8 h-8 text-blue-600" />
            Sign Language Recognition
          </h1>
          <p className="text-gray-600">Real-time sign language detection using AI models</p>
        </div>

        {/* Main Content */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Video Feed Section */}
          <div className="lg:col-span-2 space-y-4">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Video className="w-5 h-5" />
                  Camera Feed
                </CardTitle>
                <CardDescription>
                  Real-time video capture for sign language recognition
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="relative aspect-video bg-gray-900 rounded-lg overflow-hidden">
                  <video
                    ref={videoRef}
                    autoPlay
                    playsInline
                    muted
                    className="absolute inset-0 w-full h-full object-cover"
                    style={{ display: isStreaming ? 'block' : 'none' }}
                  />
                  <canvas
                    ref={canvasRef}
                    className="absolute inset-0 w-full h-full object-cover"
                    style={{ display: isStreaming ? 'block' : 'none' }}
                  />
                  
                  {!isStreaming && (
                    <div className="absolute inset-0 flex items-center justify-center bg-gray-800">
                      <div className="text-center space-y-2">
                        <CameraOff className="w-12 h-12 text-gray-400 mx-auto" />
                        <p className="text-gray-400">Camera not active</p>
                      </div>
                    </div>
                  )}

                  {/* Status Overlay */}
                  {isStreaming && (
                    <div className="absolute top-4 left-4 space-y-2">
                      <Badge variant={landmarksDetected ? "default" : "secondary"}>
                        {landmarksDetected ? (
                          <><CheckCircle className="w-3 h-3 mr-1" />Landmarks Detected</>
                        ) : (
                          <><AlertCircle className="w-3 h-3 mr-1" />Detecting...</>
                        )}
                      </Badge>
                      <Badge variant="outline">
                        <Activity className="w-3 h-3 mr-1" />
                        {fps} FPS
                      </Badge>
                    </div>
                  )}
                </div>

                {/* Camera Controls */}
                <div className="flex gap-2 justify-center">
                  {!isStreaming ? (
                    <Button onClick={startCamera} disabled={isLoading} className="flex items-center gap-2">
                      {isLoading ? (
                        <Loader2 className="w-4 h-4 animate-spin" />
                      ) : (
                        <Camera className="w-4 h-4" />
                      )}
                      Start Camera
                    </Button>
                  ) : (
                    <Button onClick={stopCamera} variant="destructive" className="flex items-center gap-2">
                      <CameraOff className="w-4 h-4" />
                      Stop Camera
                    </Button>
                  )}
                </div>
              </CardContent>
            </Card>

            {/* Recognition Controls */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Brain className="w-5 h-5" />
                  Recognition Controls
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <label className="text-sm font-medium">Model</label>
                    <Select value={selectedModel} onValueChange={setSelectedModel}>
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        {models.map(model => (
                          <SelectItem key={model.name} value={model.name}>
                            {model.name}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                  
                  <div className="space-y-2">
                    <label className="text-sm font-medium">Language</label>
                    <Select value={selectedLanguage} onValueChange={setSelectedLanguage}>
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="ASL">ASL (American)</SelectItem>
                        <SelectItem value="ISL">ISL (Indian)</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                </div>

                <div className="flex gap-2 justify-center">
                  {!isProcessing ? (
                    <Button 
                      onClick={startRecognition} 
                      disabled={!isStreaming}
                      className="flex items-center gap-2"
                    >
                      <Play className="w-4 h-4" />
                      Start Recognition
                    </Button>
                  ) : (
                    <Button 
                      onClick={stopRecognition} 
                      variant="destructive"
                      className="flex items-center gap-2"
                    >
                      <Square className="w-4 h-4" />
                      Stop Recognition
                    </Button>
                  )}
                  <Button 
                    onClick={resetRecognition} 
                    variant="outline"
                    className="flex items-center gap-2"
                  >
                    <RotateCcw className="w-4 h-4" />
                    Reset
                  </Button>
                </div>

                {isProcessing && (
                  <div className="space-y-2">
                    <div className="flex items-center gap-2">
                      <Zap className="w-4 h-4 text-yellow-500" />
                      <span className="text-sm">Processing...</span>
                    </div>
                    <Progress value={66} className="w-full" />
                  </div>
                )}
              </CardContent>
            </Card>
          </div>

          {/* Results Section */}
          <div className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle>Recognition Results</CardTitle>
                <CardDescription>
                  Detected signs with confidence scores
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-2 max-h-96 overflow-y-auto">
                  {recognitionResults.length === 0 ? (
                    <p className="text-gray-500 text-center py-4">No results yet</p>
                  ) : (
                    recognitionResults.map((result, index) => (
                      <div key={result.timestamp} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                        <div className="flex-1">
                          <p className="font-medium">{result.sign}</p>
                          <p className="text-xs text-gray-500">
                            {new Date(result.timestamp).toLocaleTimeString()}
                          </p>
                        </div>
                        <div className="text-right">
                          <Badge variant={result.confidence > 80 ? "default" : "secondary"}>
                            {result.confidence}%
                          </Badge>
                        </div>
                      </div>
                    ))
                  )}
                </div>
              </CardContent>
            </Card>

            {/* Model Info */}
            <Card>
              <CardHeader>
                <CardTitle>Model Information</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  {models.filter(m => m.name === selectedModel).map(model => (
                    <div key={model.name} className="space-y-2">
                      <div className="flex items-center justify-between">
                        <span className="text-sm font-medium">Accuracy</span>
                        <Badge>{model.accuracy}%</Badge>
                      </div>
                      <div className="flex items-center justify-between">
                        <span className="text-sm font-medium">Speed</span>
                        <Badge variant="secondary">{model.speed}%</Badge>
                      </div>
                      <Separator />
                      <p className="text-sm text-gray-600">{model.description}</p>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </div>
        </div>

        {/* Dataset Information */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Database className="w-5 h-5" />
              Training Datasets
            </CardTitle>
            <CardDescription>
              Available datasets for training sign language models
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
              {datasets.map(dataset => (
                <Card key={dataset.name} className="p-4">
                  <div className="space-y-2">
                    <div className="flex items-center gap-2">
                      <Globe className="w-4 h-4 text-blue-600" />
                      <h3 className="font-medium">{dataset.name}</h3>
                    </div>
                    <p className="text-sm text-gray-600">{dataset.language}</p>
                    <p className="text-xs text-gray-500">{dataset.size}</p>
                    {dataset.vocabulary > 0 && (
                      <p className="text-xs text-gray-500">{dataset.vocabulary} words</p>
                    )}
                    <Badge variant="outline" className="text-xs">
                      {dataset.type}
                    </Badge>
                  </div>
                </Card>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}