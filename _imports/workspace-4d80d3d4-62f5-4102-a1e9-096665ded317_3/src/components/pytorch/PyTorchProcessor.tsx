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
  Brain, 
  Zap, 
  Cpu, 
  Play, 
  Pause, 
  Settings, 
  Upload, 
  Download,
  Activity,
  Target,
  TrendingUp,
  AlertTriangle
} from 'lucide-react';
import { 
  PYTORCH_CONFIGURATIONS, 
  PERFORMANCE_THRESHOLDS, 
  THEME_COLORS 
} from '@/constants';
import { 
  PyTorchSettings, 
  PerformanceMetrics, 
  CustomModel, 
  ModelTrainingData 
} from '@/types';

interface PyTorchProcessorProps {
  isActive: boolean;
  settings: PyTorchSettings;
  onSettingsChange: (settings: PyTorchSettings) => void;
  onPerformanceUpdate: (metrics: PerformanceMetrics) => void;
  onModelUpdate: (models: CustomModel[]) => void;
}

export default function PyTorchProcessor({ 
  isActive, 
  settings, 
  onSettingsChange, 
  onPerformanceUpdate,
  onModelUpdate 
}: PyTorchProcessorProps) {
  const [isInitialized, setIsInitialized] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [gpuAvailable, setGpuAvailable] = useState(false);
  const [customModels, setCustomModels] = useState<CustomModel[]>([]);
  const [trainingJobs, setTrainingJobs] = useState<ModelTrainingData[]>([]);
  const [selectedModel, setSelectedModel] = useState<string>('');
  
  const [performance, setPerformance] = useState<PerformanceMetrics>({
    fps: 0,
    latency: 0,
    accuracy: 0,
    processingTime: 0,
    pyTorchInferenceTime: 0
  });

  const [systemInfo, setSystemInfo] = useState({
    gpuUtilization: 0,
    memoryBandwidth: 0,
    tensorOperations: 0,
    batchProcessingEfficiency: 0
  });

  // Initialize PyTorch
  useEffect(() => {
    if (isActive) {
      initializePyTorch();
    } else {
      cleanupPyTorch();
    }
    return () => cleanupPyTorch();
  }, [isActive]);

  const initializePyTorch = async () => {
    try {
      console.log('🚀 Initializing PyTorch with GPU acceleration...');
      
      // Check GPU availability
      const gpuCheck = await checkGPUAvailability();
      setGpuAvailable(gpuCheck.available);
      
      // Load custom models
      await loadCustomModels();
      
      // Initialize processing pipeline
      await new Promise(resolve => setTimeout(resolve, 1500));
      
      setIsInitialized(true);
      startInference();
      
      console.log('✅ PyTorch initialized successfully');
      console.log('🔧 Configuration:', {
        gpuAcceleration: settings.gpuAcceleration,
        batchSize: settings.batchSize,
        precision: settings.precision,
        customModels: settings.customModels.length
      });
    } catch (error) {
      console.error('❌ PyTorch initialization failed:', error);
    }
  };

  const cleanupPyTorch = () => {
    setIsProcessing(false);
    setIsInitialized(false);
  };

  const startInference = () => {
    setIsProcessing(true);
    runInference();
  };

  const checkGPUAvailability = async () => {
    // Simulate GPU check
    return {
      available: true,
      device: 'NVIDIA RTX 4090',
      memory: '24GB',
      computeCapability: '8.9'
    };
  };

  const loadCustomModels = async () => {
    // Simulate loading custom models
    const mockModels: CustomModel[] = [
      {
        id: 'custom-sign-v1',
        name: 'Custom Sign Model v1',
        type: 'sign',
        architecture: 'transformer',
        framework: 'pytorch',
        accuracy: 0.94,
        size: '1.2GB',
        inputShape: [30, 224, 224, 3],
        outputClasses: ['hello', 'thank_you', 'please', 'yes', 'no', 'help'],
        trainedAt: new Date('2024-10-01'),
        isOptimized: true
      },
      {
        id: 'custom-emotion-v2',
        name: 'Custom Emotion Model v2',
        type: 'emotion',
        architecture: 'cnn',
        framework: 'pytorch',
        accuracy: 0.91,
        size: '856MB',
        inputShape: [224, 224, 3],
        outputClasses: ['happy', 'sad', 'angry', 'surprised', 'neutral', 'fear'],
        trainedAt: new Date('2024-09-15'),
        isOptimized: false
      },
      {
        id: 'custom-pose-v3',
        name: 'Custom Pose Model v3',
        type: 'pose',
        architecture: 'vit',
        framework: 'pytorch',
        accuracy: 0.89,
        size: '2.1GB',
        inputShape: [224, 224, 3],
        outputClasses: ['standing', 'sitting', 'waving', 'pointing', 'walking'],
        trainedAt: new Date('2024-11-01'),
        isOptimized: true
      }
    ];
    
    setCustomModels(mockModels);
    onModelUpdate(mockModels);
  };

  const runInference = () => {
    const startTime = performance.now();
    
    // Simulate PyTorch inference
    const inferenceTime = 15 + Math.random() * 25; // 15-40ms
    const accuracy = 0.88 + Math.random() * 0.08;
    const fps = 1000 / inferenceTime;
    
    // Update system info
    const newSystemInfo = {
      gpuUtilization: 60 + Math.random() * 30,
      memoryBandwidth: 800 + Math.random() * 400,
      tensorOperations: Math.floor(1000000 + Math.random() * 500000),
      batchProcessingEfficiency: 0.85 + Math.random() * 0.1
    };
    
    setSystemInfo(newSystemInfo);
    
    const newPerformance = {
      fps,
      latency: inferenceTime,
      accuracy,
      processingTime: inferenceTime,
      pyTorchInferenceTime: inferenceTime
    };
    
    setPerformance(newPerformance);
    onPerformanceUpdate(newPerformance);
    
    if (isProcessing) {
      setTimeout(() => runInference(), 100);
    }
  };

  const startModelTraining = (modelId: string) => {
    const trainingJob: ModelTrainingData = {
      id: Date.now().toString(),
      modelName: `Custom Model ${modelId}`,
      status: 'training',
      progress: 0,
      accuracy: 0,
      loss: 2.5,
      epochs: 100,
      currentEpoch: 0,
      startTime: new Date(),
      estimatedCompletion: new Date(Date.now() + 3600000), // 1 hour
      dataset: {
        name: 'Custom Dataset',
        size: '2.5GB',
        samples: 10000
      },
      framework: 'pytorch'
    };
    
    setTrainingJobs(prev => [...prev, trainingJob]);
    
    // Simulate training progress
    simulateTrainingProgress(trainingJob.id);
  };

  const simulateTrainingProgress = (jobId: string) => {
    let progress = 0;
    const interval = setInterval(() => {
      progress += Math.random() * 5;
      if (progress >= 100) {
        progress = 100;
        clearInterval(interval);
        
        setTrainingJobs(prev => prev.map(job => 
          job.id === jobId 
            ? { ...job, status: 'completed', progress: 100, accuracy: 0.92, loss: 0.15 }
            : job
        ));
      } else {
        setTrainingJobs(prev => prev.map(job => 
          job.id === jobId 
            ? { ...job, progress, currentEpoch: Math.floor(progress), accuracy: 0.85 + (progress / 100) * 0.07, loss: 2.5 - (progress / 100) * 2.35 }
            : job
        ));
      }
    }, 2000);
  };

  const updateSetting = <K extends keyof PyTorchSettings>(key: K, value: PyTorchSettings[K]) => {
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
              <Brain className="w-5 h-5" />
              PyTorch Neural Engine
            </CardTitle>
            <div className="flex items-center gap-2">
              <Badge className={isInitialized ? "bg-green-600" : "bg-yellow-600"}>
                {isInitialized ? "Ready" : "Loading..."}
              </Badge>
              <Badge className={gpuAvailable ? "bg-green-600" : "bg-orange-600"}>
                {gpuAvailable ? "GPU" : "CPU"}
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
                {performance.pyTorchInferenceTime.toFixed(0)}ms
              </div>
              <div className="text-gray-400 text-sm">Inference</div>
            </div>
            <div>
              <div className={`text-2xl font-bold ${getPerformanceColor(performance.accuracy, PERFORMANCE_THRESHOLDS.TARGET_ACCURACY)}`}>
                {(performance.accuracy * 100).toFixed(1)}%
              </div>
              <div className="text-gray-400 text-sm">Accuracy</div>
            </div>
            <div>
              <div className="text-2xl font-bold text-cyan-400">
                {settings.batchSize}
              </div>
              <div className="text-gray-400 text-sm">Batch Size</div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Custom Models */}
      <Card className="bg-black/30 backdrop-blur-md border-purple-500/20">
        <CardHeader>
          <CardTitle className="text-white flex items-center gap-2">
            <Upload className="w-5 h-5" />
            Custom Neural Models
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {customModels.map((model) => (
              <div key={model.id} className="flex items-center justify-between p-4 bg-black/20 rounded-lg border border-gray-600">
                <div className="flex items-center gap-3">
                  <div className={`w-8 h-8 rounded-lg flex items-center justify-center ${
                    model.framework === 'pytorch' ? 'bg-orange-600' : 
                    model.framework === 'tensorflow' ? 'bg-blue-600' : 'bg-gray-600'
                  }`}>
                    <span className="text-white text-xs font-bold">{model.framework.toUpperCase()}</span>
                  </div>
                  <div>
                    <div className="text-white font-medium">{model.name}</div>
                    <div className="text-gray-400 text-xs">{model.architecture.toUpperCase()}</div>
                  </div>
                </div>
                <div className="text-right">
                  <div className="text-cyan-400 text-sm">{(model.accuracy * 100).toFixed(1)}%</div>
                  <div className="text-green-400 text-sm">{model.size}</div>
                </div>
                <div className="flex items-center gap-2 mt-2">
                  <Badge className={model.isOptimized ? "bg-green-600" : "bg-yellow-600"}>
                    {model.isOptimized ? "Optimized" : "Standard"}
                  </Badge>
                  <Button 
                    size="sm" 
                    variant="outline"
                    onClick={() => startModelTraining(model.id)}
                  >
                    <Target className="w-3 h-3 mr-1" />
                    Train
                  </Button>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Training Jobs */}
      {trainingJobs.length > 0 && (
        <Card className="bg-black/30 backdrop-blur-md border-purple-500/20">
          <CardHeader>
            <CardTitle className="text-white flex items-center gap-2">
              <Activity className="w-5 h-5" />
              Active Training Jobs
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {trainingJobs.map((job) => (
                <div key={job.id} className="space-y-3">
                  <div className="flex items-center justify-between">
                    <div className="text-white font-medium">{job.modelName}</div>
                    <div className="text-gray-400 text-sm">
                      Epoch {job.currentEpoch}/{job.epochs} • {job.framework.toUpperCase()}
                    </div>
                    <Badge className={
                      job.status === 'completed' ? "bg-green-600" : 
                      job.status === 'training' ? "bg-blue-600" : "bg-gray-600"
                    }>
                      {job.status === 'completed' ? "Completed" : 
                       job.status === 'training' ? "Training" : "Queued"}
                    </Badge>
                  </div>
                  
                  <div className="space-y-2">
                    <div className="flex items-center justify-between">
                      <span className="text-gray-400 text-sm">Progress</span>
                      <span className="text-white text-sm">{job.progress.toFixed(1)}%</span>
                    </div>
                    <Progress value={job.progress} className="h-2" />
                  </div>
                  
                  <div className="grid grid-cols-2 gap-4 text-sm">
                    <div>
                      <span className="text-gray-400">Accuracy</span>
                      <span className="text-white font-medium ml-2">
                        {job.accuracy ? (job.accuracy * 100).toFixed(1) + '%' : 'N/A'}
                      </span>
                    </div>
                    <div>
                      <span className="text-gray-400">Loss</span>
                      <span className="text-white font-medium ml-2">
                        {job.loss ? job.loss.toFixed(3) : 'N/A'}
                      </span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* System Configuration */}
      <Card className="bg-black/30 backdrop-blur-md border-purple-500/20">
        <CardHeader>
          <CardTitle className="text-white flex items-center gap-2">
            <Settings className="w-5 h-5" />
            PyTorch Configuration
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="text-white text-sm font-medium mb-2 block">GPU Acceleration</label>
              <Switch
                checked={settings.gpuAcceleration}
                onCheckedChange={(checked) => updateSetting('gpuAcceleration', checked)}
              />
              <div className="text-gray-400 text-xs mt-1">
                {gpuAvailable ? 'NVIDIA CUDA Available' : 'CPU Fallback Mode'}
              </div>
            </div>

            <div>
              <label className="text-white text-sm font-medium mb-2 block">Precision</label>
              <Select 
                value={settings.precision} 
                onValueChange={(value: any) => updateSetting('precision', value)}
              >
                <SelectTrigger className="bg-black/20 border-gray-600 text-white">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent className="bg-black/90 border-gray-600">
                  <SelectItem value="fp16" className="text-white hover:bg-purple-600/20">FP16 (Half Precision)</SelectItem>
                  <SelectItem value="fp32" className="text-white hover:bg-purple-600/20">FP32 (Full Precision)</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div>
              <label className="text-white text-sm font-medium mb-2 block">
                Batch Size: {settings.batchSize}
              </label>
              <Slider
                value={[settings.batchSize]}
                onValueChange={([value]) => updateSetting('batchSize', value)}
                max={128}
                min={1}
                step={1}
                className="w-full"
              />
            </div>

            <div>
              <label className="text-white text-sm font-medium mb-2 block">Model Path</label>
              <div className="bg-black/20 border border-gray-600 rounded px-3 py-2 text-white text-sm">
                {settings.modelPath || '/models/custom/'}
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* System Metrics */}
      <Card className="bg-black/30 backdrop-blur-md border-purple-500/20">
        <CardHeader>
          <CardTitle className="text-white flex items-center gap-2">
            <Cpu className="w-5 h-5" />
            System Metrics
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-center">
            <div>
              <div className="text-2xl font-bold text-orange-400">
                {systemInfo.gpuUtilization.toFixed(0)}%
              </div>
              <div className="text-gray-400 text-sm">GPU Utilization</div>
            </div>
            <div>
              <div className="text-2xl font-bold text-cyan-400">
                {(systemInfo.memoryBandwidth / 1000).toFixed(1)}GB/s
              </div>
              <div className="text-gray-400 text-sm">Memory Bandwidth</div>
            </div>
            <div>
              <div className="text-2xl font-bold text-purple-400">
                {(systemInfo.tensorOperations / 1000000).toFixed(1)}M
              </div>
              <div className="text-gray-400 text-sm">Tensor Ops/s</div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}