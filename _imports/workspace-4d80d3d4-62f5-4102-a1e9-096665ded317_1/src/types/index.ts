export interface DigitalCertificate {
  owner: string;
  email: string;
  productVersion: string;
  encryptionKey: string;
  issuedAt: string;
}

export interface TranslationResult {
  id: string;
  originalSign: string;
  translatedText: string;
  confidence: number;
  timestamp: Date;
  language: string;
  audioUrl?: string;
  duration?: number;
  gestureData?: GestureData;
  poseData?: PoseData;
}

export interface SignLanguage {
  code: string;
  name: string;
  flag: string;
}

export interface CameraSettings {
  device: string;
  resolution: string;
  frameRate: number;
  quality: number;
}

export interface UserSettings {
  preferredLanguage: string;
  autoTranslate: boolean;
  soundEnabled: boolean;
  vibrationEnabled: boolean;
  camera: CameraSettings;
  voiceSettings: VoiceSettings;
  faceRecognition: FaceRecognitionSettings;
  mediaPipeSettings: MediaPipeSettings;
  pyTorchSettings: PyTorchSettings;
}

export interface VoiceSettings {
  enabled: boolean;
  voice: string;
  speed: number;
  pitch: number;
  volume: number;
  language: string;
}

export interface FaceRecognitionSettings {
  enabled: boolean;
  saveFaces: boolean;
  alertOnUnknown: boolean;
  trackingEnabled: boolean;
  deepAnalysis: boolean;
  biometricAuth: boolean;
}

export interface MediaPipeSettings {
  enabled: boolean;
  modelComplexity: 'lite' | 'full' | 'heavy';
  minDetectionConfidence: number;
  trackingMode: 'single' | 'multi';
  smoothLandmarks: boolean;
  enableSegmentation: boolean;
}

export interface PyTorchSettings {
  enabled: boolean;
  modelPath: string;
  gpuAcceleration: boolean;
  batchSize: number;
  precision: 'fp16' | 'fp32';
  customModels: CustomModel[];
}

export interface PerformanceMetrics {
  fps: number;
  latency: number;
  accuracy: number;
  processingTime: number;
  gpuUsage?: number;
  memoryUsage?: number;
  mediaPipeLatency?: number;
  pyTorchInferenceTime?: number;
}

export interface DetectionResult {
  landmarks: number[][];
  confidence: number;
  boundingBox: {
    x: number;
    y: number;
    width: number;
    height: number;
  };
}

export interface AIModel {
  id: string;
  name: string;
  version: string;
  type: 'sign' | 'face' | 'emotion' | 'voice' | 'age' | 'gender' | 'pose' | 'hand' | 'mesh';
  accuracy: number;
  speed: number;
  loaded: boolean;
  size: string;
  description: string;
  lastTrained?: Date;
  framework: 'tensorflow' | 'pytorch' | 'onnx' | 'mediapipe';
}

export interface FaceData {
  id: string;
  name?: string;
  gender: 'male' | 'female' | 'unknown';
  age: number;
  ageRange: string;
  confidence: number;
  boundingBox: {
    x: number;
    y: number;
    width: number;
    height: number;
  };
  landmarks: number[][];
  mesh3D?: Point3D[];
  emotion?: EmotionData;
  biometrics?: BiometricData;
  timestamp: Date;
  isKnown: boolean;
  faceEmbedding?: number[];
}

export interface BiometricData {
  faceShape: 'round' | 'oval' | 'square' | 'heart' | 'triangle';
  skinTone: string;
  eyeColor: string;
  hairColor: string;
  facialAsymmetry: number;
  uniqueFeatures: string[];
  faceQuality: number;
  blurScore: number;
  illuminationScore: number;
}

export interface Point3D {
  x: number;
  y: number;
  z: number;
}

export interface EmotionData {
  primary: string;
  confidence: number;
  all: {
    happy: number;
    sad: number;
    angry: number;
    surprised: number;
    neutral: number;
    fear: number;
    disgusted: number;
  };
  valence?: number;
  arousal?: number;
  dominance?: number;
}

export interface GestureData {
  id: string;
  name: string;
  confidence: number;
  handLandmarks: HandLandmark[];
  motionVector: Point3D;
  dynamicGestures: DynamicGesture[];
  signComplexity: number;
  executionTime: number;
}

export interface HandLandmark {
  id: number;
  x: number;
  y: number;
  z: number;
  visibility: number;
  presence: number;
}

export interface DynamicGesture {
  type: 'static' | 'dynamic' | 'sequential';
  sequence: string[];
  timing: number[];
  confidence: number;
}

export interface PoseData {
  id: string;
  landmarks: PoseLandmark[];
  poseClassification: string;
  bodyOrientation: Point3D;
  movementVector: Point3D;
  confidence: number;
}

export interface PoseLandmark {
  id: number;
  name: string;
  position: Point3D;
  visibility: number;
  confidence: number;
}

export interface VoiceSynthesis {
  text: string;
  voice: string;
  speed: number;
  pitch: number;
  volume: number;
  language: string;
  audioUrl?: string;
  duration?: number;
  emotion?: string;
}

export interface SessionData {
  id: string;
  userId: string;
  startTime: Date;
  endTime?: Date;
  translations: TranslationResult[];
  faces: FaceData[];
  performance: PerformanceMetrics[];
  language: string;
  duration?: number;
  mediaPipeMetrics?: MediaPipeMetrics;
  pyTorchMetrics?: PyTorchMetrics;
}

export interface MediaPipeMetrics {
  poseLandmarks: number;
  handLandmarks: number;
  faceMeshVertices: number;
  segmentationMaskQuality: number;
  pipelineLatency: number;
  trackingAccuracy: number;
}

export interface PyTorchMetrics {
  modelInferenceTime: number;
  gpuUtilization: number;
  memoryBandwidth: number;
  tensorOperations: number;
  batchProcessingEfficiency: number;
}

export interface AnalyticsData {
  totalTranslations: number;
  uniqueFaces: number;
  averageConfidence: number;
  sessionDuration: number;
  mostUsedPhrases: Array<{ phrase: string; count: number }>;
  emotionDistribution: Record<string, number>;
  ageDistribution: Record<string, number>;
  genderDistribution: Record<string, number>;
  performanceTrends: Array<{
    timestamp: Date;
    fps: number;
    latency: number;
    accuracy: number;
  }>;
  gestureComplexity: Record<string, number>;
  mediaPipeEfficiency: number;
  pyTorchPerformance: number;
}

export interface ModelTrainingData {
  id: string;
  modelName: string;
  status: 'training' | 'completed' | 'failed' | 'queued';
  progress: number;
  accuracy?: number;
  loss?: number;
  epochs: number;
  currentEpoch: number;
  startTime: Date;
  estimatedCompletion?: Date;
  dataset: {
    name: string;
    size: number;
    samples: number;
  };
  framework: 'tensorflow' | 'pytorch' | 'mediapipe';
}

export interface CustomModel {
  id: string;
  name: string;
  type: 'sign' | 'face' | 'emotion' | 'pose';
  architecture: 'cnn' | 'rnn' | 'transformer' | 'gan' | 'vit';
  framework: 'pytorch' | 'tensorflow' | 'onnx';
  accuracy: number;
  size: string;
  inputShape: number[];
  outputClasses: string[];
  trainedAt: Date;
  isOptimized: boolean;
}

export interface AdvancedSettings {
  experimentalFeatures: boolean;
  debugMode: boolean;
  dataCollection: boolean;
  cloudSync: boolean;
  offlineMode: boolean;
  securityLevel: 'basic' | 'standard' | 'enhanced';
  apiEndpoints: {
    translation: string;
    faceRecognition: string;
    voiceSynthesis: string;
    analytics: string;
    mediaPipe: string;
    pyTorch: string;
  };
  neuralRendering: boolean;
  realTimeMeshing: boolean;
  multiModalFusion: boolean;
}

export interface Mesh3D {
  vertices: Point3D[];
  faces: number[][];
  normals: Point3D[];
  uvs: number[][];
  texture?: string;
  vertexColors?: number[];
}

export interface NeuralRenderingPipeline {
  vertexShader: string;
  fragmentShader: string;
  computeShader: string;
  neuralNetwork: any;
  renderMode: 'wireframe' | 'solid' | 'point_cloud' | 'neural_radiance';
  postProcessing: {
    bloom: boolean;
    motionBlur: boolean;
    depthOfField: boolean;
    neuralStyling: boolean;
  };
}

export interface MultiModalFusion {
  faceWeight: number;
  poseWeight: number;
  gestureWeight: number;
  emotionWeight: number;
  fusionStrategy: 'early' | 'late' | 'intermediate' | 'attention';
  confidenceCalibration: number;
}

export interface BiometricAuthentication {
  faceId: string;
  livenessScore: number;
  antiSpoofingScore: number;
  matchConfidence: number;
  authenticationMethod: 'face' | 'voice' | 'multimodal';
  timestamp: Date;
  sessionToken?: string;
}