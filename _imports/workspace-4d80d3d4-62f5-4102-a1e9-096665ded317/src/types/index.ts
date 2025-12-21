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
}

export interface PerformanceMetrics {
  fps: number;
  latency: number;
  accuracy: number;
  processingTime: number;
  gpuUsage?: number;
  memoryUsage?: number;
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
  type: 'sign' | 'face' | 'emotion' | 'voice' | 'age' | 'gender';
  accuracy: number;
  speed: number;
  loaded: boolean;
  size: string;
  description: string;
  lastTrained?: Date;
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
  emotion?: EmotionData;
  timestamp: Date;
  isKnown: boolean;
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
  };
}