import { DigitalCertificate, AIModel } from '@/types';

/**
 * DIGITAL CERTIFICATE OF OWNERSHIP
 * --------------------------------
 * This product "SignSync Omni V10" is intellectually owned by Nagabhushana Raju S.
 * Any unauthorized reproduction is prohibited.
 */
export const DIGITAL_CERTIFICATE: DigitalCertificate = {
  owner: "Nagabhushana Raju S",
  email: "nagabhushanaraju2003@gmail.com",
  productVersion: "10.0.1-Alpha-Neural",
  encryptionKey: "NRAJU-V10-SIGNSYNC-SECURE-HASH-256",
  issuedAt: new Date().toISOString()
};

// Simulation Phrases: These mimic common Sign Language glosses/sentences
export const SYSTEM_PHRASES = [
  "Hello there!",
  "How are you doing today?",
  "My name is Guest.",
  "Nice to meet you.",
  "I need some assistance here.",
  "Thank you very much.",
  "Yes, I agree.",
  "No, that is incorrect.",
  "What time is it?",
  "Can you please help me?",
  "I am happy to see you.",
  "Good morning!",
  "Have a great day!",
  "See you later!",
  "Take care!",
  "I understand.",
  "Can you repeat that?",
  "Please speak slowly.",
  "I am learning sign language.",
  "Thank you for your patience."
];

// Humanoid 3D Avatar (Cyberpunk/Neural Aesthetic)
export const AVATAR_PLACEHOLDER = "https://images.unsplash.com/photo-1618336753974-aae8e04506aa?q=80&w=1000&auto=format&fit=crop";

// Supported Sign Languages
export const SUPPORTED_LANGUAGES = [
  { code: "ASL", name: "American Sign Language", flag: "🇺🇸" },
  { code: "BSL", name: "British Sign Language", flag: "🇬🇧" },
  { code: "ISL", name: "Indian Sign Language", flag: "🇮🇳" },
  { code: "JSL", name: "Japanese Sign Language", flag: "🇯🇵" },
  { code: "CSL", name: "Chinese Sign Language", flag: "🇨🇳" },
  { code: "FSL", name: "French Sign Language", flag: "🇫🇷" },
  { code: "DGS", name: "German Sign Language", flag: "🇩🇪" },
  { code: "LIBRAS", name: "Brazilian Sign Language", flag: "🇧🇷" },
  { code: "Auslan", name: "Australian Sign Language", flag: "🇦🇺" },
  { code: "NZSL", name: "New Zealand Sign Language", flag: "🇳🇿" }
];

// Advanced AI Models with MediaPipe and PyTorch integration
export const ADVANCED_AI_MODELS: AIModel[] = [
  {
    id: "sign-transformer-v3",
    name: "Sign Transformer v3.0",
    version: "3.0.0",
    type: "sign",
    accuracy: 0.96,
    speed: 0.92,
    loaded: true,
    size: "2.3GB",
    description: "Advanced transformer-based sign language recognition model",
    lastTrained: new Date("2024-10-15"),
    framework: "pytorch"
  },
  {
    id: "mediapipe-holistic",
    name: "MediaPipe Holistic",
    version: "0.10.7",
    type: "pose",
    accuracy: 0.94,
    speed: 0.96,
    loaded: true,
    size: "45MB",
    description: "Real-time holistic pose, face, and hand tracking",
    framework: "mediapipe"
  },
  {
    id: "face-recognition-pro",
    name: "Face Recognition Pro",
    version: "2.1.0",
    type: "face",
    accuracy: 0.98,
    speed: 0.88,
    loaded: true,
    size: "1.8GB",
    description: "High-precision face recognition with 10M+ face database",
    lastTrained: new Date("2024-11-01"),
    framework: "pytorch"
  },
  {
    id: "deep-face-analyzer",
    name: "Deep Face Analyzer",
    version: "1.5.0",
    type: "face",
    accuracy: 0.97,
    speed: 0.85,
    loaded: true,
    size: "2.1GB",
    description: "Advanced biometric analysis with 3D face reconstruction",
    framework: "tensorflow"
  },
  {
    id: "emotion-net",
    name: "EmotionNet",
    version: "1.5.0",
    type: "emotion",
    accuracy: 0.91,
    speed: 0.95,
    loaded: true,
    size: "856MB",
    description: "Real-time emotion detection and sentiment analysis",
    lastTrained: new Date("2024-09-20"),
    framework: "pytorch"
  },
  {
    id: "age-gender-detector",
    name: "Age-Gender Detector",
    version: "1.2.0",
    type: "age",
    accuracy: 0.89,
    speed: 0.94,
    loaded: true,
    size: "623MB",
    description: "Age and gender detection with demographic analysis",
    framework: "tensorflow"
  },
  {
    id: "voice-synthesis-ultra",
    name: "Voice Synthesis Ultra",
    version: "4.0.0",
    type: "voice",
    accuracy: 0.93,
    speed: 0.85,
    loaded: true,
    size: "1.2GB",
    description: "Ultra-realistic voice synthesis with emotional tones",
    framework: "pytorch"
  },
  {
    id: "hand-pose-estimator",
    name: "Hand Pose Estimator",
    version: "2.3.0",
    type: "hand",
    accuracy: 0.94,
    speed: 0.96,
    loaded: true,
    size: "445MB",
    description: "21-point hand pose estimation for sign language",
    framework: "mediapipe"
  },
  {
    id: "body-language-analyzer",
    name: "Body Language Analyzer",
    version: "1.0.0",
    type: "pose",
    accuracy: 0.87,
    speed: 0.82,
    loaded: false,
    size: "3.1GB",
    description: "Full-body gesture and body language analysis",
    framework: "pytorch"
  },
  {
    id: "3d-mesh-reconstructor",
    name: "3D Mesh Reconstructor",
    version: "2.0.0",
    type: "mesh",
    accuracy: 0.92,
    speed: 0.78,
    loaded: true,
    size: "1.5GB",
    description: "Real-time 3D face and object mesh reconstruction",
    framework: "tensorflow"
  },
  {
    id: "gesture-recognition-transformer",
    name: "Gesture Recognition Transformer",
    version: "1.8.0",
    type: "sign",
    accuracy: 0.93,
    speed: 0.88,
    loaded: true,
    size: "1.8GB",
    description: "Dynamic gesture recognition with temporal analysis",
    framework: "pytorch"
  },
  {
    id: "biometric-authenticator",
    name: "Biometric Authenticator",
    version: "3.0.0",
    type: "face",
    accuracy: 0.99,
    speed: 0.75,
    loaded: true,
    size: "980MB",
    description: "Liveness detection and anti-spoofing biometric authentication",
    framework: "tensorflow"
  }
];

// MediaPipe Pipeline Configurations
export const MEDIAPIPE_CONFIGURATIONS = {
  holistic: {
    modelComplexity: ['lite', 'full', 'heavy'],
    staticImageMode: false,
    smoothLandmarks: true,
    enableSegmentation: true,
    minDetectionConfidence: 0.5,
    trackingMode: ['single', 'multi']
  },
  faceMesh: {
    maxNumFaces: 4,
    refineLandmarks: true,
    minDetectionConfidence: 0.5,
    enableFaceGeometry: true
  },
  hands: {
    maxNumHands: 2,
    modelComplexity: ['lite', 'full'],
    minDetectionConfidence: 0.5,
    trackingMode: ['single', 'multi']
  },
  pose: {
    modelComplexity: ['lite', 'full', 'heavy'],
    smoothLandmarks: true,
    enableSegmentation: false,
    minDetectionConfidence: 0.5
  }
};

// PyTorch Model Configurations
export const PYTORCH_CONFIGURATIONS = {
  signRecognition: {
    backbone: 'resnet50',
    hiddenLayers: [512, 256, 128],
    numClasses: 1000,
    sequenceLength: 30,
    dropout: 0.2,
    learningRate: 0.001,
    batchSize: 32,
    epochs: 100
  },
  faceAnalysis: {
    backbone: 'efficientnet-b0',
    embeddingSize: 512,
    numLayers: 4,
    attentionHeads: 8,
    imageResolution: [224, 224],
    batchSize: 64,
    epochs: 50
  },
  gestureClassification: {
    backbone: 'mobilenetv3',
    temporalWindow: 16,
    hiddenSize: 256,
    numGestures: 50,
    dropout: 0.3,
    learningRate: 0.0001,
    batchSize: 16,
    epochs: 80
  }
};

// Deep Face Analysis Features
export const DEEP_FACE_FEATURES = {
  biometricAnalysis: [
    'faceShape',
    'skinTone',
    'eyeColor',
    'hairColor',
    'facialAsymmetry',
    'uniqueFeatures',
    'faceQuality',
    'blurScore',
    'illuminationScore'
  ],
  faceShapes: ['round', 'oval', 'square', 'heart', 'triangle'],
  skinTones: ['very_light', 'light', 'medium', 'dark', 'very_dark'],
  eyeColors: ['blue', 'brown', 'green', 'hazel', 'gray', 'amber'],
  hairColors: ['black', 'brown', 'blonde', 'red', 'gray', 'white'],
  qualityMetrics: ['excellent', 'good', 'fair', 'poor']
};

// Advanced Gesture Recognition
export const GESTURE_TYPES = {
  static: ['hello', 'thank_you', 'please', 'yes', 'no', 'goodbye'],
  dynamic: ['wave', 'point', 'thumbs_up', 'thumbs_down', 'peace'],
  sequential: ['counting', 'spelling', 'storytelling'],
  complex: ['compound_gestures', 'facial_expressions', 'body_language']
};

// 3D Mesh Reconstruction Settings
export const MESH_RECONSTRUCTION = {
  resolution: {
    low: { vertices: 468, faces: 936 },
    medium: { vertices: 2330, faces: 4660 },
    high: { vertices: 10938, faces: 21876 }
  },
  rendering: {
    wireframe: true,
    solid: true,
    pointCloud: true,
    textured: true,
    shaded: true
  },
  optimization: {
    levelOfDetail: true,
    frustumCulling: true,
    occlusionCulling: true
  }
};

// Neural Rendering Pipeline
export const NEURAL_RENDERING = {
  shaders: {
    vertex: 'neural_vertex.glsl',
    fragment: 'neural_fragment.glsl',
    compute: 'neural_compute.glsl'
  },
  postProcessing: {
    bloom: true,
    motionBlur: true,
    depthOfField: true,
    neuralStyling: true,
    ssao: true,
    globalIllumination: true
  },
  neuralEffects: {
    styleTransfer: true,
    superResolution: true,
    temporalAntiAliasing: true,
    neuralUpscaling: true
  }
};

// Multi-Modal Fusion System
export const MULTIMODAL_FUSION = {
  strategies: ['early', 'late', 'intermediate', 'attention'],
  weights: {
    face: 0.4,
    pose: 0.3,
    gesture: 0.2,
    emotion: 0.1
  },
  calibration: {
    confidenceThreshold: 0.7,
    fusionLatency: 50,
    temporalSmoothing: 0.8
  }
};

// Biometric Authentication
export const BIOMETRIC_AUTH = {
  methods: ['face', 'voice', 'multimodal'],
  livenessDetection: {
    blinkDetection: true,
    headMovement: true,
    challengeResponse: true,
    depthAnalysis: true
  },
  antiSpoofing: {
    textureAnalysis: true,
    motionAnalysis: true,
    depthConsistency: true,
    heartRateDetection: true
  },
  security: {
    encryptionLevel: 'AES-256',
    sessionTimeout: 300,
    maxAttempts: 3,
    lockoutDuration: 900
  }
};

// Voice Options
export const VOICE_OPTIONS = [
  { id: "aria", name: "Aria", language: "en-US", gender: "female", accent: "american" },
  { id: "david", name: "David", language: "en-US", gender: "male", accent: "american" },
  { id: "emma", name: "Emma", language: "en-GB", gender: "female", accent: "british" },
  { id: "oliver", name: "Oliver", language: "en-GB", gender: "male", accent: "british" },
  { id: "sophia", name: "Sophia", language: "en-AU", gender: "female", accent: "australian" },
  { id: "james", name: "James", language: "en-CA", gender: "male", accent: "canadian" },
  { id: "zara", name: "Zara", language: "en-IN", gender: "female", accent: "indian" },
  { id: "kai", name: "Kai", language: "en-ZA", gender: "male", accent: "south-african" }
];

// Camera Resolutions
export const CAMERA_RESOLUTIONS = [
  { label: "480p", value: "640x480" },
  { label: "720p", value: "1280x720" },
  { label: "1080p", value: "1920x1080" },
  { label: "1440p", value: "2560x1440" },
  { label: "4K", value: "3840x2160" }
];

// Frame Rates
export const FRAME_RATES = [
  { label: "15 FPS", value: 15 },
  { label: "30 FPS", value: 30 },
  { label: "60 FPS", value: 60 },
  { label: "120 FPS", value: 120 },
  { label: "240 FPS", value: 240 }
];

// Quality Settings
export const QUALITY_SETTINGS = [
  { label: "Low", value: 0.3 },
  { label: "Medium", value: 0.5 },
  { label: "High", value: 0.7 },
  { label: "Ultra", value: 0.9 },
  { label: "Maximum", value: 1.0 }
];

// Performance Thresholds
export const PERFORMANCE_THRESHOLDS = {
  MIN_FPS: 15,
  TARGET_FPS: 30,
  MAX_LATENCY: 100, // ms
  TARGET_ACCURACY: 0.85,
  MAX_MEMORY_USAGE: 80, // %
  MAX_GPU_USAGE: 90 // %
};

// Emotion Categories
export const EMOTION_CATEGORIES = [
  { key: "happy", label: "Happy", color: "#10B981", icon: "😊" },
  { key: "sad", label: "Sad", color: "#3B82F6", icon: "😢" },
  { key: "angry", label: "Angry", color: "#EF4444", icon: "😠" },
  { key: "surprised", label: "Surprised", color: "#F59E0B", icon: "😮" },
  { key: "neutral", label: "Neutral", color: "#6B7280", icon: "😐" },
  { key: "fear", label: "Fear", color: "#8B5CF6", icon: "😨" },
  { key: "disgusted", label: "Disgusted", color: "#84CC16", icon: "🤢" }
];

// Age Groups
export const AGE_GROUPS = [
  { key: "child", label: "Child (0-12)", range: [0, 12] },
  { key: "teen", label: "Teen (13-19)", range: [13, 19] },
  { key: "young-adult", label: "Young Adult (20-35)", range: [20, 35] },
  { key: "adult", label: "Adult (36-55)", range: [36, 55] },
  { key: "senior", label: "Senior (56+)", range: [56, 120] }
];

// Theme Colors
export const THEME_COLORS = {
  primary: "#8B5CF6",
  secondary: "#06B6D4",
  accent: "#F59E0B",
  neural: "#10B981",
  cyberpunk: "#EC4899",
  success: "#10B981",
  warning: "#F59E0B",
  error: "#EF4444",
  info: "#3B82F6"
};

// API Endpoints
export const API_ENDPOINTS = {
  translation: "/api/translate",
  faceRecognition: "/api/face-recognition",
  voiceSynthesis: "/api/voice-synthesis",
  emotionDetection: "/api/emotion-detection",
  analytics: "/api/analytics",
  modelManagement: "/api/models",
  sessionManagement: "/api/sessions"
};

// Feature Flags
export const FEATURE_FLAGS = {
  voiceSynthesis: true,
  faceRecognition: true,
  emotionDetection: true,
  ageGenderDetection: true,
  multiUserSessions: true,
  advancedAnalytics: true,
  modelTraining: true,
  cloudSync: true,
  offlineMode: true,
  experimentalFeatures: false
};