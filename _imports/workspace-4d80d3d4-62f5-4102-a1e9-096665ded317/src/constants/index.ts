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

// Advanced AI Models
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
    lastTrained: new Date("2024-10-15")
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
    lastTrained: new Date("2024-11-01")
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
    lastTrained: new Date("2024-09-20")
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
    description: "Age and gender detection with demographic analysis"
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
    description: "Ultra-realistic voice synthesis with emotional tones"
  },
  {
    id: "hand-pose-estimator",
    name: "Hand Pose Estimator",
    version: "2.3.0",
    type: "sign",
    accuracy: 0.94,
    speed: 0.96,
    loaded: true,
    size: "445MB",
    description: "21-point hand pose estimation for sign language"
  },
  {
    id: "body-language-analyzer",
    name: "Body Language Analyzer",
    version: "1.0.0",
    type: "sign",
    accuracy: 0.87,
    speed: 0.82,
    loaded: false,
    size: "3.1GB",
    description: "Full-body gesture and body language analysis"
  }
];

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