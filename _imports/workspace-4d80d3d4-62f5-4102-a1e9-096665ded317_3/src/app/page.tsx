'use client';

import { useState, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Camera, Languages, Brain, Zap, Globe, Shield, Cpu, Wifi, Play, Settings, BarChart3, Users, Smile, Activity, Eye, Hand, Atom, Box, Radar, Dna, Network, Sun } from 'lucide-react';
import { DIGITAL_CERTIFICATE, SUPPORTED_LANGUAGES, ADVANCED_AI_MODELS, THEME_COLORS } from '@/constants';
import { SignLanguage, UserSettings, AnalyticsData, PerformanceMetrics, MediaPipeSettings, PyTorchSettings } from '@/types';
import AdvancedCameraView from '@/components/translation/AdvancedCameraView';
import SettingsPanel from '@/components/translation/SettingsPanel';
import Avatar3D from '@/components/translation/Avatar3D';
import AnalyticsDashboard from '@/components/analytics/AnalyticsDashboard';
import MediaPipeProcessor from '@/components/mediapipe/MediaPipeProcessor';
import PyTorchProcessor from '@/components/pytorch/PyTorchProcessor';
import DeepFaceAnalyzer from '@/components/face-analysis/DeepFaceAnalyzer';
import QuantumTransformerProcessor from '@/components/quantum/QuantumTransformerProcessor';
import NeuromorphicProcessor from '@/components/neuromorphic/NeuromorphicProcessor';
import Holographic4DProcessor from '@/components/holographic/Holographic4DProcessor';
import QuantumSensorFusionProcessor from '@/components/sensors/QuantumSensorFusionProcessor';
import BCIProcessor from '@/components/bci/BCIProcessor';
import NeuralEvolutionSystem from '@/components/evolution/NeuralEvolutionSystem';
import FederatedQuantumLearning from '@/components/federated/FederatedQuantumLearning';
import PhotonicNeuralNetworks from '@/components/photonic/PhotonicNeuralNetworks';
import ClientOnly from '@/components/ui/client-only';

export default function Home() {
  const [selectedLanguage, setSelectedLanguage] = useState<SignLanguage>(SUPPORTED_LANGUAGES[0]);
  const [showTranslation, setShowTranslation] = useState(false);
  const [cameraActive, setCameraActive] = useState(false);
  const [activeTab, setActiveTab] = useState("overview");
  
  const [userSettings, setUserSettings] = useState<UserSettings>({
    preferredLanguage: "ASL",
    autoTranslate: true,
    soundEnabled: true,
    vibrationEnabled: false,
    camera: {
      device: "default",
      resolution: "1280x720",
      frameRate: 30,
      quality: 0.8
    },
    voiceSettings: {
      enabled: true,
      voice: "aria",
      speed: 1.0,
      pitch: 1.0,
      volume: 0.8,
      language: "en-US"
    },
    faceRecognition: {
      enabled: true,
      saveFaces: false,
      alertOnUnknown: true,
      trackingEnabled: true,
      deepAnalysis: true,
      biometricAuth: false
    },
    mediaPipeSettings: {
      enabled: true,
      modelComplexity: 'full',
      minDetectionConfidence: 0.5,
      trackingMode: 'multi',
      smoothLandmarks: true,
      enableSegmentation: true
    },
    pyTorchSettings: {
      enabled: true,
      modelPath: '/models/custom/',
      gpuAcceleration: true,
      batchSize: 16,
      precision: 'fp16',
      customModels: []
    }
  });

  // Mock data for analytics - use stable values to avoid hydration issues
  const [analyticsData] = useState<AnalyticsData>({
    totalTranslations: 1247,
    uniqueFaces: 89,
    averageConfidence: 0.92,
    sessionDuration: 3600,
    mostUsedPhrases: [
      { phrase: "Hello there!", count: 156 },
      { phrase: "Thank you very much", count: 142 },
      { phrase: "How are you doing today?", count: 98 },
      { phrase: "Nice to meet you", count: 87 },
      { phrase: "Can you please help me?", count: 76 }
    ],
    emotionDistribution: {
      happy: 45,
      neutral: 28,
      surprised: 12,
      sad: 8,
      angry: 4,
      fear: 2,
      disgusted: 1
    },
    ageDistribution: {
      "18-25": 23,
      "26-35": 31,
      "36-45": 19,
      "46-55": 12,
      "56+": 4
    },
    genderDistribution: {
      male: 52,
      female: 37,
      unknown: 0
    },
    performanceTrends: Array.from({ length: 20 }, (_, i) => ({
      timestamp: new Date(Date.now() - (19 - i) * 60000),
      fps: 28 + Math.random() * 4,
      latency: 45 + Math.random() * 20,
      accuracy: 0.85 + Math.random() * 0.1
    })),
    gestureComplexity: {
      simple: 45,
      moderate: 30,
      complex: 20,
      advanced: 5
    },
    mediaPipeEfficiency: 0.89,
    pyTorchPerformance: 0.92
  });

  const [performanceHistory] = useState<PerformanceMetrics[]>(analyticsData.performanceTrends);

  const handleStartTranslation = () => {
    setShowTranslation(true);
    setActiveTab("translator");
    setCameraActive(true);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900">
      {/* Navigation */}
      <div className="border-b border-gray-700 bg-black/30 backdrop-blur-md">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 bg-gradient-to-br from-purple-500 to-cyan-500 rounded-xl flex items-center justify-center">
                <Languages className="w-6 h-6 text-white" />
              </div>
              <div>
                <h1 className="text-xl font-bold text-white">SignSync Omni</h1>
                <p className="text-xs text-gray-400">v25.1.0-Quantum-Neural-Ultra-Plus</p>
              </div>
            </div>
            <Tabs value={activeTab} onValueChange={setActiveTab} className="w-auto">
              <TabsList className="bg-black/30 border border-purple-500/20">
              <TabsTrigger value="overview" className="text-white data-[state=active]:bg-purple-600">
                Overview
              </TabsTrigger>
              <TabsTrigger value="translator" className="text-white data-[state=active]:bg-purple-600">
                <Camera className="w-4 h-4 mr-1" />
                Translator
              </TabsTrigger>
              <TabsTrigger value="quantum" className="text-white data-[state=active]:bg-purple-600">
                <Atom className="w-4 h-4 mr-1" />
                Quantum
              </TabsTrigger>
              <TabsTrigger value="neuromorphic" className="text-white data-[state=active]:bg-purple-600">
                <Activity className="w-4 h-4 mr-1" />
                Neuromorphic
              </TabsTrigger>
              <TabsTrigger value="holographic" className="text-white data-[state=active]:bg-purple-600">
                <Box className="w-4 h-4 mr-1" />
                4D Holographic
              </TabsTrigger>
              <TabsTrigger value="sensors" className="text-white data-[state=active]:bg-purple-600">
                <Radar className="w-4 h-4 mr-1" />
                Quantum Sensors
              </TabsTrigger>
              <TabsTrigger value="bci" className="text-white data-[state=active]:bg-purple-600">
                <Brain className="w-4 h-4 mr-1" />
                BCI
              </TabsTrigger>
              <TabsTrigger value="evolution" className="text-white data-[state=active]:bg-purple-600">
                <Dna className="w-4 h-4 mr-1" />
                Evolution
              </TabsTrigger>
              <TabsTrigger value="federated" className="text-white data-[state=active]:bg-purple-600">
                <Network className="w-4 h-4 mr-1" />
                Federated
              </TabsTrigger>
              <TabsTrigger value="photonic" className="text-white data-[state=active]:bg-purple-600">
                <Sun className="w-4 h-4 mr-1" />
                Photonic
              </TabsTrigger>
              <TabsTrigger value="mediapipe" className="text-white data-[state=active]:bg-purple-600">
                <Activity className="w-4 h-4 mr-1" />
                MediaPipe
              </TabsTrigger>
              <TabsTrigger value="pytorch" className="text-white data-[state=active]:bg-purple-600">
                <Zap className="w-4 h-4 mr-1" />
                PyTorch
              </TabsTrigger>
              <TabsTrigger value="face-analysis" className="text-white data-[state=active]:bg-purple-600">
                <Eye className="w-4 h-4 mr-1" />
                Face Analysis
              </TabsTrigger>
              <TabsTrigger value="avatar" className="text-white data-[state=active]:bg-purple-600">
                <Users className="w-4 h-4 mr-1" />
                Avatar
              </TabsTrigger>
              <TabsTrigger value="analytics" className="text-white data-[state=active]:bg-purple-600">
                <BarChart3 className="w-4 h-4 mr-1" />
                Analytics
              </TabsTrigger>
              <TabsTrigger value="settings" className="text-white data-[state=active]:bg-purple-600">
                <Settings className="w-4 h-4 mr-1" />
                Settings
              </TabsTrigger>
            </TabsList>
            </Tabs>
          </div>
        </div>
      </div>

      {/* Main Content */}
      {activeTab === "overview" && (
        <>
          {/* Hero Section */}
          <div className="relative overflow-hidden">
            <div className="absolute inset-0 bg-gradient-to-r from-purple-600/20 to-cyan-600/20 backdrop-blur-3xl" />
            <div className="relative container mx-auto px-4 py-16">
              <div className="text-center space-y-8">
                {/* Logo and Title */}
                <div className="space-y-4">
                  <div className="flex justify-center">
                    <div className="relative">
                      <div className="w-20 h-20 bg-gradient-to-br from-purple-500 to-cyan-500 rounded-2xl flex items-center justify-center shadow-2xl">
                        <Languages className="w-10 h-10 text-white" />
                      </div>
                      <div className="absolute -top-2 -right-2 w-6 h-6 bg-green-500 rounded-full animate-pulse" />
                    </div>
                  </div>
                  <h1 className="text-5xl md:text-7xl font-bold text-white">
                    SignSync<span className="text-transparent bg-clip-text bg-gradient-to-r from-purple-400 to-cyan-400"> Omni</span>
                  </h1>
                  <p className="text-xl md:text-2xl text-gray-300">
                    Version 25.1.0-Quantum-Neural-Ultra-Plus
                  </p>
                  <Badge className="bg-gradient-to-r from-purple-500 to-cyan-500 text-white px-4 py-2">
                    AI-Powered Quantum Sign Language Translation
                  </Badge>
                </div>

                {/* Main CTA */}
                <div className="space-y-4">
                  <Button 
                    size="lg" 
                    className="bg-gradient-to-r from-purple-600 to-cyan-600 hover:from-purple-700 hover:to-cyan-700 text-white px-8 py-4 text-lg shadow-2xl"
                    onClick={handleStartTranslation}
                  >
                    <Play className="w-6 h-6 mr-2" />
                    Start Translation
                  </Button>
                  <p className="text-gray-400">
                    Real-time sign language detection and translation powered by neural networks
                  </p>
                </div>
              </div>
            </div>
          </div>

          {/* Features Section */}
          <div className="container mx-auto px-4 py-16">
            <div className="text-center mb-12">
              <h2 className="text-4xl font-bold text-white mb-4">Quantum-Enhanced Ultra Features</h2>
              <p className="text-xl text-gray-300">Cutting-edge quantum neural technology for next-generation communication</p>
            </div>

            <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6 mb-16">
              <Card className="bg-black/30 backdrop-blur-md border-purple-500/20 hover:border-purple-500/40 transition-all">
                <CardHeader>
                  <Atom className="w-8 h-8 text-purple-400 mb-2" />
                  <CardTitle className="text-white">Quantum Transformers</CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-gray-300 text-sm">
                    Superposition states & quantum entanglement
                  </p>
                  <Badge className="mt-2 bg-purple-600">Quantum Ready</Badge>
                </CardContent>
              </Card>

              <Card className="bg-black/30 backdrop-blur-md border-cyan-500/20 hover:border-cyan-500/40 transition-all">
                <CardHeader>
                  <Activity className="w-8 h-8 text-cyan-400 mb-2" />
                  <CardTitle className="text-white">Neuromorphic Computing</CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-gray-300 text-sm">
                    Spiking neural networks & memristor arrays
                  </p>
                  <Badge className="mt-2 bg-cyan-600">Brain-Like</Badge>
                </CardContent>
              </Card>

              <Card className="bg-black/30 backdrop-blur-md border-green-500/20 hover:border-green-500/40 transition-all">
                <CardHeader>
                  <Box className="w-8 h-8 text-green-400 mb-2" />
                  <CardTitle className="text-white">4D Holographic Mapping</CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-gray-300 text-sm">
                    Volumetric capture & light field imaging
                  </p>
                  <Badge className="mt-2 bg-green-600">Space-Time</Badge>
                </CardContent>
              </Card>

              <Card className="bg-black/30 backdrop-blur-md border-red-500/20 hover:border-red-500/40 transition-all">
                <CardHeader>
                  <Radar className="w-8 h-8 text-red-400 mb-2" />
                  <CardTitle className="text-white">Quantum Sensor Fusion</CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-gray-300 text-sm">
                    Multi-spectral quantum sensors & entanglement
                  </p>
                  <Badge className="mt-2 bg-red-600">Quantum Vision</Badge>
                </CardContent>
              </Card>

              <Card className="bg-black/30 backdrop-blur-md border-pink-500/20 hover:border-pink-500/40 transition-all">
                <CardHeader>
                  <Brain className="w-8 h-8 text-pink-400 mb-2" />
                  <CardTitle className="text-white">Brain-Computer Interface</CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-gray-300 text-sm">
                    Direct neural communication & thought-to-sign
                  </p>
                  <Badge className="mt-2 bg-pink-600">Neural Link</Badge>
                </CardContent>
              </Card>

              <Card className="bg-black/30 backdrop-blur-md border-orange-500/20 hover:border-orange-500/40 transition-all">
                <CardHeader>
                  <Users className="w-8 h-8 text-orange-400 mb-2" />
                  <CardTitle className="text-white">3D Neural Avatar</CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-gray-300 text-sm">
                    Interactive quantum avatar with emotions
                  </p>
                  <Badge className="mt-2 bg-orange-600">Cyberpunk</Badge>
                </CardContent>
              </Card>

              <Card className="bg-black/30 backdrop-blur-md border-blue-500/20 hover:border-blue-500/40 transition-all">
                <CardHeader>
                  <BarChart3 className="w-8 h-8 text-blue-400 mb-2" />
                  <CardTitle className="text-white">Advanced Analytics</CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-gray-300 text-sm">
                    Quantum performance insights & metrics
                  </p>
                  <Badge className="mt-2 bg-blue-600">Real-time</Badge>
                </CardContent>
              </Card>

              <Card className="bg-black/30 backdrop-blur-md border-yellow-500/20 hover:border-yellow-500/40 transition-all">
                <CardHeader>
                  <Activity className="w-8 h-8 text-yellow-400 mb-2" />
                  <CardTitle className="text-white">MediaPipe Ultra</CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-gray-300 text-sm">
                    Enhanced pose & hand tracking
                  </p>
                  <Badge className="mt-2 bg-yellow-600">468 Landmarks</Badge>
                </CardContent>
              </Card>

              <Card className="bg-black/30 backdrop-blur-md border-indigo-500/20 hover:border-indigo-500/40 transition-all">
                <CardHeader>
                  <Zap className="w-8 h-8 text-indigo-400 mb-2" />
                  <CardTitle className="text-white">PyTorch Quantum</CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-gray-300 text-sm">
                    GPU-accelerated quantum inference
                  </p>
                  <Badge className="mt-2 bg-indigo-600">Light Speed</Badge>
                </CardContent>
              </Card>

              <Card className="bg-black/30 backdrop-blur-md border-teal-500/20 hover:border-teal-500/40 transition-all">
                <CardHeader>
                  <Eye className="w-8 h-8 text-teal-400 mb-2" />
                  <CardTitle className="text-white">Deep Face Analysis</CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-gray-300 text-sm">
                    Quantum biometric analysis
                  </p>
                  <Badge className="mt-2 bg-teal-600">3D Scanning</Badge>
                </CardContent>
              </Card>
            </div>

            {/* Digital Certificate */}
            <Card className="max-w-2xl mx-auto bg-black/30 backdrop-blur-md border-purple-500/20">
              <CardHeader>
                <CardTitle className="text-white flex items-center gap-2">
                  <Shield className="w-5 h-5" />
                  Digital Certificate of Ownership
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-2 text-gray-300">
                <p><span className="text-purple-400">Owner:</span> {DIGITAL_CERTIFICATE.owner}</p>
                <p><span className="text-purple-400">Email:</span> {DIGITAL_CERTIFICATE.email}</p>
                <p><span className="text-purple-400">Version:</span> {DIGITAL_CERTIFICATE.productVersion}</p>
                <p><span className="text-purple-400">Issued:</span> {new Date(DIGITAL_CERTIFICATE.issuedAt).toLocaleDateString()}</p>
              </CardContent>
            </Card>
          </div>
        </>
      )}

      {activeTab === "translator" && (
        <div className="container mx-auto px-4 py-8">
          <AdvancedCameraView 
            isActive={cameraActive} 
            onToggle={() => setCameraActive(!cameraActive)}
            settings={userSettings}
            onSettingsChange={setUserSettings}
          />
        </div>
      )}

      {activeTab === "quantum" && (
        <div className="container mx-auto px-4 py-8">
          <ClientOnly fallback={<div className="text-white text-center">Loading Quantum Processor...</div>}>
            <QuantumTransformerProcessor />
          </ClientOnly>
        </div>
      )}

      {activeTab === "neuromorphic" && (
        <div className="container mx-auto px-4 py-8">
          <ClientOnly fallback={<div className="text-white text-center">Loading Neuromorphic Processor...</div>}>
            <NeuromorphicProcessor />
          </ClientOnly>
        </div>
      )}

      {activeTab === "holographic" && (
        <div className="container mx-auto px-4 py-8">
          <ClientOnly fallback={<div className="text-white text-center">Loading Holographic Processor...</div>}>
            <Holographic4DProcessor />
          </ClientOnly>
        </div>
      )}

      {activeTab === "sensors" && (
        <div className="container mx-auto px-4 py-8">
          <ClientOnly fallback={<div className="text-white text-center">Loading Quantum Sensors...</div>}>
            <QuantumSensorFusionProcessor />
          </ClientOnly>
        </div>
      )}

      {activeTab === "bci" && (
        <div className="container mx-auto px-4 py-8">
          <ClientOnly fallback={<div className="text-white text-center">Loading BCI Processor...</div>}>
            <BCIProcessor />
          </ClientOnly>
        </div>
      )}

      {activeTab === "evolution" && (
        <div className="container mx-auto px-4 py-8">
          <ClientOnly fallback={<div className="text-white text-center">Loading Neural Evolution System...</div>}>
            <NeuralEvolutionSystem />
          </ClientOnly>
        </div>
      )}

      {activeTab === "federated" && (
        <div className="container mx-auto px-4 py-8">
          <ClientOnly fallback={<div className="text-white text-center">Loading Federated Quantum Learning...</div>}>
            <FederatedQuantumLearning />
          </ClientOnly>
        </div>
      )}

      {activeTab === "photonic" && (
        <div className="container mx-auto px-4 py-8">
          <ClientOnly fallback={<div className="text-white text-center">Loading Photonic Neural Networks...</div>}>
            <PhotonicNeuralNetworks />
          </ClientOnly>
        </div>
      )}

      {activeTab === "mediapipe" && (
        <div className="container mx-auto px-4 py-8">
          <MediaPipeProcessor 
            isActive={cameraActive} 
            settings={userSettings.mediaPipeSettings}
            onSettingsChange={(settings) => setUserSettings(prev => ({ ...prev, mediaPipeSettings: settings }))}
            onPerformanceUpdate={(metrics) => console.log('MediaPipe Performance:', metrics)}
          />
        </div>
      )}

      {activeTab === "pytorch" && (
        <div className="container mx-auto px-4 py-8">
          <PyTorchProcessor 
            isActive={cameraActive} 
            settings={userSettings.pyTorchSettings}
            onSettingsChange={(settings) => setUserSettings(prev => ({ ...prev, pyTorchSettings: settings }))}
            onPerformanceUpdate={(metrics) => console.log('PyTorch Performance:', metrics)}
            onModelUpdate={(models) => console.log('Custom Models:', models)}
          />
        </div>
      )}

      {activeTab === "face-analysis" && (
        <div className="container mx-auto px-4 py-8">
          <DeepFaceAnalyzer 
            isActive={cameraActive}
            onAuthenticationUpdate={(auth) => console.log('Authentication:', auth)}
            onPerformanceUpdate={(metrics) => console.log('Face Analysis Performance:', metrics)}
          />
        </div>
      )}

      {activeTab === "avatar" && (
        <div className="container mx-auto px-4 py-8">
          <ClientOnly fallback={<div className="text-white text-center">Loading Avatar...</div>}>
            <Avatar3D 
              isActive={cameraActive}
              detectedFaces={[]}
              currentTranslation=""
              isSpeaking={false}
            />
          </ClientOnly>
        </div>
      )}

      {activeTab === "analytics" && (
        <div className="container mx-auto px-4 py-8">
          <ClientOnly fallback={<div className="text-white text-center">Loading Analytics...</div>}>
            <AnalyticsDashboard 
              data={analyticsData}
              performanceHistory={performanceHistory}
            />
          </ClientOnly>
        </div>
      )}

      {activeTab === "settings" && (
        <div className="container mx-auto px-4 py-8">
          <SettingsPanel settings={userSettings} onSettingsChange={setUserSettings} />
        </div>
      )}

      {/* Footer */}
      <div className="border-t border-gray-700 bg-black/30 backdrop-blur-md mt-auto">
        <div className="container mx-auto px-4 py-8">
          <div className="text-center text-gray-400">
            <p>© 2024 SignSync Omni V10 - Neural Sign Language Translation</p>
            <p className="text-sm mt-2">Intellectually owned by Nagabhushana Raju S</p>
          </div>
        </div>
      </div>
    </div>
  );
}