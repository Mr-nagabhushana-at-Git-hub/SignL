'use client';

import { useState, useEffect, useRef } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { 
  Fingerprint, 
  Brain, 
  Zap, 
  Activity, 
  TrendingUp, 
  User,
  Shield,
  Eye,
  Key,
  Atom
} from 'lucide-react';

interface QuantumBiometric {
  id: string;
  name: string;
  type: 'quantum-fingerprint' | 'neural-signature' | 'consciousness-pattern' | 'quantum-iris' | 'quantum-dna';
  accuracy: number;
  quantumCoherence: number;
  entropy: number;
  uniqueness: number;
  securityLevel: number;
}

interface BiometricScan {
  id: string;
  timestamp: number;
  biometricType: string;
  quantumState: number;
  confidence: number;
  entropy: number;
  matches: number;
  falsePositives: number;
  verificationTime: number;
}

interface QuantumSignature {
  id: string;
  userId: string;
  quantumPattern: number[];
  consciousnessFingerprint: string;
  neuralFrequency: number;
  quantumEntanglement: number;
  biometricHash: string;
  securityClearance: string;
}

interface BiometricMetrics {
  totalScans: number;
  verificationAccuracy: number;
  averageVerificationTime: number;
  quantumCoherence: number;
  securityLevel: number;
  falseAcceptanceRate: number;
  biometricDiversity: number;
}

export default function QuantumBiometrics() {
  const [isScanning, setIsScanning] = useState(false);
  const [selectedBiometric, setSelectedBiometric] = useState('quantum-fingerprint');
  const [biometrics, setBiometrics] = useState<QuantumBiometric[]>([]);
  const [scans, setScans] = useState<BiometricScan[]>([]);
  const [signatures, setSignatures] = useState<QuantumSignature[]>([]);
  const [metrics, setMetrics] = useState<BiometricMetrics>({
    totalScans: 0,
    verificationAccuracy: 0,
    averageVerificationTime: 0,
    quantumCoherence: 0,
    securityLevel: 0,
    falseAcceptanceRate: 0,
    biometricDiversity: 0
  });

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationRef = useRef<number>();

  useEffect(() => {
    initializeQuantumBiometrics();
    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, []);

  const initializeQuantumBiometrics = async () => {
    console.log('🔐 Initializing Quantum Biometrics System...');
    
    // Initialize quantum biometrics
    const initialBiometrics: QuantumBiometric[] = [
      {
        id: 'quantum-fingerprint',
        name: 'Quantum Fingerprint Scanner',
        type: 'quantum-fingerprint',
        accuracy: 0.97,
        quantumCoherence: 0.92,
        entropy: 0.85,
        uniqueness: 0.99,
        securityLevel: 8
      },
      {
        id: 'neural-signature',
        name: 'Neural Signature Analyzer',
        type: 'neural-signature',
        accuracy: 0.95,
        quantumCoherence: 0.88,
        entropy: 0.91,
        uniqueness: 0.98,
        securityLevel: 9
      },
      {
        id: 'consciousness-pattern',
        name: 'Consciousness Pattern Reader',
        type: 'consciousness-pattern',
        accuracy: 0.93,
        quantumCoherence: 0.95,
        entropy: 0.94,
        uniqueness: 0.97,
        securityLevel: 10
      },
      {
        id: 'quantum-iris',
        name: 'Quantum Iris Scanner',
        type: 'quantum-iris',
        accuracy: 0.96,
        quantumCoherence: 0.87,
        entropy: 0.82,
        uniqueness: 0.98,
        securityLevel: 7
      },
      {
        id: 'quantum-dna',
        name: 'Quantum DNA Sequencer',
        type: 'quantum-dna',
        accuracy: 0.99,
        quantumCoherence: 0.94,
        entropy: 0.89,
        uniqueness: 1.0,
        securityLevel: 10
      }
    ];

    // Initialize biometric scans
    const initialScans: BiometricScan[] = Array.from({ length: 20 }, (_, i) => ({
      id: `scan-${i}`,
      timestamp: Date.now() - (20 - i) * 5000,
      biometricType: initialBiometrics[Math.floor(Math.random() * initialBiometrics.length)].type,
      quantumState: Math.random(),
      confidence: 0.7 + Math.random() * 0.3,
      entropy: Math.random(),
      matches: Math.floor(Math.random() * 5),
      falsePositives: Math.random() > 0.9 ? 1 : 0,
      verificationTime: 100 + Math.random() * 500
    }));

    // Initialize quantum signatures
    const initialSignatures: QuantumSignature[] = [
      {
        id: 'user-1',
        userId: 'User Alpha',
        quantumPattern: Array.from({ length: 32 }, () => Math.random()),
        consciousnessFingerprint: 'alpha-wave-dominant',
        neuralFrequency: 9.5 + Math.random() * 2,
        quantumEntanglement: 0.87,
        biometricHash: 'QX7B9C2D5E8F1A3G6H4I2J9K0L',
        securityClearance: 'Level 5 - Quantum Access'
      },
      {
        id: 'user-2',
        userId: 'User Beta',
        quantumPattern: Array.from({ length: 32 }, () => Math.random()),
        consciousnessFingerprint: 'theta-wave-enhanced',
        neuralFrequency: 6.2 + Math.random() * 1.5,
        quantumEntanglement: 0.91,
        biometricHash: 'R3M7N1P9Q2S8T4U6V5W8X',
        securityClearance: 'Level 8 - Neural Prime'
      },
      {
        id: 'user-3',
        userId: 'User Gamma',
        quantumPattern: Array.from({ length: 32 }, () => Math.random()),
        consciousnessFingerprint: 'delta-wave-synchronized',
        neuralFrequency: 3.8 + Math.random() * 1.2,
        quantumEntanglement: 0.94,
        biometricHash: 'S5K2L8M3N7Q1P6R9T4U',
        securityClearance: 'Level 10 - Quantum Master'
      }
    ];

    setBiometrics(initialBiometrics);
    setScans(initialScans);
    setSignatures(initialSignatures);
    
    startBiometricProcessing();
    
    console.log('✅ Quantum Biometrics initialized with', initialBiometrics.length, 'biometric types');
  };

  const startBiometricProcessing = () => {
    const process = () => {
      // Update biometric scans
      setScans(prevScans => {
        const newScan: BiometricScan = {
          id: `scan-${Date.now()}`,
          timestamp: Date.now(),
          biometricType: biometrics[Math.floor(Math.random() * biometrics.length)]?.type || 'quantum-fingerprint',
          quantumState: Math.random(),
          confidence: 0.7 + Math.random() * 0.3,
          entropy: Math.random(),
          matches: Math.floor(Math.random() * 5),
          falsePositives: Math.random() > 0.92 ? 1 : 0,
          verificationTime: 100 + Math.random() * 500
        };
        
        return [...prevScans.slice(-50), newScan];
      });

      // Update quantum signatures
      setSignatures(prevSignatures => 
        prevSignatures.map(signature => ({
          ...signature,
          quantumPattern: signature.quantumPattern.map(pattern => 
            Math.max(0, Math.min(1, pattern + (Math.random() - 0.5) * 0.05))
          ),
          neuralFrequency: Math.max(0.1, Math.min(15, signature.neuralFrequency + (Math.random() - 0.5) * 0.2)),
          quantumEntanglement: Math.max(0, Math.min(1, signature.quantumEntanglement + (Math.random() - 0.5) * 0.02))
        }))
      );

      // Update metrics
      setMetrics(prev => ({
        ...prev,
        totalScans: prev.totalScans + 1,
        verificationAccuracy: Math.min(0.99, prev.verificationAccuracy + 0.0005),
        averageVerificationTime: Math.max(50, prev.averageVerificationTime + (Math.random() - 0.5) * 10),
        quantumCoherence: Math.min(0.99, prev.quantumCoherence + 0.0003),
        securityLevel: Math.min(10, prev.securityLevel + (Math.random() - 0.8) * 0.1),
        falseAcceptanceRate: Math.max(0.001, prev.falseAcceptanceRate + (Math.random() - 0.95) * 0.0001),
        biometricDiversity: Math.min(0.99, prev.biometricDiversity + 0.0008)
      }));

      animationRef.current = requestAnimationFrame(process);
    };
    process();
  };

  const performQuantumBiometricScan = async () => {
    setIsScanning(true);
    
    // Simulate quantum biometric scan
    await new Promise(resolve => setTimeout(resolve, 2500));
    
    const selectedBiometricData = biometrics.find(b => b.type === selectedBiometric);
    
    console.log('🔐 Quantum Biometric Scan Complete:', {
      type: selectedBiometric,
      accuracy: selectedBiometricData?.accuracy,
      quantumCoherence: selectedBiometricData?.quantumCoherence,
      securityLevel: selectedBiometricData?.securityLevel
    });
    
    setIsScanning(false);
  };

  const renderQuantumBiometricVisualization = () => {
    if (!canvasRef.current) return;
    
    const ctx = canvasRef.current.getContext('2d');
    if (!ctx) return;
    
    const canvas = canvasRef.current;
    canvas.width = canvas.offsetWidth;
    canvas.height = canvas.offsetHeight;
    
    // Clear canvas
    ctx.fillStyle = 'rgba(0, 0, 0, 0.1)';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    const centerX = canvas.width / 2;
    const centerY = canvas.height / 2;
    const radius = Math.min(canvas.width, canvas.height) / 3;
    
    // Draw quantum field
    for (let i = 0; i < 6; i++) {
      const fieldRadius = radius * (1 + i * 0.15);
      const alpha = 0.2 - i * 0.03;
      
      ctx.strokeStyle = `rgba(100, 200, 255, ${alpha})`;
      ctx.lineWidth = 2;
      ctx.setLineDash([8, 12]);
      ctx.beginPath();
      ctx.arc(centerX, centerY, fieldRadius, 0, Math.PI * 2);
      ctx.stroke();
      ctx.setLineDash([]);
    }
    
    // Draw biometric sensors
    biometrics.forEach((biometric, index) => {
      const angle = (index / biometrics.length) * Math.PI * 2 - Math.PI / 2;
      const distance = (biometric.accuracy * radius * 0.7) + radius * 0.3;
      const x = centerX + Math.cos(angle) * distance;
      const y = centerY + Math.sin(angle) * distance;
      
      // Draw sensor
      const gradient = ctx.createRadialGradient(x, y, 0, x, y, 20);
      gradient.addColorStop(0, `rgba(100, 200, 255, ${biometric.quantumCoherence})`);
      gradient.addColorStop(1, `rgba(100, 200, 255, 0)`);
      
      ctx.fillStyle = gradient;
      ctx.beginPath();
      ctx.arc(x, y, 12 + biometric.accuracy * 8, 0, Math.PI * 2);
      ctx.fill();
      
      // Draw quantum connections
      biometrics.forEach((otherBiometric, otherIndex) => {
        if (index < otherIndex) {
          const otherAngle = (otherIndex / biometrics.length) * Math.PI * 2 - Math.PI / 2;
          const otherDistance = (otherBiometric.accuracy * radius * 0.7) + radius * 0.3;
          const otherX = centerX + Math.cos(otherAngle) * otherDistance;
          const otherY = centerY + Math.sin(otherAngle) * otherDistance;
          
          const quantumEntanglement = Math.min(biometric.quantumCoherence, otherBiometric.quantumCoherence);
          
          ctx.strokeStyle = `rgba(100, 200, 255, ${quantumEntanglement * 0.4})`;
          ctx.lineWidth = quantumEntanglement * 3;
          ctx.beginPath();
          ctx.moveTo(x, y);
          ctx.lineTo(otherX, otherY);
          ctx.stroke();
        }
      });
    });
    
    // Draw central quantum processor
    const processorGradient = ctx.createRadialGradient(centerX, centerY, 0, centerX, centerY, 35);
    processorGradient.addColorStop(0, `rgba(255, 255, 255, ${metrics.quantumCoherence})`);
    processorGradient.addColorStop(1, `rgba(255, 255, 255, 0.2)`);
    
    ctx.fillStyle = processorGradient;
    ctx.beginPath();
    ctx.arc(centerX, centerY, 35, 0, Math.PI * 2);
    ctx.fill();
    
    // Draw quantum symbol
    ctx.fillStyle = '#64c8ff';
    ctx.font = '24px Arial';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText('🔐', centerX, centerY);
  };

  useEffect(() => {
    renderQuantumBiometricVisualization();
  }, [biometrics, metrics]);

  const getBiometricTypeColor = (type: string) => {
    const colors: Record<string, string> = {
      'quantum-fingerprint': 'bg-blue-600',
      'neural-signature': 'bg-purple-600',
      'consciousness-pattern': 'bg-green-600',
      'quantum-iris': 'bg-orange-600',
      'quantum-dna': 'bg-red-600'
    };
    return colors[type] || 'bg-gray-600';
  };

  const getSecurityLevelColor = (level: number) => {
    if (level >= 9) return 'bg-red-600';
    if (level >= 7) return 'bg-orange-600';
    if (level >= 5) return 'bg-yellow-600';
    return 'bg-green-600';
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-2xl font-bold text-white flex items-center gap-2">
            <Fingerprint className="w-6 h-6 text-blue-400" />
            Quantum Biometrics
          </h3>
          <p className="text-gray-400">Quantum-Enhanced Identity Verification</p>
        </div>
        <div className="flex gap-2">
          <Button
            onClick={performQuantumBiometricScan}
            disabled={isScanning}
            className="bg-blue-600 hover:bg-blue-700"
          >
            {isScanning ? (
              <>
                <Brain className="w-4 h-4 mr-2 animate-spin" />
                Scanning Quantum State...
              </>
            ) : (
              <>
                <Zap className="w-4 h-4 mr-2" />
                Perform Quantum Scan
              </>
            )}
          </Button>
        </div>
      </div>

      {/* Biometric Metrics */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <Card className="bg-black/30 border-blue-500/20">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-gray-400 text-sm">Total Scans</p>
                <p className="text-2xl font-bold text-white">{metrics.totalScans}</p>
              </div>
              <Fingerprint className="w-8 h-8 text-blue-400" />
            </div>
          </CardContent>
        </Card>

        <Card className="bg-black/30 border-green-500/20">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-gray-400 text-sm">Verification Accuracy</p>
                <p className="text-2xl font-bold text-white">{(metrics.verificationAccuracy * 100).toFixed(1)}%</p>
              </div>
              <TrendingUp className="w-8 h-8 text-green-400" />
            </div>
          </CardContent>
        </Card>

        <Card className="bg-black/30 border-purple-500/20">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-gray-400 text-sm">Quantum Coherence</p>
                <p className="text-2xl font-bold text-white">{(metrics.quantumCoherence * 100).toFixed(1)}%</p>
              </div>
              <Atom className="w-8 h-8 text-purple-400" />
            </div>
          </CardContent>
        </Card>

        <Card className="bg-black/30 border-orange-500/20">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-gray-400 text-sm">Security Level</p>
                <p className="text-2xl font-bold text-white">{metrics.securityLevel}/10</p>
              </div>
              <Shield className="w-8 h-8 text-orange-400" />
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Main Content */}
      <div className="grid md:grid-cols-2 gap-6">
        {/* Quantum Field Visualization */}
        <Card className="bg-black/30 border-gray-700">
          <CardHeader>
            <CardTitle className="text-white flex items-center gap-2">
              <Eye className="w-5 h-5 text-blue-400" />
              Quantum Biometric Field
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="relative aspect-video bg-gradient-to-br from-blue-900/20 to-purple-900/20 rounded-lg overflow-hidden">
              <canvas
                ref={canvasRef}
                className="w-full h-full"
                style={{ imageRendering: 'crisp-edges' }}
              />
              
              {/* Biometric Status */}
              <div className="absolute top-4 left-4 bg-black/70 backdrop-blur-sm rounded-lg px-3 py-2">
                <div className="text-blue-400 text-sm font-medium">
                  Selected Biometric: {selectedBiometric.replace('-', ' ')}
                </div>
                <div className="text-white text-xs">
                  False Acceptance: {(metrics.falseAcceptanceRate * 100).toFixed(3)}%
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Biometric Types */}
        <Card className="bg-black/30 border-gray-700">
          <CardHeader>
            <CardTitle className="text-white flex items-center gap-2">
              <Key className="w-5 h-5 text-purple-400" />
              Quantum Biometric Types
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3 max-h-80 overflow-y-auto">
              {biometrics.map(biometric => (
                <div key={biometric.id} className="p-3 bg-black/20 border border-gray-600 rounded-lg">
                  <div className="flex items-center justify-between mb-2">
                    <h4 className="text-white font-medium">{biometric.name}</h4>
                    <Badge className={getBiometricTypeColor(biometric.type)}>
                      {biometric.type.replace('-', ' ')}
                    </Badge>
                  </div>
                  <div className="grid grid-cols-2 gap-2 text-xs">
                    <div className="flex justify-between">
                      <span className="text-gray-400">Accuracy:</span>
                      <span className="text-white">{(biometric.accuracy * 100).toFixed(1)}%</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Quantum Coherence:</span>
                      <span className="text-white">{(biometric.quantumCoherence * 100).toFixed(1)}%</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Entropy:</span>
                      <span className="text-white">{(biometric.entropy * 100).toFixed(1)}%</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Uniqueness:</span>
                      <span className="text-white">{(biometric.uniqueness * 100).toFixed(1)}%</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Security Level:</span>
                      <span className="text-white">{biometric.securityLevel}/10</span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Quantum Signatures */}
      <Card className="bg-black/30 border-gray-700">
        <CardHeader>
          <CardTitle className="text-white flex items-center gap-2">
            <User className="w-5 h-5 text-green-400" />
            Quantum User Signatures
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {signatures.map(signature => (
              <div key={signature.id} className="p-4 bg-black/20 border border-gray-600 rounded-lg">
                <div className="flex items-center justify-between mb-2">
                  <h4 className="text-white font-medium">{signature.userId}</h4>
                  <Badge className={getSecurityLevelColor(parseInt(signature.securityClearance.split(' ')[1]))}>
                    {signature.securityClearance}
                  </Badge>
                </div>
                <div className="text-gray-300 text-sm mb-2">
                  {signature.consciousnessFingerprint}
                </div>
                <div className="space-y-1 text-xs">
                  <div className="flex justify-between">
                    <span className="text-gray-400">Neural Frequency:</span>
                    <span className="text-white">{signature.neuralFrequency.toFixed(1)} Hz</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Quantum Entanglement:</span>
                    <span className="text-white">{(signature.quantumEntanglement * 100).toFixed(1)}%</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Pattern Stability:</span>
                    <span className="text-white">{(signature.quantumPattern.reduce((sum, p) => sum + Math.abs(p - 0.5), 0) / signature.quantumPattern.length * 100).toFixed(1)}%</span>
                  </div>
                  <div className="mt-2 p-2 bg-black/30 rounded font-mono text-xs text-green-300 break-all">
                    {signature.biometricHash}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Recent Scans */}
      <Card className="bg-black/30 border-gray-700">
        <CardHeader>
          <CardTitle className="text-white flex items-center gap-2">
            <Activity className="w-5 h-5 text-orange-400" />
            Recent Quantum Scans
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-2 max-h-60 overflow-y-auto">
            {scans.slice(-10).map(scan => (
              <div key={scan.id} className="p-3 bg-black/20 border border-gray-600 rounded-lg">
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center gap-2">
                    <Badge className={getBiometricTypeColor(scan.biometricType)}>
                      {scan.biometricType.replace('-', ' ')}
                    </Badge>
                    <Badge className={scan.confidence > 0.8 ? 'bg-green-600' : 'bg-yellow-600'}>
                      {(scan.confidence * 100).toFixed(1)}%
                    </Badge>
                  </div>
                  <span className="text-gray-400 text-xs">
                    {new Date(scan.timestamp).toLocaleTimeString()}
                  </span>
                </div>
                <div className="grid grid-cols-3 gap-2 text-xs">
                  <div className="flex justify-between">
                    <span className="text-gray-400">Quantum State:</span>
                    <span className="text-white">{scan.quantumState.toFixed(3)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Entropy:</span>
                    <span className="text-white">{(scan.entropy * 100).toFixed(1)}%</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Verify Time:</span>
                    <span className="text-white">{scan.verificationTime.toFixed(0)}ms</span>
                  </div>
                </div>
                {scan.falsePositives > 0 && (
                  <div className="mt-2 p-2 bg-red-900/30 rounded text-red-300 text-xs">
                    ⚠️ {scan.falsePositives} False Positive{scan.falsePositives > 1 ? 's' : ''}
                  </div>
                )}
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Status Badge */}
      <div className="flex justify-center">
        <Badge className={isScanning ? "bg-blue-600 animate-pulse" : "bg-gray-600"}>
          {isScanning ? "Quantum Scanning Active" : "Biometric System Idle"}
        </Badge>
      </div>
    </div>
  );
}