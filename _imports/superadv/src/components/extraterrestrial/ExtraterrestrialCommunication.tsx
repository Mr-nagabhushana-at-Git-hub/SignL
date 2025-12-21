'use client';

import { useState, useEffect, useRef } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { 
  Rocket, 
  Brain, 
  Zap, 
  Activity, 
  TrendingUp, 
  Satellite,
  Radio,
  Waves,
  Star,
  Globe,
  Sparkles
} from 'lucide-react';

interface AlienProtocol {
  id: string;
  species: string;
  origin: string;
  communicationMethod: 'telepathic' | 'quantum' | 'frequency' | 'symbolic' | 'mathematical';
  signalComplexity: number;
  dimensionalAccess: number;
  consciousnessLevel: number;
  translationDifficulty: number;
}

interface CosmicSignal {
  id: string;
  frequency: number; // Hz
  amplitude: number;
  phase: number;
  modulation: string;
  origin: string;
  signalType: string;
  quantumSignature: number;
  temporalAnomaly: number;
}

interface TranslationMatrix {
  fromSpecies: string;
  toSpecies: string;
  universalConcept: string;
  translationAccuracy: number;
  dimensionalBridging: number;
  consciousnessCompatibility: number;
  quantumEntanglement: number;
}

interface ExtraterrestrialMetrics {
  totalSpecies: number;
  activeSignals: number;
  translationAccuracy: number;
  dimensionalBridging: number;
  quantumCoherence: number;
  consciousnessInterface: number;
  cosmicAwareness: number;
}

export default function ExtraterrestrialCommunication() {
  const [isTranslating, setIsTranslating] = useState(false);
  const [selectedSpecies, setSelectedSpecies] = useState('human');
  const [targetDimension, setTargetDimension] = useState('3d');
  const [protocols, setProtocols] = useState<AlienProtocol[]>([]);
  const [signals, setSignals] = useState<CosmicSignal[]>([]);
  const [translationMatrix, setTranslationMatrix] = useState<TranslationMatrix[]>([]);
  const [metrics, setMetrics] = useState<ExtraterrestrialMetrics>({
    totalSpecies: 0,
    activeSignals: 0,
    translationAccuracy: 0,
    dimensionalBridging: 0,
    quantumCoherence: 0,
    consciousnessInterface: 0,
    cosmicAwareness: 0
  });

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationRef = useRef<number>();

  useEffect(() => {
    initializeExtraterrestrialCommunication();
    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, []);

  const initializeExtraterrestrialCommunication = async () => {
    console.log('🛸 Initializing Extraterrestrial Communication System...');
    
    // Initialize alien protocols
    const initialProtocols: AlienProtocol[] = [
      {
        id: 'human-protocol',
        species: 'Human (Earth)',
        origin: 'Sol System, Planet 3',
        communicationMethod: 'symbolic',
        signalComplexity: 0.75,
        dimensionalAccess: 3,
        consciousnessLevel: 0.68,
        translationDifficulty: 0.45
      },
      {
        id: 'arcturian-protocol',
        species: 'Arcturian (Arcturus)',
        origin: 'Boötes Constellation',
        communicationMethod: 'telepathic',
        signalComplexity: 0.92,
        dimensionalAccess: 5,
        consciousnessLevel: 0.95,
        translationDifficulty: 0.78
      },
      {
        id: 'pleiadian-protocol',
        species: 'Pleiadian (Pleiades)',
        origin: 'Taurus Constellation',
        communicationMethod: 'quantum',
        signalComplexity: 0.88,
        dimensionalAccess: 4,
        consciousnessLevel: 0.87,
        translationDifficulty: 0.65
      },
      {
        id: 'zeta-reticuli-protocol',
        species: 'Zeta Reticuli',
        origin: 'Reticulum Constellation',
        communicationMethod: 'frequency',
        signalComplexity: 0.95,
        dimensionalAccess: 6,
        consciousnessLevel: 0.91,
        translationDifficulty: 0.82
      },
      {
        id: 'andromedan-protocol',
        species: 'Andromedan Council',
        origin: 'Andromeda Galaxy',
        communicationMethod: 'mathematical',
        signalComplexity: 0.98,
        dimensionalAccess: 7,
        consciousnessLevel: 0.96,
        translationDifficulty: 0.91
      },
      {
        id: 'draconian-protocol',
        species: 'Draconian (Alpha Draconis)',
        origin: 'Draconis Constellation',
        communicationMethod: 'quantum',
        signalComplexity: 0.85,
        dimensionalAccess: 4,
        consciousnessLevel: 0.79,
        translationDifficulty: 0.71
      },
      {
        id: 'venusian-protocol',
        species: 'Venusian (Venus)',
        origin: 'Sol System, Planet 2',
        communicationMethod: 'telepathic',
        signalComplexity: 0.81,
        dimensionalAccess: 4,
        consciousnessLevel: 0.84,
        translationDifficulty: 0.58
      },
      {
        id: 'sirian-protocol',
        species: 'Sirian (Sirius)',
        origin: 'Canis Major Constellation',
        communicationMethod: 'frequency',
        signalComplexity: 0.89,
        dimensionalAccess: 5,
        consciousnessLevel: 0.92,
        translationDifficulty: 0.74
      }
    ];

    // Initialize cosmic signals
    const initialSignals: CosmicSignal[] = Array.from({ length: 50 }, (_, i) => ({
      id: `cosmic-signal-${i}`,
      frequency: 1000 + Math.random() * 9000, // Cosmic frequencies
      amplitude: Math.random(),
      phase: Math.random() * Math.PI * 2,
      modulation: ['quantum', 'telepathic', 'frequency', 'symbolic'][Math.floor(Math.random() * 4)],
      origin: initialProtocols[Math.floor(Math.random() * initialProtocols.length)].species,
      signalType: ['consciousness', 'mathematical', 'emotional', 'dimensional'][Math.floor(Math.random() * 4)],
      quantumSignature: Math.random(),
      temporalAnomaly: Math.random()
    }));

    // Initialize translation matrix
    const initialMatrix: TranslationMatrix[] = [];
    initialProtocols.forEach(from => {
      initialProtocols.forEach(to => {
        if (from.id !== to.id) {
          const dimensionalDiff = Math.abs(from.dimensionalAccess - to.dimensionalAccess);
          const consciousnessDiff = Math.abs(from.consciousnessLevel - to.consciousnessLevel);
          const complexityDiff = Math.abs(from.signalComplexity - to.signalComplexity);
          
          const baseAccuracy = Math.max(0.1, 1 - (dimensionalDiff + consciousnessDiff + complexityDiff) / 3);
          const dimensionalBridging = Math.max(0.1, 1 - dimensionalDiff / 6);
          const consciousnessCompatibility = Math.max(0.1, 1 - consciousnessDiff);
          const quantumEntanglement = Math.random() * 0.8 + 0.2;
          
          initialMatrix.push({
            fromSpecies: from.species,
            toSpecies: to.species,
            universalConcept: generateUniversalConcept(from, to),
            translationAccuracy: baseAccuracy,
            dimensionalBridging,
            consciousnessCompatibility,
            quantumEntanglement
          });
        }
      });
    });

    setProtocols(initialProtocols);
    setSignals(initialSignals);
    setTranslationMatrix(initialMatrix);
    
    // Calculate metrics
    const totalSpecies = initialProtocols.length;
    const avgAccuracy = initialMatrix.reduce((sum, m) => sum + m.translationAccuracy, 0) / initialMatrix.length;
    const avgDimensionalBridging = initialMatrix.reduce((sum, m) => sum + m.dimensionalBridging, 0) / initialMatrix.length;
    const avgQuantumCoherence = initialMatrix.reduce((sum, m) => sum + m.quantumEntanglement, 0) / initialMatrix.length;
    
    setMetrics({
      totalSpecies,
      activeSignals: initialSignals.filter(s => s.quantumSignature > 0.7).length,
      translationAccuracy: avgAccuracy,
      dimensionalBridging: avgDimensionalBridging,
      quantumCoherence: avgQuantumCoherence,
      consciousnessInterface: 0.78,
      cosmicAwareness: 0.82
    });
    
    startCosmicProcessing();
    
    console.log('✅ Extraterrestrial Communication initialized with', totalSpecies, 'species');
  };

  const generateUniversalConcept = (from: AlienProtocol, to: AlienProtocol): string => {
    const concepts = [
      'consciousness', 'love', 'unity', 'creation', 'knowledge', 'harmony', 
      'evolution', 'dimension', 'frequency', 'vibration', 'existence'
    ];
    return concepts[Math.floor(Math.random() * concepts.length)];
  };

  const startCosmicProcessing = () => {
    const process = () => {
      // Update cosmic signals
      setSignals(prevSignals => {
        const newSignals = prevSignals.map(signal => ({
          ...signal,
          frequency: signal.frequency + (Math.random() - 0.5) * 100,
          amplitude: Math.max(0.1, Math.min(1, signal.amplitude + (Math.random() - 0.5) * 0.1)),
          phase: (signal.phase + 0.01) % (Math.PI * 2),
          quantumSignature: Math.max(0, Math.min(1, signal.quantumSignature + (Math.random() - 0.5) * 0.05)),
          temporalAnomaly: Math.max(0, Math.min(1, signal.temporalAnomaly + (Math.random() - 0.5) * 0.02))
        }));
        
        // Occasionally add new cosmic signals
        if (Math.random() > 0.95) {
          const newSignal: CosmicSignal = {
            id: `cosmic-signal-${Date.now()}`,
            frequency: 1000 + Math.random() * 9000,
            amplitude: Math.random(),
            phase: Math.random() * Math.PI * 2,
            modulation: ['quantum', 'telepathic', 'frequency', 'symbolic'][Math.floor(Math.random() * 4)],
            origin: protocols[Math.floor(Math.random() * protocols.length)]?.species || 'Unknown',
            signalType: ['consciousness', 'mathematical', 'emotional', 'dimensional'][Math.floor(Math.random() * 4)],
            quantumSignature: Math.random(),
            temporalAnomaly: Math.random()
          };
          return [...newSignals.slice(-49), newSignal];
        }
        
        return newSignals;
      });

      // Update metrics
      setMetrics(prev => ({
        ...prev,
        activeSignals: signals.filter(s => s.quantumSignature > 0.7).length,
        translationAccuracy: Math.min(0.99, prev.translationAccuracy + 0.0005),
        dimensionalBridging: Math.min(0.99, prev.dimensionalBridging + 0.0008),
        quantumCoherence: Math.min(0.99, prev.quantumCoherence + 0.0003),
        consciousnessInterface: Math.min(0.99, prev.consciousnessInterface + 0.0006),
        cosmicAwareness: Math.min(0.99, prev.cosmicAwareness + 0.0004)
      }));

      animationRef.current = requestAnimationFrame(process);
    };
    process();
  };

  const performExtraterrestrialTranslation = async () => {
    setIsTranslating(true);
    
    // Simulate extraterrestrial translation
    await new Promise(resolve => setTimeout(resolve, 3500));
    
    const fromProtocol = protocols.find(p => p.id === selectedSpecies);
    const translations = translationMatrix.filter(t => t.fromSpecies === fromProtocol?.species);
    
    console.log('🛸 Extraterrestrial Translation Complete:', {
      from: fromProtocol?.species,
      targetDimension,
      translations: translations.length,
      accuracy: metrics.translationAccuracy,
      dimensionalBridging: metrics.dimensionalBridging
    });
    
    setIsTranslating(false);
  };

  const renderExtraterrestrialVisualization = () => {
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
    
    // Draw dimensional layers
    for (let i = 0; i < 7; i++) {
      const dimensionRadius = radius * (1 + i * 0.25);
      const alpha = 0.2 - i * 0.025;
      
      ctx.strokeStyle = `rgba(100, 255, 200, ${alpha})`;
      ctx.lineWidth = 1;
      ctx.setLineDash([8, 12]);
      ctx.beginPath();
      ctx.arc(centerX, centerY, dimensionRadius, 0, Math.PI * 2);
      ctx.stroke();
      ctx.setLineDash([]);
    }
    
    // Draw alien species nodes
    protocols.slice(0, 8).forEach((protocol, index) => {
      const angle = (index / 8) * Math.PI * 2 - Math.PI / 2;
      const distance = (protocol.dimensionalAccess / 7) * radius * 0.8 + radius * 0.2;
      const x = centerX + Math.cos(angle) * distance;
      const y = centerY + Math.sin(angle) * distance;
      
      // Draw species node
      const gradient = ctx.createRadialGradient(x, y, 0, x, y, 20);
      gradient.addColorStop(0, `rgba(100, 255, 200, ${protocol.consciousnessLevel})`);
      gradient.addColorStop(1, `rgba(100, 255, 200, 0)`);
      
      ctx.fillStyle = gradient;
      ctx.beginPath();
      ctx.arc(x, y, 12 + protocol.consciousnessLevel * 8, 0, Math.PI * 2);
      ctx.fill();
      
      // Draw dimensional connections
      protocols.slice(0, 8).forEach((otherProtocol, otherIndex) => {
        if (index < otherIndex) {
          const otherAngle = (otherIndex / 8) * Math.PI * 2 - Math.PI / 2;
          const otherDistance = (otherProtocol.dimensionalAccess / 7) * radius * 0.8 + radius * 0.2;
          const otherX = centerX + Math.cos(otherAngle) * otherDistance;
          const otherY = centerY + Math.sin(otherAngle) * otherDistance;
          
          const dimensionalCompatibility = 1 - Math.abs(protocol.dimensionalAccess - otherProtocol.dimensionalAccess) / 7;
          const consciousnessCompatibility = 1 - Math.abs(protocol.consciousnessLevel - otherProtocol.consciousnessLevel);
          
          ctx.strokeStyle = `rgba(100, 255, 200, ${dimensionalCompatibility * 0.3})`;
          ctx.lineWidth = dimensionalCompatibility * 3;
          ctx.setLineDash([5, 10]);
          ctx.beginPath();
          ctx.moveTo(x, y);
          ctx.lineTo(otherX, otherY);
          ctx.stroke();
          ctx.setLineDash([]);
        }
      });
    });
    
    // Draw central cosmic hub
    const hubGradient = ctx.createRadialGradient(centerX, centerY, 0, centerX, centerY, 35);
    hubGradient.addColorStop(0, `rgba(255, 255, 255, ${metrics.cosmicAwareness})`);
    hubGradient.addColorStop(1, `rgba(255, 255, 255, 0.2)`);
    
    ctx.fillStyle = hubGradient;
    ctx.beginPath();
    ctx.arc(centerX, centerY, 35, 0, Math.PI * 2);
    ctx.fill();
    
    // Draw cosmic symbol
    ctx.fillStyle = '#64ffc8';
    ctx.font = '24px Arial';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText('🛸', centerX, centerY);
  };

  useEffect(() => {
    renderExtraterrestrialVisualization();
  }, [protocols, metrics]);

  const getCommunicationMethodColor = (method: string) => {
    const colors: Record<string, string> = {
      'telepathic': 'bg-purple-600',
      'quantum': 'bg-blue-600',
      'frequency': 'bg-green-600',
      'symbolic': 'bg-orange-600',
      'mathematical': 'bg-red-600'
    };
    return colors[method] || 'bg-gray-600';
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-2xl font-bold text-white flex items-center gap-2">
            <Rocket className="w-6 h-6 text-cyan-400" />
            Extraterrestrial Communication
          </h3>
          <p className="text-gray-400">Interstellar Language Translation</p>
        </div>
        <div className="flex gap-2">
          <Button
            onClick={performExtraterrestrialTranslation}
            disabled={isTranslating}
            className="bg-cyan-600 hover:bg-cyan-700"
          >
            {isTranslating ? (
              <>
                <Brain className="w-4 h-4 mr-2 animate-spin" />
                Translating Across Dimensions...
              </>
            ) : (
              <>
                <Sparkles className="w-4 h-4 mr-2" />
                Translate Extraterrestrial
              </>
            )}
          </Button>
        </div>
      </div>

      {/* Cosmic Metrics */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <Card className="bg-black/30 border-cyan-500/20">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-gray-400 text-sm">Species Contacted</p>
                <p className="text-2xl font-bold text-white">{metrics.totalSpecies}</p>
              </div>
              <Globe className="w-8 h-8 text-cyan-400" />
            </div>
          </CardContent>
        </Card>

        <Card className="bg-black/30 border-purple-500/20">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-gray-400 text-sm">Active Signals</p>
                <p className="text-2xl font-bold text-white">{metrics.activeSignals}</p>
              </div>
              <Radio className="w-8 h-8 text-purple-400" />
            </div>
          </CardContent>
        </Card>

        <Card className="bg-black/30 border-green-500/20">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-gray-400 text-sm">Dimensional Bridging</p>
                <p className="text-2xl font-bold text-white">{(metrics.dimensionalBridging * 100).toFixed(1)}%</p>
              </div>
              <Star className="w-8 h-8 text-green-400" />
            </div>
          </CardContent>
        </Card>

        <Card className="bg-black/30 border-orange-500/20">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-gray-400 text-sm">Cosmic Awareness</p>
                <p className="text-2xl font-bold text-white">{(metrics.cosmicAwareness * 100).toFixed(1)}%</p>
              </div>
              <Satellite className="w-8 h-8 text-orange-400" />
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Main Content */}
      <div className="grid md:grid-cols-2 gap-6">
        {/* Cosmic Network Visualization */}
        <Card className="bg-black/30 border-gray-700">
          <CardHeader>
            <CardTitle className="text-white flex items-center gap-2">
              <Satellite className="w-5 h-5 text-cyan-400" />
              Interdimensional Network
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="relative aspect-video bg-gradient-to-br from-cyan-900/20 to-purple-900/20 rounded-lg overflow-hidden">
              <canvas
                ref={canvasRef}
                className="w-full h-full"
                style={{ imageRendering: 'crisp-edges' }}
              />
              
              {/* Cosmic Status */}
              <div className="absolute top-4 left-4 bg-black/70 backdrop-blur-sm rounded-lg px-3 py-2">
                <div className="text-cyan-400 text-sm font-medium">
                  Target Dimension: {targetDimension.toUpperCase()}
                </div>
                <div className="text-white text-xs">
                  Quantum Coherence: {(metrics.quantumCoherence * 100).toFixed(1)}%
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Alien Protocols */}
        <Card className="bg-black/30 border-gray-700">
          <CardHeader>
            <CardTitle className="text-white flex items-center gap-2">
              <Brain className="w-5 h-5 text-purple-400" />
              Extraterrestrial Protocols
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3 max-h-80 overflow-y-auto">
              {protocols.slice(0, 6).map(protocol => (
                <div key={protocol.id} className="p-3 bg-black/20 border border-gray-600 rounded-lg">
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center gap-2">
                      <h4 className="text-white font-medium">{protocol.species}</h4>
                      <Badge className={getCommunicationMethodColor(protocol.communicationMethod)}>
                        {protocol.communicationMethod}
                      </Badge>
                    </div>
                    <Badge className="bg-purple-600">
                      {protocol.dimensionalAccess}D
                    </Badge>
                  </div>
                  <div className="text-gray-300 text-sm mb-2">
                    {protocol.origin}
                  </div>
                  <div className="grid grid-cols-2 gap-2 text-xs">
                    <div className="flex justify-between">
                      <span className="text-gray-400">Signal Complexity:</span>
                      <span className="text-white">{(protocol.signalComplexity * 100).toFixed(1)}%</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Consciousness:</span>
                      <span className="text-white">{(protocol.consciousnessLevel * 100).toFixed(1)}%</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Translation Diff:</span>
                      <span className="text-white">{(protocol.translationDifficulty * 100).toFixed(1)}%</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Dimension Access:</span>
                      <span className="text-white">{protocol.dimensionalAccess}D</span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Translation Matrix */}
      <Card className="bg-black/30 border-gray-700">
        <CardHeader>
          <CardTitle className="text-white flex items-center gap-2">
            <Waves className="w-5 h-5 text-green-400" />
            Universal Translation Matrix
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {translationMatrix.slice(0, 6).map((translation, index) => (
              <div key={index} className="p-4 bg-black/20 border border-gray-600 rounded-lg">
                <div className="text-white font-medium mb-2">
                  {translation.fromSpecies} ↔ {translation.toSpecies}
                </div>
                <div className="text-gray-300 text-sm mb-2">
                  Universal Concept: {translation.universalConcept}
                </div>
                <div className="grid grid-cols-2 gap-2 text-xs">
                  <div className="flex justify-between">
                    <span className="text-gray-400">Translation Accuracy:</span>
                    <span className="text-white">{(translation.translationAccuracy * 100).toFixed(1)}%</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Dimensional Bridging:</span>
                    <span className="text-white">{(translation.dimensionalBridging * 100).toFixed(1)}%</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Consciousness Compatibility:</span>
                    <span className="text-white">{(translation.consciousnessCompatibility * 100).toFixed(1)}%</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Quantum Entanglement:</span>
                    <span className="text-white">{(translation.quantumEntanglement * 100).toFixed(1)}%</span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Status Badge */}
      <div className="flex justify-center">
        <Badge className={isTranslating ? "bg-cyan-600 animate-pulse" : "bg-gray-600"}>
          {isTranslating ? "Extraterrestrial Translation Active" : "Cosmic Communication Idle"}
        </Badge>
      </div>
    </div>
  );
}