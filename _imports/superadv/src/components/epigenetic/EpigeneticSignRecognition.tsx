'use client';

import { useState, useEffect, useRef } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { 
  Dna, 
  Brain, 
  Zap, 
  Activity, 
  TrendingUp, 
  Microscope,
  Heart,
  TreePine,
  Sparkles,
  Waves
} from 'lucide-react';

interface EpigeneticMarker {
  id: string;
  name: string;
  type: 'methylation' | 'acetylation' | 'phosphorylation' | 'ubiquitination';
  influence: number;
  heritability: number;
  environmentalFactor: string;
  signModification: string;
  expressionLevel: number;
}

interface GeneticMemory {
  id: string;
  ancestralSign: string;
  epigeneticState: number;
  generationalDepth: number;
  culturalContext: string;
  emotionalImprint: number;
  activationThreshold: number;
  patternStability: number;
}

interface EpigeneticSignal {
  id: string;
  timestamp: number;
  markerType: string;
  signalStrength: number;
  environmentalTrigger: string;
  geneExpression: number;
  signModification: string;
  hereditaryInfluence: number;
}

interface EpigeneticMetrics {
  totalMarkers: number;
  averageInfluence: number;
  heritabilityRate: number;
  environmentalAdaptation: number;
  signModificationRate: number;
  ancestralRecall: number;
  epigeneticDiversity: number;
}

export default function EpigeneticSignRecognition() {
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [selectedMarker, setSelectedMarker] = useState('methylation');
  const [markers, setMarkers] = useState<EpigeneticMarker[]>([]);
  const [memories, setMemories] = useState<GeneticMemory[]>([]);
  const [signals, setSignals] = useState<EpigeneticSignal[]>([]);
  const [metrics, setMetrics] = useState<EpigeneticMetrics>({
    totalMarkers: 0,
    averageInfluence: 0,
    heritabilityRate: 0,
    environmentalAdaptation: 0,
    signModificationRate: 0,
    ancestralRecall: 0,
    epigeneticDiversity: 0
  });

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationRef = useRef<number>();

  useEffect(() => {
    initializeEpigeneticSystem();
    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, []);

  const initializeEpigeneticSystem = async () => {
    console.log('🧬 Initializing Epigenetic Sign Recognition System...');
    
    // Initialize epigenetic markers
    const initialMarkers: EpigeneticMarker[] = [
      {
        id: 'methylation-marker-1',
        name: 'Sign Language Methylation',
        type: 'methylation',
        influence: 0.87,
        heritability: 0.72,
        environmentalFactor: 'cultural exposure',
        signModification: 'gesture amplification',
        expressionLevel: 0.78
      },
      {
        id: 'acetylation-marker-2',
        name: 'Neural Acetylation',
        type: 'acetylation',
        influence: 0.91,
        heritability: 0.68,
        environmentalFactor: 'social interaction',
        signModification: 'expression refinement',
        expressionLevel: 0.85
      },
      {
        id: 'phosphorylation-marker-3',
        name: 'Motor Phosphorylation',
        type: 'phosphorylation',
        influence: 0.84,
        heritability: 0.75,
        environmentalFactor: 'physical training',
        signModification: 'movement efficiency',
        expressionLevel: 0.82
      },
      {
        id: 'ubiquitination-marker-4',
        name: 'Cognitive Ubiquitination',
        type: 'ubiquitination',
        influence: 0.89,
        heritability: 0.65,
        environmentalFactor: 'learning environment',
        signModification: 'conceptual abstraction',
        expressionLevel: 0.91
      }
    ];

    // Initialize genetic memories
    const initialMemories: GeneticMemory[] = [
      {
        id: 'memory-1',
        ancestralSign: 'ancient greeting gesture',
        epigeneticState: 0.73,
        generationalDepth: 7,
        culturalContext: 'tribal communication',
        emotionalImprint: 0.68,
        activationThreshold: 0.45,
        patternStability: 0.82
      },
      {
        id: 'memory-2',
        ancestralSign: 'hunting coordination signs',
        epigeneticState: 0.81,
        generationalDepth: 12,
        culturalContext: 'survival communication',
        emotionalImprint: 0.74,
        activationThreshold: 0.38,
        patternStability: 0.79
      },
      {
        id: 'memory-3',
        ancestralSign: 'ceremonial sign language',
        epigeneticState: 0.67,
        generationalDepth: 5,
        culturalContext: 'ritual communication',
        emotionalImprint: 0.91,
        activationThreshold: 0.52,
        patternStability: 0.88
      },
      {
        id: 'memory-4',
        ancestralSign: 'emotional expression signs',
        epigeneticState: 0.85,
        generationalDepth: 9,
        culturalContext: 'family communication',
        emotionalImprint: 0.86,
        activationThreshold: 0.41,
        patternStability: 0.76
      }
    ];

    // Initialize epigenetic signals
    const initialSignals: EpigeneticSignal[] = Array.from({ length: 30 }, (_, i) => ({
      id: `signal-${i}`,
      timestamp: Date.now() - (30 - i) * 8000,
      markerType: initialMarkers[Math.floor(Math.random() * initialMarkers.length)].type,
      signalStrength: Math.random(),
      environmentalTrigger: ['stress', 'learning', 'social', 'environmental', 'cultural'][Math.floor(Math.random() * 5)],
      geneExpression: Math.random(),
      signModification: 'adaptive gesture modification',
      hereditaryInfluence: Math.random()
    }));

    setMarkers(initialMarkers);
    setMemories(initialMemories);
    setSignals(initialSignals);
    
    startEpigeneticProcessing();
    
    console.log('✅ Epigenetic System initialized with', initialMarkers.length, 'markers');
  };

  const startEpigeneticProcessing = () => {
    const process = () => {
      // Update epigenetic markers
      setMarkers(prevMarkers => 
        prevMarkers.map(marker => ({
          ...marker,
          influence: Math.min(0.99, marker.influence + (Math.random() - 0.4) * 0.01),
          expressionLevel: Math.min(0.99, marker.expressionLevel + (Math.random() - 0.3) * 0.008),
          heritability: Math.min(0.99, marker.heritability + (Math.random() - 0.5) * 0.005)
        }))
      );

      // Update genetic memories
      setMemories(prevMemories => 
        prevMemories.map(memory => ({
          ...memory,
          epigeneticState: Math.max(0.1, Math.min(0.99, memory.epigeneticState + (Math.random() - 0.5) * 0.02)),
          activationThreshold: Math.max(0.1, Math.min(0.99, memory.activationThreshold + (Math.random() - 0.5) * 0.01)),
          patternStability: Math.max(0.2, Math.min(0.99, memory.patternStability + (Math.random() - 0.4) * 0.008))
        }))
      );

      // Update epigenetic signals
      setSignals(prevSignals => {
        const newSignal: EpigeneticSignal = {
          id: `signal-${Date.now()}`,
          timestamp: Date.now(),
          markerType: markers[Math.floor(Math.random() * markers.length)]?.type || 'methylation',
          signalStrength: Math.random(),
          environmentalTrigger: ['stress', 'learning', 'social', 'environmental', 'cultural'][Math.floor(Math.random() * 5)],
          geneExpression: Math.random(),
          signModification: 'adaptive gesture modification',
          hereditaryInfluence: Math.random()
        };
        
        return [...prevSignals.slice(-50), newSignal];
      });

      // Update metrics
      setMetrics(prev => ({
        ...prev,
        totalMarkers: prev.totalMarkers + (Math.random() > 0.95 ? 1 : 0),
        averageInfluence: Math.min(0.99, prev.averageInfluence + 0.0005),
        heritabilityRate: Math.min(0.99, prev.heritabilityRate + 0.0003),
        environmentalAdaptation: Math.min(0.99, prev.environmentalAdaptation + 0.0008),
        signModificationRate: Math.min(0.99, prev.signModificationRate + 0.0006),
        ancestralRecall: Math.min(0.99, prev.ancestralRecall + 0.0004),
        epigeneticDiversity: Math.min(0.99, prev.epigeneticDiversity + 0.0007)
      }));

      animationRef.current = requestAnimationFrame(process);
    };
    process();
  };

  const performEpigeneticAnalysis = async () => {
    setIsAnalyzing(true);
    
    // Simulate epigenetic analysis
    await new Promise(resolve => setTimeout(resolve, 3000));
    
    const selectedMarkerData = markers.find(m => m.type === selectedMarker);
    
    console.log('🧬 Epigenetic Analysis Complete:', {
      marker: selectedMarker,
      memories: memories.length,
      signals: signals.length,
      influence: selectedMarkerData?.influence,
      heritability: selectedMarkerData?.heritability
    });
    
    setIsAnalyzing(false);
  };

  const renderEpigeneticVisualization = () => {
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
    
    // Draw epigenetic layers
    for (let i = 0; i < 5; i++) {
      const layerRadius = radius * (1 + i * 0.2);
      const alpha = 0.15 - i * 0.025;
      
      ctx.strokeStyle = `rgba(147, 51, 234, ${alpha})`;
      ctx.lineWidth = 2;
      ctx.setLineDash([6, 10]);
      ctx.beginPath();
      ctx.arc(centerX, centerY, layerRadius, 0, Math.PI * 2);
      ctx.stroke();
      ctx.setLineDash([]);
    }
    
    // Draw epigenetic markers
    markers.forEach((marker, index) => {
      const angle = (index / markers.length) * Math.PI * 2 - Math.PI / 2;
      const distance = (marker.influence * radius * 0.7) + radius * 0.3;
      const x = centerX + Math.cos(angle) * distance;
      const y = centerY + Math.sin(angle) * distance;
      
      // Draw marker
      const colors = {
        'methylation': 'rgba(255, 100, 100, ',
        'acetylation': 'rgba(100, 255, 100, ',
        'phosphorylation': 'rgba(100, 100, 255, ',
        'ubiquitination': 'rgba(255, 255, 100, '
      };
      
      const baseColor = colors[marker.type] || 'rgba(200, 200, 200, ';
      
      // Draw marker node
      const gradient = ctx.createRadialGradient(x, y, 0, x, y, 18);
      gradient.addColorStop(0, baseColor + marker.expressionLevel + ')');
      gradient.addColorStop(1, baseColor + '0)');
      
      ctx.fillStyle = gradient;
      ctx.beginPath();
      ctx.arc(x, y, 10 + marker.influence * 8, 0, Math.PI * 2);
      ctx.fill();
      
      // Draw expression level
      if (marker.expressionLevel > 0.7) {
        ctx.strokeStyle = baseColor + marker.expressionLevel + ')';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(x, y, 12 + marker.influence * 10, 0, Math.PI * 2);
        ctx.stroke();
      }
      
      // Draw hereditary connections
      markers.forEach((otherMarker, otherIndex) => {
        if (index < otherIndex) {
          const otherAngle = (otherIndex / markers.length) * Math.PI * 2 - Math.PI / 2;
          const otherDistance = (otherMarker.influence * radius * 0.7) + radius * 0.3;
          const otherX = centerX + Math.cos(otherAngle) * otherDistance;
          const otherY = centerY + Math.sin(otherAngle) * otherDistance;
          
          const hereditaryLink = Math.min(marker.heritability, otherMarker.heritability);
          
          ctx.strokeStyle = baseColor + hereditaryLink + ')';
          ctx.lineWidth = hereditaryLink * 3;
          ctx.beginPath();
          ctx.moveTo(x, y);
          ctx.lineTo(otherX, otherY);
          ctx.stroke();
        }
      });
    });
    
    // Draw central epigenetic core
    const coreGradient = ctx.createRadialGradient(centerX, centerY, 0, centerX, centerY, 30);
    coreGradient.addColorStop(0, `rgba(255, 255, 255, ${metrics.environmentalAdaptation})`);
    coreGradient.addColorStop(1, `rgba(255, 255, 255, 0.2)`);
    
    ctx.fillStyle = coreGradient;
    ctx.beginPath();
    ctx.arc(centerX, centerY, 30, 0, Math.PI * 2);
    ctx.fill();
    
    // Draw DNA symbol
    ctx.fillStyle = '#9333ea';
    ctx.font = '24px Arial';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText('🧬', centerX, centerY);
  };

  useEffect(() => {
    renderEpigeneticVisualization();
  }, [markers, metrics]);

  const getMarkerTypeColor = (type: string) => {
    const colors: Record<string, string> = {
      'methylation': 'bg-red-600',
      'acetylation': 'bg-green-600',
      'phosphorylation': 'bg-blue-600',
      'ubiquitination': 'bg-yellow-600'
    };
    return colors[type] || 'bg-gray-600';
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-2xl font-bold text-white flex items-center gap-2">
            <Dna className="w-6 h-6 text-pink-400" />
            Epigenetic Sign Recognition
          </h3>
          <p className="text-gray-400">Hereditary Sign Pattern Analysis</p>
        </div>
        <div className="flex gap-2">
          <Button
            onClick={performEpigeneticAnalysis}
            disabled={isAnalyzing}
            className="bg-pink-600 hover:bg-pink-700"
          >
            {isAnalyzing ? (
              <>
                <Brain className="w-4 h-4 mr-2 animate-spin" />
                Analyzing Epigenetics...
              </>
            ) : (
              <>
                <Sparkles className="w-4 h-4 mr-2" />
                Analyze Epigenetic Patterns
              </>
            )}
          </Button>
        </div>
      </div>

      {/* Epigenetic Metrics */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <Card className="bg-black/30 border-pink-500/20">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-gray-400 text-sm">Total Markers</p>
                <p className="text-2xl font-bold text-white">{metrics.totalMarkers}</p>
              </div>
              <Microscope className="w-8 h-8 text-pink-400" />
            </div>
          </CardContent>
        </Card>

        <Card className="bg-black/30 border-green-500/20">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-gray-400 text-sm">Heritability Rate</p>
                <p className="text-2xl font-bold text-white">{(metrics.heritabilityRate * 100).toFixed(1)}%</p>
              </div>
              <TrendingUp className="w-8 h-8 text-green-400" />
            </div>
          </CardContent>
        </Card>

        <Card className="bg-black/30 border-blue-500/20">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-gray-400 text-sm">Environmental Adaptation</p>
                <p className="text-2xl font-bold text-white">{(metrics.environmentalAdaptation * 100).toFixed(1)}%</p>
              </div>
              <TreePine className="w-8 h-8 text-blue-400" />
            </div>
          </CardContent>
        </Card>

        <Card className="bg-black/30 border-purple-500/20">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-gray-400 text-sm">Ancestral Recall</p>
                <p className="text-2xl font-bold text-white">{(metrics.ancestralRecall * 100).toFixed(1)}%</p>
              </div>
              <Heart className="w-8 h-8 text-purple-400" />
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Main Content */}
      <div className="grid md:grid-cols-2 gap-6">
        {/* Epigenetic Field Visualization */}
        <Card className="bg-black/30 border-gray-700">
          <CardHeader>
            <CardTitle className="text-white flex items-center gap-2">
              <Waves className="w-5 h-5 text-pink-400" />
              Epigenetic Field
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="relative aspect-video bg-gradient-to-br from-pink-900/20 to-purple-900/20 rounded-lg overflow-hidden">
              <canvas
                ref={canvasRef}
                className="w-full h-full"
                style={{ imageRendering: 'crisp-edges' }}
              />
              
              {/* Epigenetic Status */}
              <div className="absolute top-4 left-4 bg-black/70 backdrop-blur-sm rounded-lg px-3 py-2">
                <div className="text-pink-400 text-sm font-medium">
                  Selected Marker: {selectedMarker}
                </div>
                <div className="text-white text-xs">
                  Sign Modification Rate: {(metrics.signModificationRate * 100).toFixed(2)}%
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Epigenetic Markers */}
        <Card className="bg-black/30 border-gray-700">
          <CardHeader>
            <CardTitle className="text-white flex items-center gap-2">
              <Dna className="w-5 h-5 text-green-400" />
              Epigenetic Markers
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3 max-h-80 overflow-y-auto">
              {markers.map(marker => (
                <div key={marker.id} className="p-3 bg-black/20 border border-gray-600 rounded-lg">
                  <div className="flex items-center justify-between mb-2">
                    <h4 className="text-white font-medium">{marker.name}</h4>
                    <Badge className={getMarkerTypeColor(marker.type)}>
                      {marker.type}
                    </Badge>
                  </div>
                  <div className="text-gray-300 text-sm mb-2">
                    Environmental Factor: {marker.environmentalFactor}
                  </div>
                  <div className="grid grid-cols-2 gap-2 text-xs">
                    <div className="flex justify-between">
                      <span className="text-gray-400">Influence:</span>
                      <span className="text-white">{(marker.influence * 100).toFixed(1)}%</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Heritability:</span>
                      <span className="text-white">{(marker.heritability * 100).toFixed(1)}%</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Expression:</span>
                      <span className="text-white">{(marker.expressionLevel * 100).toFixed(1)}%</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Sign Modification:</span>
                      <span className="text-white">{marker.signModification}</span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Genetic Memories */}
      <Card className="bg-black/30 border-gray-700">
        <CardHeader>
          <CardTitle className="text-white flex items-center gap-2">
            <Brain className="w-5 h-5 text-purple-400" />
            Ancestral Genetic Memories
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {memories.map(memory => (
              <div key={memory.id} className="p-4 bg-black/20 border border-gray-600 rounded-lg">
                <div className="flex items-center justify-between mb-2">
                  <h4 className="text-white font-medium">{memory.ancestralSign}</h4>
                  <Badge className="bg-purple-600">
                    {memory.generationalDepth} Gen
                  </Badge>
                </div>
                <div className="text-gray-300 text-sm mb-2">
                  {memory.culturalContext}
                </div>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-gray-400">Epigenetic State:</span>
                    <span className="text-white">{(memory.epigeneticState * 100).toFixed(1)}%</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Emotional Imprint:</span>
                    <span className="text-white">{(memory.emotionalImprint * 100).toFixed(1)}%</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Activation Threshold:</span>
                    <span className="text-white">{(memory.activationThreshold * 100).toFixed(1)}%</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Pattern Stability:</span>
                    <span className="text-white">{(memory.patternStability * 100).toFixed(1)}%</span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Epigenetic Signals */}
      <Card className="bg-black/30 border-gray-700">
        <CardHeader>
          <CardTitle className="text-white flex items-center gap-2">
            <Activity className="w-5 h-5 text-orange-400" />
            Epigenetic Signals
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-2 max-h-60 overflow-y-auto">
            {signals.slice(-12).map(signal => (
              <div key={signal.id} className="p-3 bg-black/20 border border-gray-600 rounded-lg">
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center gap-2">
                    <Badge className={getMarkerTypeColor(signal.markerType)}>
                      {signal.markerType}
                    </Badge>
                    <Badge className={signal.signalStrength > 0.7 ? 'bg-green-600' : 'bg-yellow-600'}>
                      {(signal.signalStrength * 100).toFixed(1)}%
                    </Badge>
                  </div>
                  <span className="text-gray-400 text-xs">
                    {new Date(signal.timestamp).toLocaleTimeString()}
                  </span>
                </div>
                <div className="text-gray-300 text-sm mb-2">
                  Trigger: {signal.environmentalTrigger}
                </div>
                <div className="grid grid-cols-2 gap-2 text-xs">
                  <div className="flex justify-between">
                    <span className="text-gray-400">Gene Expression:</span>
                    <span className="text-white">{(signal.geneExpression * 100).toFixed(1)}%</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Hereditary Influence:</span>
                    <span className="text-white">{(signal.hereditaryInfluence * 100).toFixed(1)}%</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Sign Modification:</span>
                    <span className="text-white">{signal.signModification}</span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Status Badge */}
      <div className="flex justify-center">
        <Badge className={isAnalyzing ? "bg-pink-600 animate-pulse" : "bg-gray-600"}>
          {isAnalyzing ? "Epigenetic Analysis Active" : "Epigenetic System Idle"}
        </Badge>
      </div>
    </div>
  );
}