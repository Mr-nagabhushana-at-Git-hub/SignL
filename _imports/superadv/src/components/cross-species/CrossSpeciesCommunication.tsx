'use client';

import { useState, useEffect, useRef } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { 
  Heart, 
  Brain, 
  Zap, 
  Activity, 
  TrendingUp, 
  PawPrint, 
  Bird, 
  Fish, 
  Bug, 
  TreePine,
  Radio,
  Waves
} from 'lucide-react';

interface SpeciesProtocol {
  id: string;
  species: string;
  communicationType: 'vocal' | 'visual' | 'chemical' | 'electrical' | 'bioluminescent';
  signalComplexity: number;
  bandwidth: number;
  vocabulary: number;
  syntax: boolean;
  semantics: boolean;
  neuralCompatibility: number;
}

interface TranslationMatrix {
  fromSpecies: string;
  toSpecies: string;
  accuracy: number;
  latency: number;
  semanticLoss: number;
  culturalContext: number;
}

interface CrossSpeciesSignal {
  id: string;
  species: string;
  signalType: string;
  pattern: number[];
  meaning: string;
  emotionalContent: number;
  complexity: number;
  timestamp: number;
}

interface UniversalTranslator {
  id: string;
  name: string;
  supportedSpecies: string[];
  translationAccuracy: number;
  processingSpeed: number;
  neuralInterface: boolean;
  quantumEntanglement: boolean;
  bioSignalProcessing: boolean;
}

export default function CrossSpeciesCommunication() {
  const [isTranslating, setIsTranslating] = useState(false);
  const [selectedSpecies, setSelectedSpecies] = useState('human');
  const [targetSpecies, setTargetSpecies] = useState('dolphin');
  const [protocols, setProtocols] = useState<SpeciesProtocol[]>([]);
  const [signals, setSignals] = useState<CrossSpeciesSignal[]>([]);
  const [translationMatrix, setTranslationMatrix] = useState<TranslationMatrix[]>([]);
  const [translator, setTranslator] = useState<UniversalTranslator | null>(null);
  const [activeSignals, setActiveSignals] = useState(0);
  const [translationAccuracy, setTranslationAccuracy] = useState(0);

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationRef = useRef<number>();

  useEffect(() => {
    initializeCrossSpeciesCommunication();
    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, []);

  const initializeCrossSpeciesCommunication = async () => {
    console.log('🐾 Initializing Cross-Species Communication System...');
    
    // Initialize species protocols
    const initialProtocols: SpeciesProtocol[] = [
      {
        id: 'human',
        species: 'Human',
        communicationType: 'vocal',
        signalComplexity: 0.95,
        bandwidth: 8500, // Hz
        vocabulary: 170000,
        syntax: true,
        semantics: true,
        neuralCompatibility: 0.98
      },
      {
        id: 'dolphin',
        species: 'Bottlenose Dolphin',
        communicationType: 'vocal',
        signalComplexity: 0.88,
        bandwidth: 150000, // Hz
        vocabulary: 32000,
        syntax: true,
        semantics: true,
        neuralCompatibility: 0.82
      },
      {
        id: 'whale',
        species: 'Humpback Whale',
        communicationType: 'vocal',
        signalComplexity: 0.92,
        bandwidth: 8000, // Hz
        vocabulary: 45000,
        syntax: true,
        semantics: true,
        neuralCompatibility: 0.79
      },
      {
        id: 'elephant',
        species: 'African Elephant',
        communicationType: 'vocal',
        signalComplexity: 0.76,
        bandwidth: 12000, // Hz
        vocabulary: 12000,
        syntax: false,
        semantics: true,
        neuralCompatibility: 0.71
      },
      {
        id: 'bee',
        species: 'Honey Bee',
        communicationType: 'visual',
        signalComplexity: 0.84,
        bandwidth: 250, // Hz (waggle dance)
        vocabulary: 500,
        syntax: true,
        semantics: true,
        neuralCompatibility: 0.45
      },
      {
        id: 'octopus',
        species: 'Common Octopus',
        communicationType: 'visual',
        signalComplexity: 0.91,
        bandwidth: 30000, // Hz (color/pattern changes)
        vocabulary: 15000,
        syntax: true,
        semantics: true,
        neuralCompatibility: 0.68
      },
      {
        id: 'firefly',
        species: 'Firefly',
        communicationType: 'bioluminescent',
        signalComplexity: 0.67,
        bandwidth: 100, // Hz (light patterns)
        vocabulary: 200,
        syntax: true,
        semantics: false,
        neuralCompatibility: 0.32
      },
      {
        id: 'tree',
        species: 'Oak Network',
        communicationType: 'chemical',
        signalComplexity: 0.58,
        bandwidth: 0.001, // Hz (chemical signals)
        vocabulary: 50,
        syntax: false,
        semantics: true,
        neuralCompatibility: 0.25
      }
    ];

    // Initialize translation matrix
    const initialMatrix: TranslationMatrix[] = [];
    initialProtocols.forEach(from => {
      initialProtocols.forEach(to => {
        if (from.id !== to.id) {
          const baseAccuracy = Math.min(from.neuralCompatibility, to.neuralCompatibility);
          const complexityPenalty = Math.abs(from.signalComplexity - to.signalComplexity) * 0.5;
          const accuracy = Math.max(0.1, baseAccuracy - complexityPenalty);
          
          initialMatrix.push({
            fromSpecies: from.species,
            toSpecies: to.species,
            accuracy,
            latency: 100 + Math.random() * 500, // ms
            semanticLoss: (1 - accuracy) * 0.3,
            culturalContext: Math.random() * 0.5
          });
        }
      });
    });

    // Initialize universal translator
    const universalTranslator: UniversalTranslator = {
      id: 'quantum-bio-translator-v1',
      name: 'Quantum-Biological Universal Translator',
      supportedSpecies: initialProtocols.map(p => p.species),
      translationAccuracy: 0.89,
      processingSpeed: 0.95,
      neuralInterface: true,
      quantumEntanglement: true,
      bioSignalProcessing: true
    };

    setProtocols(initialProtocols);
    setTranslationMatrix(initialMatrix);
    setTranslator(universalTranslator);
    
    // Start signal processing
    startSignalProcessing();
    
    console.log('✅ Cross-Species Communication initialized with', initialProtocols.length, 'species');
  };

  const startSignalProcessing = () => {
    const process = () => {
      // Generate random cross-species signals
      const newSignals: CrossSpeciesSignal[] = protocols.map(protocol => ({
        id: `signal-${protocol.id}-${Date.now()}`,
        species: protocol.species,
        signalType: protocol.communicationType,
        pattern: Array.from({ length: 10 }, () => Math.random()),
        meaning: generateSpeciesMeaning(protocol.species),
        emotionalContent: Math.random(),
        complexity: protocol.signalComplexity,
        timestamp: Date.now()
      }));

      setSignals(prevSignals => [...prevSignals.slice(-50), ...newSignals]);
      setActiveSignals(newSignals.filter(s => s.emotionalContent > 0.7).length);
      
      // Update translation accuracy
      setTranslationAccuracy(prev => Math.min(0.99, prev + (Math.random() - 0.3) * 0.01));

      animationRef.current = requestAnimationFrame(process);
    };
    process();
  };

  const generateSpeciesMeaning = (species: string): string => {
    const meanings: Record<string, string[]> = {
      'Human': ['hello', 'danger', 'food', 'help', 'love', 'goodbye'],
      'Bottlenose Dolphin': ['prey_found', 'danger_shark', 'mating_call', 'group_coordination', 'play'],
      'Humpback Whale': ['migration_route', 'feeding_grounds', 'mating_song', 'warning', 'greeting'],
      'African Elephant': ['water_source', 'danger_predator', 'family_separation', 'food_available', 'death'],
      'Honey Bee': ['nectar_source', 'hive_threat', 'new_flower_field', 'queen_location', 'swarm'],
      'Common Octopus': ['prey_detected', 'mating_display', 'danger_escape', 'camouflage', 'hunting'],
      'Firefly': ['mating_signal', 'territory_mark', 'species_identification', 'swarm_coordination'],
      'Oak Network': ['drought_stress', 'insect_attack', 'nutrient_deficiency', 'fungal_spread', 'season_change']
    };
    
    const speciesMeanings = meanings[species] || ['unknown'];
    return speciesMeanings[Math.floor(Math.random() * speciesMeanings.length)];
  };

  const performCrossSpeciesTranslation = async () => {
    setIsTranslating(true);
    
    // Simulate cross-species translation
    await new Promise(resolve => setTimeout(resolve, 2500));
    
    const fromProtocol = protocols.find(p => p.id === selectedSpecies);
    const toProtocol = protocols.find(p => p.id === targetSpecies);
    const translation = translationMatrix.find(t => 
      t.fromSpecies === fromProtocol?.species && t.toSpecies === toProtocol?.species
    );
    
    console.log('🐾 Cross-Species Translation Complete:', {
      from: fromProtocol?.species,
      to: toProtocol?.species,
      accuracy: translation?.accuracy,
      latency: translation?.latency,
      semanticLoss: translation?.semanticLoss
    });
    
    setIsTranslating(false);
  };

  const getSpeciesIcon = (species: string) => {
    const icons: Record<string, React.ReactNode> = {
      'Human': <Brain className="w-4 h-4" />,
      'Bottlenose Dolphin': <Fish className="w-4 h-4" />,
      'Humpback Whale': <Fish className="w-4 h-4" />,
      'African Elephant': <PawPrint className="w-4 h-4" />,
      'Honey Bee': <Bug className="w-4 h-4" />,
      'Common Octopus': <Zap className="w-4 h-4" />,
      'Firefly': <Radio className="w-4 h-4" />,
      'Oak Network': <TreePine className="w-4 h-4" />
    };
    return icons[species] || <Brain className="w-4 h-4" />;
  };

  const getCommunicationTypeColor = (type: string) => {
    const colors: Record<string, string> = {
      'vocal': 'bg-blue-600',
      'visual': 'bg-green-600',
      'chemical': 'bg-purple-600',
      'electrical': 'bg-yellow-600',
      'bioluminescent': 'bg-orange-600'
    };
    return colors[type] || 'bg-gray-600';
  };

  const renderCrossSpeciesVisualization = () => {
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
    
    // Draw species network
    protocols.slice(0, 6).forEach((protocol, i) => {
      const angle = (i / 6) * Math.PI * 2 - Math.PI / 2;
      const x = centerX + Math.cos(angle) * radius;
      const y = centerY + Math.sin(angle) * radius;
      
      // Draw species node
      const gradient = ctx.createRadialGradient(x, y, 0, x, y, 25);
      gradient.addColorStop(0, `rgba(100, 200, 255, ${protocol.neuralCompatibility})`);
      gradient.addColorStop(1, `rgba(100, 200, 255, 0)`);
      
      ctx.fillStyle = gradient;
      ctx.beginPath();
      ctx.arc(x, y, 15 + protocol.neuralCompatibility * 10, 0, Math.PI * 2);
      ctx.fill();
      
      // Draw communication connections
      protocols.slice(0, 6).forEach((otherProtocol, j) => {
        if (i < j) {
          const otherAngle = (j / 6) * Math.PI * 2 - Math.PI / 2;
          const otherX = centerX + Math.cos(otherAngle) * radius;
          const otherY = centerY + Math.sin(otherAngle) * radius;
          
          const compatibility = Math.min(protocol.neuralCompatibility, otherProtocol.neuralCompatibility);
          
          ctx.strokeStyle = `rgba(100, 200, 255, ${compatibility * 0.3})`;
          ctx.lineWidth = compatibility * 2;
          ctx.setLineDash([5, 5]);
          ctx.beginPath();
          ctx.moveTo(x, y);
          ctx.lineTo(otherX, otherY);
          ctx.stroke();
          ctx.setLineDash([]);
        }
      });
    });
    
    // Draw central translator
    const translatorGradient = ctx.createRadialGradient(centerX, centerY, 0, centerX, centerY, 30);
    translatorGradient.addColorStop(0, `rgba(255, 255, 100, ${translator?.translationAccuracy || 0.8})`);
    translatorGradient.addColorStop(1, `rgba(255, 255, 100, 0.2)`);
    
    ctx.fillStyle = translatorGradient;
    ctx.beginPath();
    ctx.arc(centerX, centerY, 30, 0, Math.PI * 2);
    ctx.fill();
    
    // Draw translator icon
    ctx.fillStyle = '#ffff64';
    ctx.font = '20px Arial';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText('🌐', centerX, centerY);
  };

  useEffect(() => {
    renderCrossSpeciesVisualization();
  }, [protocols, translator]);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-2xl font-bold text-white flex items-center gap-2">
            <PawPrint className="w-6 h-6 text-orange-400" />
            Cross-Species Communication
          </h3>
          <p className="text-gray-400">Inter-Species Language Translation</p>
        </div>
        <div className="flex gap-2">
          <Button
            onClick={performCrossSpeciesTranslation}
            disabled={isTranslating}
            className="bg-orange-600 hover:bg-orange-700"
          >
            {isTranslating ? (
              <>
                <Brain className="w-4 h-4 mr-2 animate-spin" />
                Translating...
              </>
            ) : (
              <>
                <Zap className="w-4 h-4 mr-2" />
                Translate Cross-Species
              </>
            )}
          </Button>
        </div>
      </div>

      {/* Communication Metrics */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <Card className="bg-black/30 border-orange-500/20">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-gray-400 text-sm">Species</p>
                <p className="text-2xl font-bold text-white">{protocols.length}</p>
              </div>
              <PawPrint className="w-8 h-8 text-orange-400" />
            </div>
          </CardContent>
        </Card>

        <Card className="bg-black/30 border-blue-500/20">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-gray-400 text-sm">Active Signals</p>
                <p className="text-2xl font-bold text-white">{activeSignals}</p>
              </div>
              <Radio className="w-8 h-8 text-blue-400" />
            </div>
          </CardContent>
        </Card>

        <Card className="bg-black/30 border-green-500/20">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-gray-400 text-sm">Translation Accuracy</p>
                <p className="text-2xl font-bold text-white">{(translationAccuracy * 100).toFixed(1)}%</p>
              </div>
              <TrendingUp className="w-8 h-8 text-green-400" />
            </div>
          </CardContent>
        </Card>

        <Card className="bg-black/30 border-purple-500/20">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-gray-400 text-sm">Neural Interface</p>
                <p className="text-2xl font-bold text-white">
                  {translator?.neuralInterface ? 'Active' : 'Inactive'}
                </p>
              </div>
              <Brain className="w-8 h-8 text-purple-400" />
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Main Content */}
      <div className="grid md:grid-cols-2 gap-6">
        {/* Species Network Visualization */}
        <Card className="bg-black/30 border-gray-700">
          <CardHeader>
            <CardTitle className="text-white flex items-center gap-2">
              <Heart className="w-5 h-5 text-orange-400" />
              Species Communication Network
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="relative aspect-video bg-gradient-to-br from-orange-900/20 to-blue-900/20 rounded-lg overflow-hidden">
              <canvas
                ref={canvasRef}
                className="w-full h-full"
                style={{ imageRendering: 'crisp-edges' }}
              />
              
              {/* Translator Status */}
              <div className="absolute top-4 left-4 bg-black/70 backdrop-blur-sm rounded-lg px-3 py-2">
                <div className="text-orange-400 text-sm font-medium">
                  Universal Translator: {translator?.name}
                </div>
                <div className="text-white text-xs">
                  Accuracy: {(translator?.translationAccuracy * 100 || 0).toFixed(1)}%
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Species Protocols */}
        <Card className="bg-black/30 border-gray-700">
          <CardHeader>
            <CardTitle className="text-white flex items-center gap-2">
              <Brain className="w-5 h-5 text-blue-400" />
              Species Communication Protocols
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3 max-h-80 overflow-y-auto">
              {protocols.slice(0, 6).map(protocol => (
                <div key={protocol.id} className="p-3 bg-black/20 border border-gray-600 rounded-lg">
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center gap-2">
                      {getSpeciesIcon(protocol.species)}
                      <h4 className="text-white font-medium">{protocol.species}</h4>
                    </div>
                    <Badge className={getCommunicationTypeColor(protocol.communicationType)}>
                      {protocol.communicationType}
                    </Badge>
                  </div>
                  <div className="grid grid-cols-2 gap-2 text-xs">
                    <div className="flex justify-between">
                      <span className="text-gray-400">Complexity:</span>
                      <span className="text-white">{(protocol.signalComplexity * 100).toFixed(1)}%</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Vocabulary:</span>
                      <span className="text-white">{protocol.vocabulary.toLocaleString()}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Bandwidth:</span>
                      <span className="text-white">{protocol.bandwidth.toLocaleString()} Hz</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Neural Comp:</span>
                      <span className="text-white">{(protocol.neuralCompatibility * 100).toFixed(1)}%</span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Universal Translator */}
      <Card className="bg-black/30 border-gray-700">
        <CardHeader>
          <CardTitle className="text-white flex items-center gap-2">
            <Zap className="w-5 h-5 text-purple-400" />
            Universal Quantum Translator
          </CardTitle>
        </CardHeader>
        <CardContent>
          {translator && (
            <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
              <div className="bg-black/20 rounded p-3">
                <div className="text-gray-400 text-sm mb-1">Supported Species</div>
                <div className="text-purple-400 font-medium">
                  {translator.supportedSpecies.length}
                </div>
              </div>
              <div className="bg-black/20 rounded p-3">
                <div className="text-gray-400 text-sm mb-1">Processing Speed</div>
                <div className="text-purple-400 font-medium">
                  {(translator.processingSpeed * 100).toFixed(1)}%
                </div>
              </div>
              <div className="bg-black/20 rounded p-3">
                <div className="text-gray-400 text-sm mb-1">Neural Interface</div>
                <div className="text-purple-400 font-medium">
                  {translator.neuralInterface ? 'Quantum' : 'Standard'}
                </div>
              </div>
              <div className="bg-black/20 rounded p-3">
                <div className="text-gray-400 text-sm mb-1">Quantum Entanglement</div>
                <div className="text-purple-400 font-medium">
                  {translator.quantumEntanglement ? 'Active' : 'Inactive'}
                </div>
              </div>
              <div className="bg-black/20 rounded p-3">
                <div className="text-gray-400 text-sm mb-1">Bio-Signal Processing</div>
                <div className="text-purple-400 font-medium">
                  {translator.bioSignalProcessing ? 'Active' : 'Inactive'}
                </div>
              </div>
              <div className="bg-black/20 rounded p-3">
                <div className="text-gray-400 text-sm mb-1">Translation Accuracy</div>
                <div className="text-purple-400 font-medium">
                  {(translator.translationAccuracy * 100).toFixed(1)}%
                </div>
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Status Badge */}
      <div className="flex justify-center">
        <Badge className={isTranslating ? "bg-orange-600 animate-pulse" : "bg-gray-600"}>
          {isTranslating ? "Cross-Species Translation Active" : "Translation Idle"}
        </Badge>
      </div>
    </div>
  );
}