'use client';

import { useState, useEffect, useRef } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { 
  Moon, 
  Brain, 
  Zap, 
  Activity, 
  TrendingUp, 
  Cloud,
  Sparkles,
  Heart,
  Eye,
  Wind
} from 'lucide-react';

interface DreamPattern {
  id: string;
  dreamType: 'lucid' | 'rem' | 'deep' | 'nightmare' | 'prophetic';
  intensity: number;
  emotionalContent: number;
  symbolicMeaning: string;
  neuralActivity: number[];
  duration: number;
  clarity: number;
}

interface LearningModule {
  id: string;
  name: string;
  dreamPhase: 'pre-sleep' | 'light-sleep' | 'deep-sleep' | 'rem-sleep';
  learningRate: number;
  memoryConsolidation: number;
  patternRecognition: number;
  neuralPlasticity: number;
  dreamIntegration: number;
}

interface DreamSign {
  id: string;
  signPattern: string;
  dreamContext: string;
  emotionalWeight: number;
  symbolicAssociation: string;
  memoryStrength: number;
  dreamFrequency: number;
  translationAccuracy: number;
}

interface DreamMetrics {
  totalDreams: number;
  learningEfficiency: number;
  memoryConsolidation: number;
  patternRecognition: number;
  dreamClarity: number;
  emotionalProcessing: number;
  signIntegration: number;
}

export default function DreamStateLearning() {
  const [isDreaming, setIsDreaming] = useState(false);
  const [dreamPhase, setDreamPhase] = useState<'pre-sleep' | 'light-sleep' | 'deep-sleep' | 'rem-sleep'>('rem-sleep');
  const [patterns, setPatterns] = useState<DreamPattern[]>([]);
  const [modules, setModules] = useState<LearningModule[]>([]);
  const [signs, setSigns] = useState<DreamSign[]>([]);
  const [metrics, setMetrics] = useState<DreamMetrics>({
    totalDreams: 0,
    learningEfficiency: 0,
    memoryConsolidation: 0,
    patternRecognition: 0,
    dreamClarity: 0,
    emotionalProcessing: 0,
    signIntegration: 0
  });

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationRef = useRef<number>();

  useEffect(() => {
    initializeDreamStateLearning();
    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, []);

  const initializeDreamStateLearning = async () => {
    console.log('🌙 Initializing Dream-State Learning System...');
    
    // Initialize learning modules
    const initialModules: LearningModule[] = [
      {
        id: 'pre-sleep-module',
        name: 'Pre-Sleep Pattern Recognition',
        dreamPhase: 'pre-sleep',
        learningRate: 0.65,
        memoryConsolidation: 0.45,
        patternRecognition: 0.72,
        neuralPlasticity: 0.58,
        dreamIntegration: 0.38
      },
      {
        id: 'light-sleep-module',
        name: 'Light Sleep Memory Processing',
        dreamPhase: 'light-sleep',
        learningRate: 0.78,
        memoryConsolidation: 0.62,
        patternRecognition: 0.85,
        neuralPlasticity: 0.71,
        dreamIntegration: 0.54
      },
      {
        id: 'deep-sleep-module',
        name: 'Deep Sleep Consolidation',
        dreamPhase: 'deep-sleep',
        learningRate: 0.89,
        memoryConsolidation: 0.94,
        patternRecognition: 0.67,
        neuralPlasticity: 0.82,
        dreamIntegration: 0.73
      },
      {
        id: 'rem-sleep-module',
        name: 'REM Dream Integration',
        dreamPhase: 'rem-sleep',
        learningRate: 0.95,
        memoryConsolidation: 0.87,
        patternRecognition: 0.91,
        neuralPlasticity: 0.96,
        dreamIntegration: 0.98
      }
    ];

    // Initialize dream patterns
    const initialPatterns: DreamPattern[] = [
      {
        id: 'lucid-dream-1',
        dreamType: 'lucid',
        intensity: 0.92,
        emotionalContent: 0.78,
        symbolicMeaning: 'self-awareness and control',
        neuralActivity: Array.from({ length: 20 }, () => Math.random()),
        duration: 25, // minutes
        clarity: 0.89
      },
      {
        id: 'rem-dream-2',
        dreamType: 'rem',
        intensity: 0.87,
        emotionalContent: 0.65,
        symbolicMeaning: 'memory consolidation',
        neuralActivity: Array.from({ length: 20 }, () => Math.random()),
        duration: 18,
        clarity: 0.76
      },
      {
        id: 'deep-dream-3',
        dreamType: 'deep',
        intensity: 0.71,
        emotionalContent: 0.43,
        symbolicMeaning: 'subconscious processing',
        neuralActivity: Array.from({ length: 20 }, () => Math.random()),
        duration: 35,
        clarity: 0.58
      },
      {
        id: 'prophetic-dream-4',
        dreamType: 'prophetic',
        intensity: 0.84,
        emotionalContent: 0.91,
        symbolicMeaning: 'future insight',
        neuralActivity: Array.from({ length: 20 }, () => Math.random()),
        duration: 22,
        clarity: 0.82
      }
    ];

    // Initialize dream signs
    const initialSigns: DreamSign[] = [
      {
        id: 'dream-sign-1',
        signPattern: 'floating-hands-communication',
        dreamContext: 'lucid dream of flying',
        emotionalWeight: 0.87,
        symbolicAssociation: 'freedom and transcendence',
        memoryStrength: 0.92,
        dreamFrequency: 0.78,
        translationAccuracy: 0.89
      },
      {
        id: 'dream-sign-2',
        signPattern: 'water-flow-emotion',
        dreamContext: 'ocean dream with waves',
        emotionalWeight: 0.73,
        symbolicAssociation: 'emotional depth and intuition',
        memoryStrength: 0.85,
        dreamFrequency: 0.65,
        translationAccuracy: 0.82
      },
      {
        id: 'dream-sign-3',
        signPattern: 'light-illumination',
        dreamContext: 'bright light in darkness',
        emotionalWeight: 0.91,
        symbolicAssociation: 'hope and enlightenment',
        memoryStrength: 0.88,
        dreamFrequency: 0.71,
        translationAccuracy: 0.86
      }
    ];

    setModules(initialModules);
    setPatterns(initialPatterns);
    setSigns(initialSigns);
    
    startDreamProcessing();
    
    console.log('✅ Dream-State Learning initialized with', initialModules.length, 'modules');
  };

  const startDreamProcessing = () => {
    const process = () => {
      // Update dream patterns
      setPatterns(prevPatterns => 
        prevPatterns.map(pattern => ({
          ...pattern,
          intensity: Math.max(0.3, pattern.intensity + (Math.random() - 0.5) * 0.05),
          emotionalContent: Math.max(0.2, pattern.emotionalContent + (Math.random() - 0.5) * 0.08),
          clarity: Math.max(0.2, pattern.clarity + (Math.random() - 0.5) * 0.06),
          neuralActivity: pattern.neuralActivity.map(activity => 
            Math.max(0, Math.min(1, activity + (Math.random() - 0.5) * 0.1))
          )
        }))
      );

      // Update learning modules
      setModules(prevModules => 
        prevModules.map(module => ({
          ...module,
          learningRate: Math.min(0.99, module.learningRate + (Math.random() - 0.4) * 0.02),
          memoryConsolidation: Math.min(0.99, module.memoryConsolidation + (Math.random() - 0.3) * 0.01),
          patternRecognition: Math.min(0.99, module.patternRecognition + (Math.random() - 0.4) * 0.015),
          neuralPlasticity: Math.min(0.99, module.neuralPlasticity + (Math.random() - 0.3) * 0.008),
          dreamIntegration: Math.min(0.99, module.dreamIntegration + (Math.random() - 0.2) * 0.005)
        }))
      );

      // Update dream signs
      setSigns(prevSigns => 
        prevSigns.map(sign => ({
          ...sign,
          memoryStrength: Math.min(0.99, sign.memoryStrength + (Math.random() - 0.4) * 0.01),
          dreamFrequency: Math.max(0.1, sign.dreamFrequency + (Math.random() - 0.5) * 0.02),
          translationAccuracy: Math.min(0.99, sign.translationAccuracy + (Math.random() - 0.3) * 0.008)
        }))
      );

      // Update metrics
      setMetrics(prev => ({
        ...prev,
        totalDreams: prev.totalDreams + (Math.random() > 0.9 ? 1 : 0),
        learningEfficiency: Math.min(0.99, prev.learningEfficiency + 0.001),
        memoryConsolidation: Math.min(0.99, prev.memoryConsolidation + 0.0008),
        patternRecognition: Math.min(0.99, prev.patternRecognition + 0.0012),
        dreamClarity: Math.min(0.99, prev.dreamClarity + 0.0006),
        emotionalProcessing: Math.min(0.99, prev.emotionalProcessing + 0.0009),
        signIntegration: Math.min(0.99, prev.signIntegration + 0.0011)
      }));

      animationRef.current = requestAnimationFrame(process);
    };
    process();
  };

  const performDreamLearning = async () => {
    setIsDreaming(true);
    
    // Simulate dream-state learning
    await new Promise(resolve => setTimeout(resolve, 3000));
    
    console.log('🌙 Dream-State Learning Complete:', {
      phase: dreamPhase,
      patterns: patterns.length,
      signs: signs.length,
      efficiency: metrics.learningEfficiency,
      integration: metrics.signIntegration
    });
    
    setIsDreaming(false);
  };

  const renderDreamVisualization = () => {
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
    
    // Draw dream layers
    for (let i = 0; i < 4; i++) {
      const layerRadius = radius * (1 + i * 0.3);
      const alpha = 0.15 - i * 0.03;
      
      ctx.strokeStyle = `rgba(147, 51, 234, ${alpha})`;
      ctx.lineWidth = 2;
      ctx.setLineDash([10, 15]);
      ctx.beginPath();
      ctx.arc(centerX, centerY, layerRadius, 0, Math.PI * 2);
      ctx.stroke();
      ctx.setLineDash([]);
    }
    
    // Draw dream patterns
    patterns.forEach((pattern, index) => {
      const angle = (index / patterns.length) * Math.PI * 2 - Math.PI / 2;
      const distance = (pattern.intensity * radius * 0.7) + radius * 0.3;
      const x = centerX + Math.cos(angle) * distance;
      const y = centerY + Math.sin(angle) * distance;
      
      // Draw pattern based on dream type
      const colors = {
        'lucid': 'rgba(255, 255, 100, ',
        'rem': 'rgba(100, 200, 255, ',
        'deep': 'rgba(150, 100, 200, ',
        'nightmare': 'rgba(255, 100, 100, ',
        'prophetic': 'rgba(100, 255, 200, '
      };
      
      const baseColor = colors[pattern.dreamType] || 'rgba(200, 200, 200, ';
      
      // Draw dream bubble
      const gradient = ctx.createRadialGradient(x, y, 0, x, y, 20);
      gradient.addColorStop(0, baseColor + pattern.intensity + ')');
      gradient.addColorStop(1, baseColor + '0)');
      
      ctx.fillStyle = gradient;
      ctx.beginPath();
      ctx.arc(x, y, 12 + pattern.intensity * 8, 0, Math.PI * 2);
      ctx.fill();
      
      // Draw emotional content
      if (pattern.emotionalContent > 0.6) {
        ctx.strokeStyle = baseColor + pattern.emotionalContent + ')';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(x, y, 15 + pattern.intensity * 10, 0, Math.PI * 2);
        ctx.stroke();
      }
      
      // Draw neural connections
      pattern.neuralActivity.slice(0, 5).forEach((activity, i) => {
        if (activity > 0.7) {
          const connectionAngle = angle + (i - 2) * 0.3;
          const connectionX = centerX + Math.cos(connectionAngle) * radius * 1.2;
          const connectionY = centerY + Math.sin(connectionAngle) * radius * 1.2;
          
          ctx.strokeStyle = baseColor + activity + ')';
          ctx.lineWidth = activity * 2;
          ctx.beginPath();
          ctx.moveTo(x, y);
          ctx.lineTo(connectionX, connectionY);
          ctx.stroke();
        }
      });
    });
    
    // Draw central dream core
    const coreGradient = ctx.createRadialGradient(centerX, centerY, 0, centerX, centerY, 35);
    coreGradient.addColorStop(0, `rgba(255, 255, 255, ${metrics.dreamClarity})`);
    coreGradient.addColorStop(1, `rgba(255, 255, 255, 0.2)`);
    
    ctx.fillStyle = coreGradient;
    ctx.beginPath();
    ctx.arc(centerX, centerY, 35, 0, Math.PI * 2);
    ctx.fill();
    
    // Draw dream symbol
    ctx.fillStyle = '#9333ea';
    ctx.font = '28px Arial';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText('🌙', centerX, centerY);
  };

  useEffect(() => {
    renderDreamVisualization();
  }, [patterns, metrics]);

  const getDreamTypeColor = (type: string) => {
    const colors: Record<string, string> = {
      'lucid': 'bg-yellow-600',
      'rem': 'bg-blue-600',
      'deep': 'bg-purple-600',
      'nightmare': 'bg-red-600',
      'prophetic': 'bg-green-600'
    };
    return colors[type] || 'bg-gray-600';
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-2xl font-bold text-white flex items-center gap-2">
            <Moon className="w-6 h-6 text-indigo-400" />
            Dream-State Learning
          </h3>
          <p className="text-gray-400">Subconscious Sign Integration</p>
        </div>
        <div className="flex gap-2">
          <Button
            onClick={performDreamLearning}
            disabled={isDreaming}
            className="bg-indigo-600 hover:bg-indigo-700"
          >
            {isDreaming ? (
              <>
                <Brain className="w-4 h-4 mr-2 animate-pulse" />
                Dream Processing...
              </>
            ) : (
              <>
                <Sparkles className="w-4 h-4 mr-2" />
                Start Dream Learning
              </>
            )}
          </Button>
        </div>
      </div>

      {/* Dream Metrics */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <Card className="bg-black/30 border-indigo-500/20">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-gray-400 text-sm">Total Dreams</p>
                <p className="text-2xl font-bold text-white">{metrics.totalDreams}</p>
              </div>
              <Cloud className="w-8 h-8 text-indigo-400" />
            </div>
          </CardContent>
        </Card>

        <Card className="bg-black/30 border-blue-500/20">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-gray-400 text-sm">Learning Efficiency</p>
                <p className="text-2xl font-bold text-white">{(metrics.learningEfficiency * 100).toFixed(1)}%</p>
              </div>
              <TrendingUp className="w-8 h-8 text-blue-400" />
            </div>
          </CardContent>
        </Card>

        <Card className="bg-black/30 border-purple-500/20">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-gray-400 text-sm">Memory Consolidation</p>
                <p className="text-2xl font-bold text-white">{(metrics.memoryConsolidation * 100).toFixed(1)}%</p>
              </div>
              <Brain className="w-8 h-8 text-purple-400" />
            </div>
          </CardContent>
        </Card>

        <Card className="bg-black/30 border-pink-500/20">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-gray-400 text-sm">Sign Integration</p>
                <p className="text-2xl font-bold text-white">{(metrics.signIntegration * 100).toFixed(1)}%</p>
              </div>
              <Heart className="w-8 h-8 text-pink-400" />
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Main Content */}
      <div className="grid md:grid-cols-2 gap-6">
        {/* Dream Visualization */}
        <Card className="bg-black/30 border-gray-700">
          <CardHeader>
            <CardTitle className="text-white flex items-center gap-2">
              <Eye className="w-5 h-5 text-indigo-400" />
              Dream State Visualization
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="relative aspect-video bg-gradient-to-br from-indigo-900/20 to-purple-900/20 rounded-lg overflow-hidden">
              <canvas
                ref={canvasRef}
                className="w-full h-full"
                style={{ imageRendering: 'crisp-edges' }}
              />
              
              {/* Dream Phase Status */}
              <div className="absolute top-4 left-4 bg-black/70 backdrop-blur-sm rounded-lg px-3 py-2">
                <div className="text-indigo-400 text-sm font-medium">
                  Current Phase: {dreamPhase.replace('-', ' ')}
                </div>
                <div className="text-white text-xs">
                  Clarity: {(metrics.dreamClarity * 100).toFixed(1)}%
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Dream Patterns */}
        <Card className="bg-black/30 border-gray-700">
          <CardHeader>
            <CardTitle className="text-white flex items-center gap-2">
              <Wind className="w-5 h-5 text-blue-400" />
              Dream Patterns
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3 max-h-80 overflow-y-auto">
              {patterns.map(pattern => (
                <div key={pattern.id} className="p-3 bg-black/20 border border-gray-600 rounded-lg">
                  <div className="flex items-center justify-between mb-2">
                    <h4 className="text-white font-medium capitalize">{pattern.dreamType} Dream</h4>
                    <Badge className={getDreamTypeColor(pattern.dreamType)}>
                      {(pattern.intensity * 100).toFixed(1)}% Intensity
                    </Badge>
                  </div>
                  <div className="text-gray-300 text-sm mb-2">
                    {pattern.symbolicMeaning}
                  </div>
                  <div className="grid grid-cols-2 gap-2 text-xs">
                    <div className="flex justify-between">
                      <span className="text-gray-400">Emotional:</span>
                      <span className="text-white">{(pattern.emotionalContent * 100).toFixed(1)}%</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Clarity:</span>
                      <span className="text-white">{(pattern.clarity * 100).toFixed(1)}%</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Duration:</span>
                      <span className="text-white">{pattern.duration}m</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Neural Activity:</span>
                      <span className="text-white">{(pattern.neuralActivity.reduce((sum, a) => sum + a, 0) / pattern.neuralActivity.length * 100).toFixed(1)}%</span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Learning Modules */}
      <Card className="bg-black/30 border-gray-700">
        <CardHeader>
          <CardTitle className="text-white flex items-center gap-2">
            <Brain className="w-5 h-5 text-purple-400" />
            Dream Learning Modules
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {modules.map(module => (
              <div key={module.id} className="p-4 bg-black/20 border border-gray-600 rounded-lg">
                <div className="flex items-center justify-between mb-2">
                  <h4 className="text-white font-medium">{module.name}</h4>
                  <Badge className="bg-purple-600">
                    {module.dreamPhase.replace('-', ' ')}
                  </Badge>
                </div>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-gray-400">Learning Rate:</span>
                    <span className="text-white">{(module.learningRate * 100).toFixed(1)}%</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Memory Consolidation:</span>
                    <span className="text-white">{(module.memoryConsolidation * 100).toFixed(1)}%</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Pattern Recognition:</span>
                    <span className="text-white">{(module.patternRecognition * 100).toFixed(1)}%</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Neural Plasticity:</span>
                    <span className="text-white">{(module.neuralPlasticity * 100).toFixed(1)}%</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Dream Integration:</span>
                    <span className="text-white">{(module.dreamIntegration * 100).toFixed(1)}%</span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Dream Signs */}
      <Card className="bg-black/30 border-gray-700">
        <CardHeader>
          <CardTitle className="text-white flex items-center gap-2">
            <Heart className="w-5 h-5 text-pink-400" />
            Dream-Integrated Signs
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {signs.map(sign => (
              <div key={sign.id} className="p-3 bg-black/20 border border-gray-600 rounded-lg">
                <div className="flex items-center justify-between mb-2">
                  <h4 className="text-white font-medium text-sm">{sign.signPattern}</h4>
                  <Badge className="bg-pink-600">
                    {(sign.translationAccuracy * 100).toFixed(1)}%
                  </Badge>
                </div>
                <div className="text-gray-300 text-xs mb-2">
                  {sign.dreamContext}
                </div>
                <div className="space-y-1 text-xs">
                  <div className="flex justify-between">
                    <span className="text-gray-400">Emotional Weight:</span>
                    <span className="text-white">{(sign.emotionalWeight * 100).toFixed(1)}%</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Memory Strength:</span>
                    <span className="text-white">{(sign.memoryStrength * 100).toFixed(1)}%</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Dream Frequency:</span>
                    <span className="text-white">{(sign.dreamFrequency * 100).toFixed(1)}%</span>
                  </div>
                  <div className="mt-2 p-2 bg-black/30 rounded text-xs text-pink-300">
                    {sign.symbolicAssociation}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Status Badge */}
      <div className="flex justify-center">
        <Badge className={isDreaming ? "bg-indigo-600 animate-pulse" : "bg-gray-600"}>
          {isDreaming ? "Dream Learning Active" : "Dream State Idle"}
        </Badge>
      </div>
    </div>
  );
}