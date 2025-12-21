'use client';

import { useState, useEffect, useRef } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Globe, Languages, BookOpen, Brain, Zap, TrendingUp, Infinity, Network } from 'lucide-react';

interface UniversalGrammar {
  id: string;
  name: string;
  universality: number;
  complexity: number;
  languages: number;
  patterns: string[];
  mathematicalStructure: number;
  cognitiveBasis: number;
}

interface SignPattern {
  id: string;
  universalPattern: string;
  culturalVariations: Record<string, string>;
  frequency: number;
  complexity: number;
  meaning: string;
  cognitiveLoad: number;
}

interface LinguisticLaw {
  id: string;
  name: string;
  description: string;
  universality: number;
  evidence: number;
  languages: string[];
  mathematicalProof: string;
}

interface UniversalMetrics {
  totalLanguages: number;
  universalPatterns: number;
  linguisticLaws: number;
  convergenceRate: number;
  theoreticalCompleteness: number;
  practicalAccuracy: number;
  crossCulturalValidity: number;
}

export default function UniversalSignLanguageTheory() {
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [selectedLanguage, setSelectedLanguage] = useState('global');
  const [grammars, setGrammars] = useState<UniversalGrammar[]>([]);
  const [patterns, setPatterns] = useState<SignPattern[]>([]);
  const [laws, setLaws] = useState<LinguisticLaw[]>([]);
  const [metrics, setMetrics] = useState<UniversalMetrics>({
    totalLanguages: 0,
    universalPatterns: 0,
    linguisticLaws: 0,
    convergenceRate: 0,
    theoreticalCompleteness: 0,
    practicalAccuracy: 0,
    crossCulturalValidity: 0
  });

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationRef = useRef<number>();

  useEffect(() => {
    initializeUniversalTheory();
    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, []);

  const initializeUniversalTheory = async () => {
    console.log('🌍 Initializing Universal Sign Language Theory...');
    
    // Initialize universal grammars
    const initialGrammars: UniversalGrammar[] = [
      {
        id: 'spatial-grammar',
        name: 'Spatial Topology Grammar',
        universality: 0.95,
        complexity: 0.78,
        languages: 142,
        patterns: ['subject-object-verb', 'location-based', 'directional'],
        mathematicalStructure: 0.89,
        cognitiveBasis: 0.92
      },
      {
        id: 'temporal-grammar',
        name: 'Temporal Aspect Grammar',
        universality: 0.91,
        complexity: 0.82,
        languages: 138,
        patterns: ['tense-modality', 'aspectual', 'sequential'],
        mathematicalStructure: 0.85,
        cognitiveBasis: 0.88
      },
      {
        id: 'iconic-grammar',
        name: 'Iconic Representation Grammar',
        universality: 0.87,
        complexity: 0.75,
        languages: 145,
        patterns: ['pictorial', 'metaphorical', 'representational'],
        mathematicalStructure: 0.79,
        cognitiveBasis: 0.94
      },
      {
        id: 'classifier-grammar',
        name: 'Classifier System Grammar',
        universality: 0.93,
        complexity: 0.88,
        languages: 127,
        patterns: ['size-classifier', 'shape-classifier', 'quantity-classifier'],
        mathematicalStructure: 0.91,
        cognitiveBasis: 0.86
      }
    ];

    // Initialize universal patterns
    const initialPatterns: SignPattern[] = [
      {
        id: 'greeting-universal',
        universalPattern: 'open-hand-forward-motion',
        culturalVariations: {
          'ASL': 'HELLO-hand-wave',
          'BSL': 'HELLO-hand-motion',
          'JSL': 'KONNICHIWA-bow',
          'ISL': 'NAMASTE-hands',
          'CSL': 'NI-HAO-nod'
        },
        frequency: 0.98,
        complexity: 0.12,
        meaning: 'greeting/acknowledgment',
        cognitiveLoad: 0.15
      },
      {
        id: 'question-universal',
        universalPattern: 'eyebrows-raused-inquiry',
        culturalVariations: {
          'ASL': 'QUESTION-facial-expression',
          'BSL': 'QUESTION-head-tilt',
          'JSL': 'QUESTION-eyebrow-raise',
          'ISL': 'QUESTION-nose-wrinkle',
          'CSL': 'QUESTION-eye-widen'
        },
        frequency: 0.95,
        complexity: 0.25,
        meaning: 'interrogative',
        cognitiveLoad: 0.32
      },
      {
        id: 'negation-universal',
        universalPattern: 'hand-flick-negative',
        culturalVariations: {
          'ASL': 'NOT-hand-flick',
          'BSL': 'NO-head-shake',
          'JSL': 'NIE-hand-twist',
          'ISL': 'NAH-palm-out',
          'CSL': 'BU-index-finger'
        },
        frequency: 0.92,
        complexity: 0.18,
        meaning: 'negation',
        cognitiveLoad: 0.28
      }
    ];

    // Initialize linguistic laws
    const initialLaws: LinguisticLaw[] = [
      {
        id: 'economy-principle',
        name: 'Principle of Linguistic Economy',
        description: 'Sign languages minimize effort while maximizing meaning',
        universality: 0.96,
        evidence: 0.94,
        languages: ['ASL', 'BSL', 'JSL', 'ISL', 'CSL', 'LSF', 'DGS'],
        mathematicalProof: 'Minimization: ∑effort → min, ∑meaning → max'
      },
      {
        id: 'iconicity-principle',
        name: 'Principle of Systematic Iconicity',
        description: 'Form-meaning relationships are systematic across languages',
        universality: 0.89,
        evidence: 0.87,
        languages: ['ASL', 'BSL', 'JSL', 'ISL', 'CSL'],
        mathematicalProof: 'Correlation: corr(form, meaning) > 0.7'
      },
      {
        id: 'duality-principle',
        name: 'Principle of Articulatory-Perceptual Duality',
        description: 'Production constraints mirror perception constraints',
        universality: 0.93,
        evidence: 0.91,
        languages: ['ASL', 'BSL', 'JSL', 'ISL', 'CSL', 'LSF'],
        mathematicalProof: 'Constraint: ∏production ≈ ∏perception'
      }
    ];

    setGrammars(initialGrammars);
    setPatterns(initialPatterns);
    setLaws(initialLaws);
    
    // Calculate universal metrics
    const totalLanguages = new Set([
      ...initialGrammars.flatMap(g => g.languages),
      ...initialLaws.flatMap(l => l.languages)
    ]).size;

    setMetrics({
      totalLanguages,
      universalPatterns: initialPatterns.length,
      linguisticLaws: initialLaws.length,
      convergenceRate: 0.87,
      theoreticalCompleteness: 0.91,
      practicalAccuracy: 0.89,
      crossCulturalValidity: 0.93
    });

    startUniversalAnalysis();
    
    console.log('✅ Universal Sign Language Theory initialized with', totalLanguages, 'languages');
  };

  const startUniversalAnalysis = () => {
    const analyze = () => {
      // Update patterns frequency
      setPatterns(prevPatterns => 
        prevPatterns.map(pattern => ({
          ...pattern,
          frequency: Math.max(0.8, pattern.frequency + (Math.random() - 0.5) * 0.02),
          cognitiveLoad: Math.max(0.1, pattern.cognitiveLoad + (Math.random() - 0.5) * 0.01)
        }))
      );

      // Update grammars
      setGrammars(prevGrammars => 
        prevGrammars.map(grammar => ({
          ...grammar,
          universality: Math.min(0.99, grammar.universality + (Math.random() - 0.5) * 0.01),
          cognitiveBasis: Math.min(0.99, grammar.cognitiveBasis + (Math.random() - 0.5) * 0.01)
        }))
      );

      // Update metrics
      setMetrics(prev => ({
        ...prev,
        convergenceRate: Math.min(0.99, prev.convergenceRate + 0.001),
        theoreticalCompleteness: Math.min(0.99, prev.theoreticalCompleteness + 0.0005),
        practicalAccuracy: Math.min(0.99, prev.practicalAccuracy + 0.0008),
        crossCulturalValidity: Math.min(0.99, prev.crossCulturalValidity + 0.0006)
      }));

      animationRef.current = requestAnimationFrame(analyze);
    };
    analyze();
  };

  const performUniversalTranslation = async () => {
    setIsAnalyzing(true);
    
    // Simulate universal translation analysis
    await new Promise(resolve => setTimeout(resolve, 3000));
    
    console.log('🌍 Universal Translation Analysis Complete:', {
      languages: metrics.totalLanguages,
      patterns: metrics.universalPatterns,
      laws: metrics.linguisticLaws,
      convergence: metrics.convergenceRate
    });
    
    setIsAnalyzing(false);
  };

  const renderUniversalVisualization = () => {
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
    
    // Draw universal network
    grammars.forEach((grammar, i) => {
      const angle = (i / grammars.length) * Math.PI * 2 - Math.PI / 2;
      const x = centerX + Math.cos(angle) * radius;
      const y = centerY + Math.sin(angle) * radius;
      
      // Draw grammar node
      const gradient = ctx.createRadialGradient(x, y, 0, x, y, 30);
      gradient.addColorStop(0, `rgba(100, 255, 100, ${grammar.universality})`);
      gradient.addColorStop(1, `rgba(100, 255, 100, 0)`);
      
      ctx.fillStyle = gradient;
      ctx.beginPath();
      ctx.arc(x, y, 20 + grammar.universality * 10, 0, Math.PI * 2);
      ctx.fill();
      
      // Draw connections to other grammars
      grammars.forEach((otherGrammar, j) => {
        if (i < j) {
          const otherAngle = (j / grammars.length) * Math.PI * 2 - Math.PI / 2;
          const otherX = centerX + Math.cos(otherAngle) * radius;
          const otherY = centerY + Math.sin(otherAngle) * radius;
          
          const similarity = 1 - Math.abs(grammar.universality - otherGrammar.universality);
          
          ctx.strokeStyle = `rgba(100, 255, 100, ${similarity * 0.3})`;
          ctx.lineWidth = similarity * 3;
          ctx.beginPath();
          ctx.moveTo(x, y);
          ctx.lineTo(otherX, otherY);
          ctx.stroke();
        }
      });
    });
    
    // Draw central universal core
    const coreGradient = ctx.createRadialGradient(centerX, centerY, 0, centerX, centerY, 40);
    coreGradient.addColorStop(0, `rgba(255, 255, 255, ${metrics.theoreticalCompleteness})`);
    coreGradient.addColorStop(1, `rgba(255, 255, 255, 0.2)`);
    
    ctx.fillStyle = coreGradient;
    ctx.beginPath();
    ctx.arc(centerX, centerY, 40, 0, Math.PI * 2);
    ctx.fill();
    
    // Draw infinity symbol
    ctx.strokeStyle = `rgba(255, 255, 255, ${metrics.crossCulturalValidity})`;
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(centerX - 20, centerY);
    ctx.bezierCurveTo(centerX - 20, centerY - 15, centerX + 20, centerY - 15, centerX + 20, centerY);
    ctx.bezierCurveTo(centerX + 20, centerY + 15, centerX - 20, centerY + 15, centerX - 20, centerY);
    ctx.stroke();
  };

  useEffect(() => {
    renderUniversalVisualization();
  }, [grammars, metrics]);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-2xl font-bold text-white flex items-center gap-2">
            <Globe className="w-6 h-6 text-green-400" />
            Universal Sign Language Theory
          </h3>
          <p className="text-gray-400">Cross-Cultural Linguistic Universals</p>
        </div>
        <div className="flex gap-2">
          <Button
            onClick={performUniversalTranslation}
            disabled={isAnalyzing}
            className="bg-green-600 hover:bg-green-700"
          >
            {isAnalyzing ? (
              <>
                <Brain className="w-4 h-4 mr-2 animate-spin" />
                Analyzing Universals...
              </>
            ) : (
              <>
                <Zap className="w-4 h-4 mr-2" />
                Analyze Universal Patterns
              </>
            )}
          </Button>
        </div>
      </div>

      {/* Universal Metrics */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <Card className="bg-black/30 border-green-500/20">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-gray-400 text-sm">Languages</p>
                <p className="text-2xl font-bold text-white">{metrics.totalLanguages}</p>
              </div>
              <Languages className="w-8 h-8 text-green-400" />
            </div>
          </CardContent>
        </Card>

        <Card className="bg-black/30 border-blue-500/20">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-gray-400 text-sm">Universal Patterns</p>
                <p className="text-2xl font-bold text-white">{metrics.universalPatterns}</p>
              </div>
              <Network className="w-8 h-8 text-blue-400" />
            </div>
          </CardContent>
        </Card>

        <Card className="bg-black/30 border-purple-500/20">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-gray-400 text-sm">Convergence</p>
                <p className="text-2xl font-bold text-white">{(metrics.convergenceRate * 100).toFixed(1)}%</p>
              </div>
              <TrendingUp className="w-8 h-8 text-purple-400" />
            </div>
          </CardContent>
        </Card>

        <Card className="bg-black/30 border-orange-500/20">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-gray-400 text-sm">Cross-Cultural</p>
                <p className="text-2xl font-bold text-white">{(metrics.crossCulturalValidity * 100).toFixed(1)}%</p>
              </div>
              <Infinity className="w-8 h-8 text-orange-400" />
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Main Content */}
      <div className="grid md:grid-cols-2 gap-6">
        {/* Universal Grammar Network */}
        <Card className="bg-black/30 border-gray-700">
          <CardHeader>
            <CardTitle className="text-white flex items-center gap-2">
              <BookOpen className="w-5 h-5 text-green-400" />
              Universal Grammar Network
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="relative aspect-video bg-gradient-to-br from-green-900/20 to-blue-900/20 rounded-lg overflow-hidden">
              <canvas
                ref={canvasRef}
                className="w-full h-full"
                style={{ imageRendering: 'crisp-edges' }}
              />
              
              {/* Universal Theory Overlay */}
              <div className="absolute top-4 left-4 bg-black/70 backdrop-blur-sm rounded-lg px-3 py-2">
                <div className="text-green-400 text-sm font-medium">
                  Theoretical Completeness: {(metrics.theoreticalCompleteness * 100).toFixed(1)}%
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Universal Patterns */}
        <Card className="bg-black/30 border-gray-700">
          <CardHeader>
            <CardTitle className="text-white flex items-center gap-2">
              <Languages className="w-5 h-5 text-blue-400" />
              Universal Sign Patterns
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3 max-h-80 overflow-y-auto">
              {patterns.map(pattern => (
                <div key={pattern.id} className="p-3 bg-black/20 border border-gray-600 rounded-lg">
                  <div className="flex items-center justify-between mb-2">
                    <h4 className="text-white font-medium capitalize">{pattern.meaning}</h4>
                    <Badge className="bg-blue-600">
                      {(pattern.frequency * 100).toFixed(1)}% Universal
                    </Badge>
                  </div>
                  <div className="text-gray-300 text-sm mb-2">
                    {pattern.universalPattern}
                  </div>
                  <div className="grid grid-cols-2 gap-2 text-xs">
                    <div className="flex justify-between">
                      <span className="text-gray-400">Complexity:</span>
                      <span className="text-white">{(pattern.complexity * 100).toFixed(1)}%</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Cognitive Load:</span>
                      <span className="text-white">{(pattern.cognitiveLoad * 100).toFixed(1)}%</span>
                    </div>
                  </div>
                  <div className="mt-2">
                    <div className="text-gray-400 text-xs mb-1">Cultural Variations:</div>
                    <div className="flex flex-wrap gap-1">
                      {Object.entries(pattern.culturalVariations).slice(0, 3).map(([lang, variation]) => (
                        <Badge key={lang} className="bg-gray-600 text-xs">
                          {lang}: {variation.split('-')[0]}
                        </Badge>
                      ))}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Linguistic Laws */}
      <Card className="bg-black/30 border-gray-700">
        <CardHeader>
          <CardTitle className="text-white flex items-center gap-2">
            <Brain className="w-5 h-5 text-purple-400" />
            Universal Linguistic Laws
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {laws.map(law => (
              <div key={law.id} className="p-4 bg-black/20 border border-gray-600 rounded-lg">
                <div className="flex items-center justify-between mb-2">
                  <h4 className="text-white font-medium">{law.name}</h4>
                  <Badge className="bg-purple-600">
                    {(law.universality * 100).toFixed(1)}%
                  </Badge>
                </div>
                <div className="text-gray-300 text-sm mb-3">
                  {law.description}
                </div>
                <div className="space-y-1 text-xs">
                  <div className="flex justify-between">
                    <span className="text-gray-400">Evidence:</span>
                    <span className="text-white">{(law.evidence * 100).toFixed(1)}%</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Languages:</span>
                    <span className="text-white">{law.languages.length}</span>
                  </div>
                  <div className="mt-2 p-2 bg-black/30 rounded font-mono text-green-400 text-xs">
                    {law.mathematicalProof}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Status Badge */}
      <div className="flex justify-center">
        <Badge className={isAnalyzing ? "bg-green-600 animate-pulse" : "bg-gray-600"}>
          {isAnalyzing ? "Universal Analysis Active" : "Analysis Idle"}
        </Badge>
      </div>
    </div>
  );
}