'use client';

import { useState, useEffect, useRef } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Zap, Sun, Cpu, Activity, TrendingUp, Lightbulb, Radio, Waves, Bolt } from 'lucide-react';

interface PhotonicProcessor {
  id: string;
  name: string;
  type: 'silicon' | 'inP' | 'lithium' | 'diamond';
  wavelength: number;
  processingSpeed: number;
  efficiency: number;
  temperature: number;
  powerConsumption: number;
  status: 'active' | 'idle' | 'thermal_throttle';
}

interface OpticalSignal {
  id: string;
  frequency: number;
  amplitude: number;
  phase: number;
  polarization: string;
  dataRate: number;
  latency: number;
  errorRate: number;
}

interface PhotonicLayer {
  id: string;
  name: string;
  neurons: number;
  connections: number;
  wavelength: number;
  modulation: string;
  throughput: number;
  accuracy: number;
}

interface ProcessingStats {
  totalThroughput: number;
  averageLatency: number;
  powerEfficiency: number;
  thermalLoad: number;
  opticalAccuracy: number;
  quantumCoherence: number;
  signalToNoise: number;
  processingSpeed: number;
}

export default function PhotonicNeuralNetworks() {
  const [isProcessing, setIsProcessing] = useState(false);
  const [processors, setProcessors] = useState<PhotonicProcessor[]>([]);
  const [signals, setSignals] = useState<OpticalSignal[]>([]);
  const [layers, setLayers] = useState<PhotonicLayer[]>([]);
  const [processingStats, setProcessingStats] = useState<ProcessingStats>({
    totalThroughput: 0,
    averageLatency: 0,
    powerEfficiency: 0,
    thermalLoad: 0,
    opticalAccuracy: 0,
    quantumCoherence: 0,
    signalToNoise: 0,
    processingSpeed: 0
  });
  const [performanceHistory, setPerformanceHistory] = useState<number[]>([]);
  const [selectedProcessor, setSelectedProcessor] = useState<PhotonicProcessor | null>(null);
  const animationRef = useRef<number>();

  // Initialize photonic processors
  useEffect(() => {
    initializeProcessors();
    initializeSignals();
    initializeLayers();
  }, []);

  const initializeProcessors = () => {
    const initialProcessors: PhotonicProcessor[] = [
      {
        id: 'silicon-photonic-1',
        name: 'Silicon Photonic Core Alpha',
        type: 'silicon',
        wavelength: 1550,
        processingSpeed: 100, // Teraflops equivalent
        efficiency: 0.89,
        temperature: 45.2,
        powerConsumption: 12.5,
        status: 'active'
      },
      {
        id: 'inp-processor-2',
        name: 'InP Quantum Processor Beta',
        type: 'inP',
        wavelength: 1310,
        processingSpeed: 150,
        efficiency: 0.92,
        temperature: 38.7,
        powerConsumption: 8.3,
        status: 'active'
      },
      {
        id: 'lithium-modulator-3',
        name: 'Lithium Niobate Modulator Gamma',
        type: 'lithium',
        wavelength: 780,
        processingSpeed: 75,
        efficiency: 0.85,
        temperature: 52.1,
        powerConsumption: 15.8,
        status: 'thermal_throttle'
      },
      {
        id: 'diamond-nv-4',
        name: 'Diamond NV Center Delta',
        type: 'diamond',
        wavelength: 637,
        processingSpeed: 200,
        efficiency: 0.95,
        temperature: 4.2, // Cryogenic
        powerConsumption: 3.2,
        status: 'active'
      }
    ];

    setProcessors(initialProcessors);
  };

  const initializeSignals = () => {
    const initialSignals: OpticalSignal[] = [
      {
        id: 'signal-1',
        frequency: 193.1, // THz
        amplitude: 0.85,
        phase: 0,
        polarization: 'TE',
        dataRate: 400, // Gbps
        latency: 0.2, // picoseconds
        errorRate: 0.0001
      },
      {
        id: 'signal-2',
        frequency: 228.8,
        amplitude: 0.92,
        phase: 45,
        polarization: 'TM',
        dataRate: 800,
        latency: 0.15,
        errorRate: 0.00005
      },
      {
        id: 'signal-3',
        frequency: 231.2,
        amplitude: 0.78,
        phase: 90,
        polarization: 'Circular',
        dataRate: 1200,
        latency: 0.1,
        errorRate: 0.00002
      }
    ];

    setSignals(initialSignals);
  };

  const initializeLayers = () => {
    const initialLayers: PhotonicLayer[] = [
      {
        id: 'input-layer',
        name: 'Photonic Input Layer',
        neurons: 1024,
        connections: 1048576,
        wavelength: 1550,
        modulation: 'QAM-16',
        throughput: 0,
        accuracy: 0
      },
      {
        id: 'hidden-layer-1',
        name: 'Quantum Hidden Layer 1',
        neurons: 2048,
        connections: 4194304,
        wavelength: 1310,
        modulation: 'QPSK',
        throughput: 0,
        accuracy: 0
      },
      {
        id: 'hidden-layer-2',
        name: 'Optical Hidden Layer 2',
        neurons: 1536,
        connections: 2359296,
        wavelength: 780,
        modulation: '8-PSK',
        throughput: 0,
        accuracy: 0
      },
      {
        id: 'output-layer',
        name: 'Photonic Output Layer',
        neurons: 512,
        connections: 786432,
        wavelength: 637,
        modulation: 'BPSK',
        throughput: 0,
        accuracy: 0
      }
    ];

    setLayers(initialLayers);
  };

  const startPhotonicProcessing = () => {
    setIsProcessing(true);
    runPhotonicComputation();
  };

  const stopProcessing = () => {
    setIsProcessing(false);
    if (animationRef.current) {
      cancelAnimationFrame(animationRef.current);
    }
  };

  const runPhotonicComputation = () => {
    if (!isProcessing) return;

    // Update processing stats
    setProcessingStats(prev => {
      const activeProcessors = processors.filter(p => p.status === 'active');
      const avgSpeed = activeProcessors.reduce((sum, p) => sum + p.processingSpeed, 0) / activeProcessors.length;
      const avgEfficiency = activeProcessors.reduce((sum, p) => sum + p.efficiency, 0) / activeProcessors.length;
      const avgTemp = activeProcessors.reduce((sum, p) => sum + p.temperature, 0) / activeProcessors.length;
      
      return {
        totalThroughput: Math.min(5000, prev.totalThroughput + Math.random() * 100),
        averageLatency: Math.max(0.05, prev.averageLatency + (Math.random() - 0.5) * 0.02),
        powerEfficiency: Math.min(0.98, avgEfficiency + Math.random() * 0.01),
        thermalLoad: Math.min(80, avgTemp + Math.random() * 2),
        opticalAccuracy: Math.min(0.99, prev.opticalAccuracy + 0.001),
        quantumCoherence: Math.min(0.95, prev.quantumCoherence + 0.002),
        signalToNoise: Math.min(40, prev.signalToNoise + Math.random() * 0.5),
        processingSpeed: avgSpeed
      };
    });

    // Update processors
    setProcessors(prevProcessors => prevProcessors.map(processor => {
      if (processor.status === 'idle') return processor;
      
      const tempChange = (Math.random() - 0.3) * 2;
      const newTemp = Math.max(20, Math.min(85, processor.temperature + tempChange));
      const newStatus = newTemp > 75 ? 'thermal_throttle' : 
                       newTemp > 60 ? 'active' : 'active';
      
      return {
        ...processor,
        temperature: newTemp,
        efficiency: Math.max(0.7, Math.min(0.98, processor.efficiency + (Math.random() - 0.5) * 0.02)),
        status: newStatus
      };
    }));

    // Update signals
    setSignals(prevSignals => prevSignals.map(signal => ({
      ...signal,
      amplitude: Math.max(0.5, Math.min(1.0, signal.amplitude + (Math.random() - 0.5) * 0.05)),
      phase: (signal.phase + Math.random() * 10) % 360,
      latency: Math.max(0.05, signal.latency + (Math.random() - 0.5) * 0.02),
      errorRate: Math.max(0.00001, signal.errorRate + (Math.random() - 0.7) * 0.00001)
    })));

    // Update layers
    setLayers(prevLayers => prevLayers.map(layer => ({
      ...layer,
      throughput: Math.min(1000, layer.throughput + Math.random() * 50),
      accuracy: Math.min(0.99, layer.accuracy + 0.002)
    })));

    // Continue processing
    setTimeout(() => {
      animationRef.current = requestAnimationFrame(runPhotonicComputation);
    }, 100);
  };

  // Update performance history
  useEffect(() => {
    if (processingStats.totalThroughput > 0) {
      setPerformanceHistory(prev => [...prev.slice(-49), processingStats.totalThroughput]);
    }
  }, [processingStats.totalThroughput]);

  const getProcessorTypeColor = (type: string) => {
    switch (type) {
      case 'silicon': return 'bg-blue-600';
      case 'inP': return 'bg-green-600';
      case 'lithium': return 'bg-purple-600';
      case 'diamond': return 'bg-cyan-600';
      default: return 'bg-gray-600';
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active': return 'bg-green-600';
      case 'idle': return 'bg-yellow-600';
      case 'thermal_throttle': return 'bg-red-600 animate-pulse';
      default: return 'bg-gray-600';
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-2xl font-bold text-white flex items-center gap-2">
            <Sun className="w-6 h-6 text-yellow-400" />
            Photonic Neural Networks
          </h3>
          <p className="text-gray-400">Light-Speed Optical Processing</p>
        </div>
        <div className="flex gap-2">
          <Button
            onClick={isProcessing ? stopProcessing : startPhotonicProcessing}
            className={isProcessing ? "bg-red-600 hover:bg-red-700" : "bg-yellow-600 hover:bg-yellow-700"}
          >
            {isProcessing ? (
              <>
                <Activity className="w-4 h-4 mr-2" />
                Stop Processing
              </>
            ) : (
              <>
                <Bolt className="w-4 h-4 mr-2" />
                Start Photonic Processing
              </>
            )}
          </Button>
        </div>
      </div>

      {/* Processing Stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <Card className="bg-black/30 border-yellow-500/20">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-gray-400 text-sm">Throughput</p>
                <p className="text-2xl font-bold text-white">{processingStats.totalThroughput.toFixed(0)} Gbps</p>
              </div>
              <Zap className="w-8 h-8 text-yellow-400" />
            </div>
          </CardContent>
        </Card>

        <Card className="bg-black/30 border-blue-500/20">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-gray-400 text-sm">Latency</p>
                <p className="text-2xl font-bold text-white">{processingStats.averageLatency.toFixed(2)} ps</p>
              </div>
              <Radio className="w-8 h-8 text-blue-400" />
            </div>
          </CardContent>
        </Card>

        <Card className="bg-black/30 border-green-500/20">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-gray-400 text-sm">Power Efficiency</p>
                <p className="text-2xl font-bold text-white">{(processingStats.powerEfficiency * 100).toFixed(1)}%</p>
              </div>
              <Lightbulb className="w-8 h-8 text-green-400" />
            </div>
          </CardContent>
        </Card>

        <Card className="bg-black/30 border-purple-500/20">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-gray-400 text-sm">Quantum Coherence</p>
                <p className="text-2xl font-bold text-white">{(processingStats.quantumCoherence * 100).toFixed(1)}%</p>
              </div>
              <Waves className="w-8 h-8 text-purple-400" />
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Main Content */}
      <div className="grid md:grid-cols-2 gap-6">
        {/* Photonic Processors */}
        <Card className="bg-black/30 border-gray-700">
          <CardHeader>
            <CardTitle className="text-white flex items-center gap-2">
              <Cpu className="w-5 h-5 text-blue-400" />
              Photonic Processors
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3 max-h-80 overflow-y-auto">
              {processors.map(processor => (
                <div
                  key={processor.id}
                  className="p-3 bg-black/20 border border-gray-600 rounded-lg cursor-pointer hover:border-yellow-500 transition-colors"
                  onClick={() => setSelectedProcessor(processor)}
                >
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center gap-2">
                      <Badge className={getProcessorTypeColor(processor.type)}>
                        {processor.type}
                      </Badge>
                      <Badge className={getStatusColor(processor.status)}>
                        {processor.status}
                      </Badge>
                    </div>
                    <span className="text-gray-400 text-xs">{processor.wavelength}nm</span>
                  </div>
                  <div className="text-sm text-white font-medium mb-2">{processor.name}</div>
                  <div className="grid grid-cols-2 gap-2 text-xs">
                    <div className="flex justify-between">
                      <span className="text-gray-400">Speed:</span>
                      <span className="text-white">{processor.processingSpeed} Tflops</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Efficiency:</span>
                      <span className="text-white">{(processor.efficiency * 100).toFixed(1)}%</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Temperature:</span>
                      <span className="text-white">{processor.temperature.toFixed(1)}°C</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Power:</span>
                      <span className="text-white">{processor.powerConsumption}W</span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* Optical Signals & Neural Layers */}
        <div className="space-y-6">
          {/* Optical Signals */}
          <Card className="bg-black/30 border-gray-700">
            <CardHeader>
              <CardTitle className="text-white flex items-center gap-2">
                <Radio className="w-5 h-5 text-green-400" />
                Optical Signals
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-2">
                {signals.map(signal => (
                  <div key={signal.id} className="p-2 bg-black/20 border border-gray-600 rounded">
                    <div className="flex justify-between items-center mb-1">
                      <span className="text-white text-sm">Signal {signal.id.split('-')[1]}</span>
                      <Badge className="bg-green-600">{signal.polarization}</Badge>
                    </div>
                    <div className="grid grid-cols-3 gap-2 text-xs">
                      <div>
                        <span className="text-gray-400">Freq:</span>
                        <span className="text-white ml-1">{signal.frequency.toFixed(1)} THz</span>
                      </div>
                      <div>
                        <span className="text-gray-400">Rate:</span>
                        <span className="text-white ml-1">{signal.dataRate} Gbps</span>
                      </div>
                      <div>
                        <span className="text-gray-400">Error:</span>
                        <span className="text-white ml-1">{(signal.errorRate * 100000).toFixed(1)} ppm</span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>

          {/* Performance Chart */}
          <Card className="bg-black/30 border-gray-700">
            <CardHeader>
              <CardTitle className="text-white">Photonic Performance</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="h-32 flex items-end justify-between gap-1">
                {performanceHistory.length > 0 ? performanceHistory.map((throughput, index) => (
                  <div
                    key={index}
                    className="flex-1 bg-gradient-to-t from-yellow-600 to-orange-400 rounded-t"
                    style={{ height: `${(throughput / 5000) * 100}%` }}
                  />
                )) : Array.from({ length: 50 }, (_, i) => (
                  <div
                    key={i}
                    className="flex-1 bg-gray-700 rounded-t"
                    style={{ height: "10%" }}
                  />
                ))}
              </div>
              <div className="mt-2 text-center text-xs text-gray-400">
                Throughput Over Time (Gbps)
              </div>
            </CardContent>
          </Card>
        </div>
      </div>

      {/* Neural Layers */}
      <Card className="bg-black/30 border-gray-700">
        <CardHeader>
          <CardTitle className="text-white flex items-center gap-2">
            <Activity className="w-5 h-5 text-purple-400" />
            Photonic Neural Layers
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {layers.map(layer => (
              <div key={layer.id} className="p-3 bg-black/20 border border-gray-600 rounded-lg">
                <div className="text-white font-medium mb-2">{layer.name}</div>
                <div className="space-y-1 text-xs">
                  <div className="flex justify-between">
                    <span className="text-gray-400">Neurons:</span>
                    <span className="text-white">{layer.neurons.toLocaleString()}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Connections:</span>
                    <span className="text-white">{(layer.connections / 1000000).toFixed(1)}M</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Wavelength:</span>
                    <span className="text-white">{layer.wavelength}nm</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Modulation:</span>
                    <span className="text-white">{layer.modulation}</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-gray-400">Throughput:</span>
                    <Progress value={(layer.throughput / 1000) * 100} className="w-16 h-2" />
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-gray-400">Accuracy:</span>
                    <Progress value={layer.accuracy * 100} className="w-16 h-2" />
                  </div>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Selected Processor Details */}
      {selectedProcessor && (
        <Card className="bg-black/30 border-gray-700">
          <CardHeader>
            <CardTitle className="text-white flex items-center gap-2">
              <Sun className="w-5 h-5 text-yellow-400" />
              Processor Details: {selectedProcessor.name}
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div>
                <p className="text-gray-400 text-sm">Type</p>
                <Badge className={getProcessorTypeColor(selectedProcessor.type)}>
                  {selectedProcessor.type}
                </Badge>
              </div>
              <div>
                <p className="text-gray-400 text-sm">Wavelength</p>
                <p className="text-white">{selectedProcessor.wavelength} nm</p>
              </div>
              <div>
                <p className="text-gray-400 text-sm">Processing Speed</p>
                <p className="text-white">{selectedProcessor.processingSpeed} Tflops</p>
              </div>
              <div>
                <p className="text-gray-400 text-sm">Status</p>
                <Badge className={getStatusColor(selectedProcessor.status)}>
                  {selectedProcessor.status}
                </Badge>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Status Badge */}
      <div className="flex justify-center">
        <Badge className={isProcessing ? "bg-yellow-600 animate-pulse" : "bg-gray-600"}>
          {isProcessing ? "Photonic Processing Active" : "Processing Idle"}
        </Badge>
      </div>
    </div>
  );
}