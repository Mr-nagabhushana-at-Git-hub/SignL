'use client';

import { useState, useEffect, useRef, useCallback } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Slider } from '@/components/ui/slider';
import { Progress } from '@/components/ui/progress';
import { 
  Brain, 
  Zap, 
  Activity, 
  Cpu, 
  Radio,
  Battery
} from 'lucide-react';

interface NeuromorphicConfig {
  spikingNeurons: number;
  memristorArrays: number;
  eventBasedProcessing: boolean;
  neuroSynapticPlasticity: boolean;
  spikeThreshold: number;
  refractoryPeriod: number;
  synapticDelay: number;
}

interface SpikingNeuron {
  id: number;
  membranePotential: number;
  threshold: number;
  refractoryTimer: number;
  lastSpikeTime: number;
  spikeTrain: number[];
  synapticWeights: number[];
}

interface MemristorArray {
  id: number;
  resistance: number;
  conductance: number;
  memoryState: number;
  plasticity: boolean;
}

interface EventPacket {
  timestamp: number;
  neuronId: number;
  eventType: 'spike' | 'reset' | 'plasticity';
  amplitude: number;
}

export default function NeuromorphicProcessor() {
  const [isInitialized, setIsInitialized] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [config, setConfig] = useState<NeuromorphicConfig>({
    spikingNeurons: 1000,
    memristorArrays: 64,
    eventBasedProcessing: true,
    neuroSynapticPlasticity: true,
    spikeThreshold: -55,
    refractoryPeriod: 2,
    synapticDelay: 1
  });

  const [neurons, setNeurons] = useState<SpikingNeuron[]>([]);
  const [memristors, setMemristors] = useState<MemristorArray[]>([]);
  const [eventStream, setEventStream] = useState<EventPacket[]>([]);
  const [networkActivity, setNetworkActivity] = useState(0);
  const [energyConsumption, setEnergyConsumption] = useState(0);

  const animationRef = useRef<number>();
  const simulationTime = useRef(0);

  useEffect(() => {
    initializeNeuromorphicProcessor();
    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, []);

  const initializeNeuromorphicProcessor = async () => {
    console.log('🧬 Initializing Neuromorphic Processor...');
    
    // Initialize spiking neurons
    const initialNeurons: SpikingNeuron[] = Array.from({ length: config.spikingNeurons }, (_, i) => ({
      id: i,
      membranePotential: -70 + Math.random() * 10,
      threshold: config.spikeThreshold,
      refractoryTimer: 0,
      lastSpikeTime: 0,
      spikeTrain: [],
      synapticWeights: Array.from({ length: 100 }, () => Math.random() * 0.1 - 0.05)
    }));

    // Initialize memristor arrays
    const initialMemristors: MemristorArray[] = Array.from({ length: config.memristorArrays }, (_, i) => ({
      id: i,
      resistance: 1000 + Math.random() * 9000,
      conductance: 1 / (1000 + Math.random() * 9000),
      memoryState: Math.random(),
      plasticity: config.neuroSynapticPlasticity
    }));

    setNeurons(initialNeurons);
    setMemristors(initialMemristors);
    setIsInitialized(true);
    startSpikingSimulation();
    
    console.log('✅ Neuromorphic Processor initialized with', config.spikingNeurons, 'spiking neurons');
  };

  const startSpikingSimulation = () => {
    const simulate = () => {
      simulationTime.current += 0.1;
      
      setNeurons(prevNeurons => {
        const newNeurons = prevNeurons.map(neuron => {
          let newNeuron = { ...neuron };
          
          // Handle refractory period
          if (newNeuron.refractoryTimer > 0) {
            newNeuron.refractoryTimer -= 0.1;
            return newNeuron;
          }
          
          // Simulate membrane potential dynamics
          const synapticInput = newNeuron.synapticWeights.reduce((sum, weight) => 
            sum + weight * Math.random() * 10, 0
          );
          
          newNeuron.membranePotential += synapticInput * 0.1;
          
          // Leak current
          newNeuron.membranePotential *= 0.98;
          
          // Check for spike
          if (newNeuron.membranePotential >= newNeuron.threshold) {
            newNeuron.membranePotential = -70;
            newNeuron.refractoryTimer = config.refractoryPeriod;
            newNeuron.lastSpikeTime = simulationTime.current;
            newNeuron.spikeTrain.push(simulationTime.current);
            
            // Add to event stream
            setEventStream(prev => [...prev.slice(-100), {
              timestamp: simulationTime.current,
              neuronId: newNeuron.id,
              eventType: 'spike',
              amplitude: newNeuron.membranePotential
            }]);
          }
          
          return newNeuron;
        });
        
        // Calculate network activity
        const activity = newNeurons.filter(n => n.refractoryTimer > 0).length / newNeurons.length;
        setNetworkActivity(activity);
        
        return newNeurons;
      });

      // Update memristors
      setMemristors(prevMemristors => 
        prevMemristors.map(memristor => ({
          ...memristor,
          resistance: memristor.resistance + (Math.random() - 0.5) * 10,
          conductance: 1 / (memristor.resistance + (Math.random() - 0.5) * 10),
          memoryState: memristor.memoryState * 0.99 + Math.random() * 0.01
        }))
      );

      // Update energy consumption (picojoules per spike)
      const currentSpikes = neurons.filter(n => n.refractoryTimer === config.refractoryPeriod - 0.1).length;
      setEnergyConsumption(prev => prev + currentSpikes * 0.1);

      animationRef.current = requestAnimationFrame(simulate);
    };
    simulate();
  };

  const performNeuromorphicInference = useCallback(async () => {
    setIsProcessing(true);
    
    // Simulate neuromorphic inference process
    await new Promise(resolve => setTimeout(resolve, 2000));
    
    // Calculate network metrics
    const avgSpikeRate = neurons.reduce((sum, neuron) => 
      sum + neuron.spikeTrain.length, 0
    ) / neurons.length;
    
    const synapticWeightSum = neurons.reduce((sum, neuron) => {
      const weightSum = neuron.synapticWeights.reduce((s, w) => s + Math.abs(w), 0);
      return sum + weightSum;
    }, 0) / neurons.length;
    
    const memristorEfficiency = memristors.reduce((sum, m) => 
      sum + (1 / m.resistance), 0
    ) / memristors.length;
    
    console.log('🧬 Neuromorphic Inference Complete:', {
      avgSpikeRate,
      synapticWeightSum,
      memristorEfficiency,
      networkActivity
    });
    
    setIsProcessing(false);
  }, [neurons, memristors, networkActivity]);

  const updateConfig = <K extends keyof NeuromorphicConfig>(
    key: K, 
    value: NeuromorphicConfig[K]
  ) => {
    setConfig(prev => ({ ...prev, [key]: value }));
  };

  const resetNetwork = () => {
    initializeNeuromorphicProcessor();
    setEventStream([]);
    setEnergyConsumption(0);
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <Card className="bg-black/30 backdrop-blur-md border-cyan-500/20">
        <CardHeader>
          <CardTitle className="text-white flex items-center gap-2">
            <Activity className="w-5 h-5 text-cyan-400" />
            Neuromorphic Computing
          </CardTitle>
          <div className="flex items-center gap-2">
            <Badge className={isInitialized ? "bg-green-600" : "bg-yellow-600"}>
              {isInitialized ? "Neural Active" : "Initializing"}
            </Badge>
            <Badge className={isProcessing ? "bg-blue-600" : "bg-gray-600"}>
              {isProcessing ? "Processing" : "Idle"}
            </Badge>
            <Badge className="bg-cyan-600">
              {config.spikingNeurons} Neurons
            </Badge>
          </div>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-center">
            <div>
              <div className="text-2xl font-bold text-cyan-400">
                {(networkActivity * 100).toFixed(1)}%
              </div>
              <div className="text-gray-400 text-sm">Network Activity</div>
            </div>
            <div>
              <div className="text-2xl font-bold text-green-400">
                {energyConsumption.toFixed(1)}pJ
              </div>
              <div className="text-gray-400 text-sm">Energy Used</div>
            </div>
            <div>
              <div className="text-2xl font-bold text-purple-400">
                {neurons.filter(n => n.refractoryTimer > 0).length}
              </div>
              <div className="text-gray-400 text-sm">Active Neurons</div>
            </div>
            <div>
              <div className="text-2xl font-bold text-orange-400">
                {memristors.filter(m => m.plasticity).length}
              </div>
              <div className="text-gray-400 text-sm">Plastic Memristors</div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Neural Network Visualization */}
      <Card className="bg-black/30 backdrop-blur-md border-cyan-500/20">
        <CardHeader>
          <CardTitle className="text-white flex items-center gap-2">
            <Activity className="w-5 h-5" />
            Spiking Neural Network
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="relative aspect-video bg-gradient-to-br from-cyan-900/20 to-blue-900/20 rounded-lg overflow-hidden">
            {/* Neural Network Visualization */}
            <div className="absolute inset-0 flex items-center justify-center">
              <div className="relative w-full h-full">
                {/* Visualize subset of neurons */}
                {neurons.slice(0, 50).map((neuron, i) => {
                  const x = (i % 10) * 10 + 5;
                  const y = Math.floor(i / 10) * 20 + 10;
                  const isActive = neuron.refractoryTimer > 0;
                  const intensity = Math.abs(neuron.membranePotential) / 100;
                  
                  return (
                    <div key={neuron.id}>
                      {/* Neuron */}
                      <div
                        className={`absolute w-3 h-3 rounded-full transition-all duration-100 ${
                          isActive ? 'bg-cyan-400' : 'bg-gray-600'
                        }`}
                        style={{
                          left: `${x}%`,
                          top: `${y}%`,
                          transform: 'translate(-50%, -50%)',
                          boxShadow: isActive ? `0 0 ${20 * intensity}px rgba(6, 182, 212, 0.8)` : 'none',
                          opacity: 0.3 + intensity * 0.7
                        }}
                      />
                      
                      {/* Connections to nearby neurons */}
                      {neurons.slice(i + 1, i + 4).map((connectedNeuron, j) => {
                        const jx = ((i + j + 1) % 10) * 10 + 5;
                        const jy = Math.floor((i + j + 1) / 10) * 20 + 10;
                        const weight = neuron.synapticWeights[j] || 0;
                        
                        return (
                          <svg
                            key={`${neuron.id}-${connectedNeuron.id}`}
                            className="absolute top-0 left-0 w-full h-full pointer-events-none"
                          >
                            <line
                              x1={`${x}%`}
                              y1={`${y}%`}
                              x2={`${jx}%`}
                              y2={`${jy}%`}
                              stroke={weight > 0 ? "rgba(6, 182, 212, 0.3)" : "rgba(239, 68, 68, 0.3)"}
                              strokeWidth={Math.abs(weight) * 2}
                              strokeDasharray={isActive ? "0" : "5,5"}
                            />
                          </svg>
                        );
                      })}
                    </div>
                  );
                })}
                
                {/* Event Stream Visualization */}
                <div className="absolute top-4 right-4 w-32 h-32 bg-black/50 rounded-lg p-2">
                  <div className="text-cyan-400 text-xs font-medium mb-2">Event Stream</div>
                  <div className="space-y-1">
                    {eventStream.slice(-10).reverse().map((event, i) => (
                      <div key={i} className="flex items-center gap-2 text-xs">
                        <div className={`w-2 h-2 rounded-full ${
                          event.eventType === 'spike' ? 'bg-cyan-400' : 'bg-gray-400'
                        }`} />
                        <span className="text-gray-300">
                          N{event.neuronId}: {event.amplitude.toFixed(1)}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Memristor Array Visualization */}
      <Card className="bg-black/30 backdrop-blur-md border-cyan-500/20">
        <CardHeader>
          <CardTitle className="text-white flex items-center gap-2">
            <Battery className="w-5 h-5" />
            Memristor Arrays
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-8 md:grid-cols-16 gap-2">
            {memristors.map((memristor) => (
              <div
                key={memristor.id}
                className={`aspect-square rounded flex items-center justify-center text-xs font-medium transition-all ${
                  memristor.plasticity 
                    ? 'bg-cyan-600/30 border border-cyan-500/50' 
                    : 'bg-gray-600/30 border border-gray-500/50'
                }`}
                style={{
                  opacity: 0.3 + (memristor.memoryState * 0.7)
                }}
              >
                <div className="text-cyan-300">
                  {Math.round(memristor.conductance * 1000)}
                </div>
              </div>
            ))}
          </div>
          
          <div className="mt-4 grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
            <div className="bg-black/20 rounded p-3">
              <div className="text-gray-400">Avg Resistance</div>
              <div className="text-cyan-400 font-medium">
                {(memristors.reduce((sum, m) => sum + m.resistance, 0) / memristors.length).toFixed(0)}Ω
              </div>
            </div>
            <div className="bg-black/20 rounded p-3">
              <div className="text-gray-400">Avg Conductance</div>
              <div className="text-cyan-400 font-medium">
                {(memristors.reduce((sum, m) => sum + m.conductance, 0) / memristors.length * 1000).toFixed(2)}mS
              </div>
            </div>
            <div className="bg-black/20 rounded p-3">
              <div className="text-gray-400">Memory Retention</div>
              <div className="text-cyan-400 font-medium">
                {(memristors.reduce((sum, m) => sum + m.memoryState, 0) / memristors.length * 100).toFixed(1)}%
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Configuration */}
      <Card className="bg-black/30 backdrop-blur-md border-cyan-500/20">
        <CardHeader>
          <CardTitle className="text-white flex items-center gap-2">
            <Cpu className="w-5 h-5" />
            Neuromorphic Configuration
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="text-white text-sm font-medium mb-2 block">
                Spiking Neurons: {config.spikingNeurons}
              </label>
              <Slider
                value={[config.spikingNeurons]}
                onValueChange={([value]) => updateConfig('spikingNeurons', value)}
                max={10000}
                min={100}
                step={100}
                className="w-full"
              />
            </div>

            <div>
              <label className="text-white text-sm font-medium mb-2 block">
                Memristor Arrays: {config.memristorArrays}
              </label>
              <Slider
                value={[config.memristorArrays]}
                onValueChange={([value]) => updateConfig('memristorArrays', value)}
                max={256}
                min={16}
                step={16}
                className="w-full"
              />
            </div>

            <div>
              <label className="text-white text-sm font-medium mb-2 block">
                Spike Threshold: {config.spikeThreshold}mV
              </label>
              <Slider
                value={[config.spikeThreshold]}
                onValueChange={([value]) => updateConfig('spikeThreshold', value)}
                max={-40}
                min={-70}
                step={1}
                className="w-full"
              />
            </div>

            <div>
              <label className="text-white text-sm font-medium mb-2 block">
                Refractory Period: {config.refractoryPeriod}ms
              </label>
              <Slider
                value={[config.refractoryPeriod]}
                onValueChange={([value]) => updateConfig('refractoryPeriod', value)}
                max={10}
                min={1}
                step={0.1}
                className="w-full"
              />
            </div>
          </div>

          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2">
              <input
                type="checkbox"
                id="eventBased"
                checked={config.eventBasedProcessing}
                onChange={(e) => updateConfig('eventBasedProcessing', e.target.checked)}
                className="rounded"
              />
              <label htmlFor="eventBased" className="text-white text-sm">
                Event-Based Processing
              </label>
            </div>

            <div className="flex items-center gap-2">
              <input
                type="checkbox"
                id="plasticity"
                checked={config.neuroSynapticPlasticity}
                onChange={(e) => updateConfig('neuroSynapticPlasticity', e.target.checked)}
                className="rounded"
              />
              <label htmlFor="plasticity" className="text-white text-sm">
                Synaptic Plasticity
              </label>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Control Panel */}
      <Card className="bg-black/30 backdrop-blur-md border-cyan-500/20">
        <CardContent className="pt-6">
          <div className="flex items-center justify-center gap-4">
            <Button
              onClick={performNeuromorphicInference}
              disabled={!isInitialized || isProcessing}
              className="bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-700 hover:to-blue-700"
            >
              <Zap className="w-4 h-4 mr-2" />
              {isProcessing ? 'Neural Processing...' : 'Execute Neural Inference'}
            </Button>
            
            <Button
              variant="outline"
              onClick={resetNetwork}
              className="border-cyan-500 text-cyan-400 hover:bg-cyan-600/20"
            >
              Reset Network
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}