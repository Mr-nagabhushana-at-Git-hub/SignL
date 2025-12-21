'use client';

import { useState, useEffect, useRef } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Network, Cloud, Shield, Globe, Wifi, Database, Zap, Users, Clock, TrendingUp } from 'lucide-react';

interface FederatedNode {
  id: string;
  name: string;
  location: string;
  status: 'online' | 'offline' | 'training';
  accuracy: number;
  dataPoints: number;
  contribution: number;
  latency: number;
  lastUpdate: Date;
}

interface QuantumModel {
  id: string;
  name: string;
  version: string;
  accuracy: number;
  parameters: number;
  quantumBits: number;
  entanglementStrength: number;
  trainingRounds: number;
  federatedScore: number;
}

interface TrainingStats {
  totalNodes: number;
  activeNodes: number;
  globalAccuracy: number;
  totalDataPoints: number;
  trainingRound: number;
  quantumFidelity: number;
  aggregationMethod: string;
  privacyBudget: number;
}

export default function FederatedQuantumLearning() {
  const [isTraining, setIsTraining] = useState(false);
  const [nodes, setNodes] = useState<FederatedNode[]>([]);
  const [globalModel, setGlobalModel] = useState<QuantumModel | null>(null);
  const [trainingStats, setTrainingStats] = useState<TrainingStats>({
    totalNodes: 12,
    activeNodes: 8,
    globalAccuracy: 0.0,
    totalDataPoints: 0,
    trainingRound: 0,
    quantumFidelity: 0.0,
    aggregationMethod: 'Quantum Weighted Average',
    privacyBudget: 100.0
  });
  const [trainingHistory, setTrainingHistory] = useState<number[]>([]);
  const [selectedNode, setSelectedNode] = useState<FederatedNode | null>(null);
  const animationRef = useRef<number>();

  // Initialize federated nodes
  useEffect(() => {
    initializeNodes();
    initializeGlobalModel();
  }, []);

  const initializeNodes = () => {
    const initialNodes: FederatedNode[] = [
      {
        id: 'node-us-east',
        name: 'US East Quantum Hub',
        location: 'Virginia, USA',
        status: 'online',
        accuracy: 0.89,
        dataPoints: 15420,
        contribution: 0.15,
        latency: 12,
        lastUpdate: new Date()
      },
      {
        id: 'node-eu-west',
        name: 'EU West Neural Center',
        location: 'Dublin, Ireland',
        status: 'training',
        accuracy: 0.91,
        dataPoints: 12890,
        contribution: 0.12,
        latency: 18,
        lastUpdate: new Date()
      },
      {
        id: 'node-asia-pacific',
        name: 'Asia Pacific Quantum Node',
        location: 'Singapore',
        status: 'online',
        accuracy: 0.87,
        dataPoints: 18760,
        contribution: 0.18,
        latency: 25,
        lastUpdate: new Date()
      },
      {
        id: 'node-quantum-1',
        name: 'Quantum Processor Alpha',
        location: 'Zurich, Switzerland',
        status: 'training',
        accuracy: 0.94,
        dataPoints: 9870,
        contribution: 0.10,
        latency: 8,
        lastUpdate: new Date()
      },
      {
        id: 'node-neural-2',
        name: 'Neural Network Beta',
        location: 'Tokyo, Japan',
        status: 'online',
        accuracy: 0.88,
        dataPoints: 11230,
        contribution: 0.11,
        latency: 14,
        lastUpdate: new Date()
      },
      {
        id: 'node-edge-3',
        name: 'Edge Computing Gamma',
        location: 'Sydney, Australia',
        status: 'offline',
        accuracy: 0.85,
        dataPoints: 8950,
        contribution: 0.09,
        latency: 32,
        lastUpdate: new Date(Date.now() - 300000)
      }
    ];

    setNodes(initialNodes);
    setTrainingStats(prev => ({
      ...prev,
      totalDataPoints: initialNodes.reduce((sum, node) => sum + node.dataPoints, 0),
      activeNodes: initialNodes.filter(node => node.status !== 'offline').length
    }));
  };

  const initializeGlobalModel = () => {
    setGlobalModel({
      id: 'global-quantum-v1',
      name: 'Global Quantum Sign Language Model',
      version: 'v25.0.0',
      accuracy: 0.89,
      parameters: 184000000,
      quantumBits: 1024,
      entanglementStrength: 0.78,
      trainingRounds: 0,
      federatedScore: 0.0
    });
  };

  const startFederatedTraining = () => {
    setIsTraining(true);
    runTrainingRound();
  };

  const stopTraining = () => {
    setIsTraining(false);
    if (animationRef.current) {
      cancelAnimationFrame(animationRef.current);
    }
  };

  const runTrainingRound = () => {
    if (!isTraining) return;

    // Simulate federated learning round
    setTrainingStats(prev => {
      const newRound = prev.trainingRound + 1;
      const activeNodes = nodes.filter(node => node.status !== 'offline');
      const avgNodeAccuracy = activeNodes.reduce((sum, node) => sum + node.accuracy, 0) / activeNodes.length;
      
      // Quantum aggregation improves accuracy
      const quantumImprovement = 0.001 * Math.sin(newRound * 0.1) + 0.002;
      const newGlobalAccuracy = Math.min(0.99, prev.globalAccuracy + quantumImprovement);
      const newQuantumFidelity = Math.min(0.95, prev.quantumFidelity + 0.003);

      return {
        ...prev,
        trainingRound: newRound,
        globalAccuracy: newGlobalAccuracy,
        quantumFidelity: newQuantumFidelity,
        privacyBudget: Math.max(0, prev.privacyBudget - 0.1)
      };
    });

    // Update global model
    setGlobalModel(prev => prev ? {
      ...prev,
      accuracy: Math.min(0.99, prev.accuracy + 0.002),
      trainingRounds: prev.trainingRounds + 1,
      entanglementStrength: Math.min(0.95, prev.entanglementStrength + 0.001),
      federatedScore: Math.min(0.99, prev.federatedScore + 0.003)
    } : null);

    // Update random nodes
    setNodes(prevNodes => prevNodes.map(node => {
      if (node.status === 'offline') return node;
      
      const shouldUpdate = Math.random() > 0.7;
      if (!shouldUpdate) return node;

      const accuracyImprovement = (Math.random() - 0.3) * 0.01;
      const newDataPoints = Math.floor(Math.random() * 100);
      
      return {
        ...node,
        accuracy: Math.max(0.7, Math.min(0.95, node.accuracy + accuracyImprovement)),
        dataPoints: node.dataPoints + newDataPoints,
        contribution: Math.random() * 0.2,
        latency: Math.max(5, node.latency + (Math.random() - 0.5) * 2),
        lastUpdate: new Date(),
        status: Math.random() > 0.9 ? 'training' : 'online'
      };
    }));

    // Continue training
    setTimeout(() => {
      animationRef.current = requestAnimationFrame(runTrainingRound);
    }, 2000);
  };

  // Update training history
  useEffect(() => {
    if (trainingStats.trainingRound > 0) {
      setTrainingHistory(prev => [...prev.slice(-29), trainingStats.globalAccuracy]);
    }
  }, [trainingStats.globalAccuracy]);

  const getNodeStatusColor = (status: string) => {
    switch (status) {
      case 'online': return 'bg-green-600';
      case 'training': return 'bg-blue-600 animate-pulse';
      case 'offline': return 'bg-red-600';
      default: return 'bg-gray-600';
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-2xl font-bold text-white flex items-center gap-2">
            <Network className="w-6 h-6 text-blue-400" />
            Federated Quantum Learning
          </h3>
          <p className="text-gray-400">Distributed Quantum Neural Network Training</p>
        </div>
        <div className="flex gap-2">
          <Button
            onClick={isTraining ? stopTraining : startFederatedTraining}
            className={isTraining ? "bg-red-600 hover:bg-red-700" : "bg-blue-600 hover:bg-blue-700"}
          >
            {isTraining ? (
              <>
                <Users className="w-4 h-4 mr-2" />
                Stop Training
              </>
            ) : (
              <>
                <Zap className="w-4 h-4 mr-2" />
                Start Federated Training
              </>
            )}
          </Button>
        </div>
      </div>

      {/* Training Stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <Card className="bg-black/30 border-blue-500/20">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-gray-400 text-sm">Active Nodes</p>
                <p className="text-2xl font-bold text-white">{trainingStats.activeNodes}/{trainingStats.totalNodes}</p>
              </div>
              <Globe className="w-8 h-8 text-blue-400" />
            </div>
          </CardContent>
        </Card>

        <Card className="bg-black/30 border-green-500/20">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-gray-400 text-sm">Global Accuracy</p>
                <p className="text-2xl font-bold text-white">{(trainingStats.globalAccuracy * 100).toFixed(1)}%</p>
              </div>
              <TrendingUp className="w-8 h-8 text-green-400" />
            </div>
          </CardContent>
        </Card>

        <Card className="bg-black/30 border-purple-500/20">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-gray-400 text-sm">Quantum Fidelity</p>
                <p className="text-2xl font-bold text-white">{(trainingStats.quantumFidelity * 100).toFixed(1)}%</p>
              </div>
              <Database className="w-8 h-8 text-purple-400" />
            </div>
          </CardContent>
        </Card>

        <Card className="bg-black/30 border-orange-500/20">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-gray-400 text-sm">Training Round</p>
                <p className="text-2xl font-bold text-white">{trainingStats.trainingRound}</p>
              </div>
              <Clock className="w-8 h-8 text-orange-400" />
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Main Content */}
      <div className="grid md:grid-cols-2 gap-6">
        {/* Federated Nodes */}
        <Card className="bg-black/30 border-gray-700">
          <CardHeader>
            <CardTitle className="text-white flex items-center gap-2">
              <Wifi className="w-5 h-5 text-blue-400" />
              Federated Nodes
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3 max-h-96 overflow-y-auto">
              {nodes.map(node => (
                <div
                  key={node.id}
                  className="p-3 bg-black/20 border border-gray-600 rounded-lg cursor-pointer hover:border-blue-500 transition-colors"
                  onClick={() => setSelectedNode(node)}
                >
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center gap-2">
                      <Badge className={getNodeStatusColor(node.status)}>
                        {node.status}
                      </Badge>
                      <span className="text-white font-medium text-sm">{node.name}</span>
                    </div>
                    <span className="text-gray-400 text-xs">{node.location}</span>
                  </div>
                  <div className="grid grid-cols-2 gap-2 text-xs">
                    <div className="flex justify-between">
                      <span className="text-gray-400">Accuracy:</span>
                      <span className="text-white">{(node.accuracy * 100).toFixed(1)}%</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Data Points:</span>
                      <span className="text-white">{node.dataPoints.toLocaleString()}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Contribution:</span>
                      <span className="text-white">{(node.contribution * 100).toFixed(1)}%</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Latency:</span>
                      <span className="text-white">{node.latency}ms</span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* Global Model & Training Progress */}
        <div className="space-y-6">
          {/* Global Model */}
          <Card className="bg-black/30 border-gray-700">
            <CardHeader>
              <CardTitle className="text-white flex items-center gap-2">
                <Cloud className="w-5 h-5 text-purple-400" />
                Global Quantum Model
              </CardTitle>
            </CardHeader>
            <CardContent>
              {globalModel && (
                <div className="space-y-3">
                  <div>
                    <p className="text-gray-400 text-sm mb-1">Model Name</p>
                    <p className="text-white font-medium">{globalModel.name}</p>
                  </div>
                  <div>
                    <p className="text-gray-400 text-sm mb-1">Version</p>
                    <Badge className="bg-purple-600">{globalModel.version}</Badge>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-gray-400 text-sm">Global Accuracy</span>
                    <span className="text-white font-bold">{(globalModel.accuracy * 100).toFixed(2)}%</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-gray-400 text-sm">Quantum Bits</span>
                    <span className="text-white">{globalModel.quantumBits}</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-gray-400 text-sm">Entanglement Strength</span>
                    <Progress value={globalModel.entanglementStrength * 100} className="w-24 h-2" />
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-gray-400 text-sm">Federated Score</span>
                    <Progress value={globalModel.federatedScore * 100} className="w-24 h-2" />
                  </div>
                </div>
              )}
            </CardContent>
          </Card>

          {/* Training Progress */}
          <Card className="bg-black/30 border-gray-700">
            <CardHeader>
              <CardTitle className="text-white">Training Progress</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="h-32 flex items-end justify-between gap-1">
                {trainingHistory.length > 0 ? trainingHistory.map((accuracy, index) => (
                  <div
                    key={index}
                    className="flex-1 bg-gradient-to-t from-blue-600 to-purple-400 rounded-t"
                    style={{ height: `${accuracy * 100}%` }}
                  />
                )) : Array.from({ length: 30 }, (_, i) => (
                  <div
                    key={i}
                    className="flex-1 bg-gray-700 rounded-t"
                    style={{ height: "20%" }}
                  />
                ))}
              </div>
              <div className="mt-4 grid grid-cols-2 gap-4 text-xs">
                <div>
                  <span className="text-gray-400">Aggregation Method:</span>
                  <span className="text-white ml-2">{trainingStats.aggregationMethod}</span>
                </div>
                <div>
                  <span className="text-gray-400">Privacy Budget:</span>
                  <span className="text-white ml-2">{trainingStats.privacyBudget.toFixed(1)}%</span>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>

      {/* Selected Node Details */}
      {selectedNode && (
        <Card className="bg-black/30 border-gray-700">
          <CardHeader>
            <CardTitle className="text-white flex items-center gap-2">
              <Shield className="w-5 h-5 text-green-400" />
              Node Details: {selectedNode.name}
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div>
                <p className="text-gray-400 text-sm">Status</p>
                <Badge className={getNodeStatusColor(selectedNode.status)}>
                  {selectedNode.status}
                </Badge>
              </div>
              <div>
                <p className="text-gray-400 text-sm">Location</p>
                <p className="text-white">{selectedNode.location}</p>
              </div>
              <div>
                <p className="text-gray-400 text-sm">Last Update</p>
                <p className="text-white">{selectedNode.lastUpdate.toLocaleTimeString()}</p>
              </div>
              <div>
                <p className="text-gray-400 text-sm">Node ID</p>
                <p className="text-white font-mono text-xs">{selectedNode.id}</p>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Status Badge */}
      <div className="flex justify-center">
        <Badge className={isTraining ? "bg-blue-600 animate-pulse" : "bg-gray-600"}>
          {isTraining ? "Federated Training Active" : "Training Idle"}
        </Badge>
      </div>
    </div>
  );
}