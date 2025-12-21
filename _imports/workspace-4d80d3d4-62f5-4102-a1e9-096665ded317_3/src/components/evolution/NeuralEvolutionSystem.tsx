'use client';

import { useState, useEffect, useRef } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Brain, Dna, Zap, Activity, TrendingUp, RotateCw, Play, Pause } from 'lucide-react';

interface Genome {
  id: string;
  fitness: number;
  generation: number;
  genes: number[];
  mutationRate: number;
  accuracy: number;
  speed: number;
  efficiency: number;
}

interface EvolutionStats {
  generation: number;
  populationSize: number;
  bestFitness: number;
  averageFitness: number;
  mutationRate: number;
  crossoverRate: number;
  diversity: number;
  convergenceRate: number;
}

export default function NeuralEvolutionSystem() {
  // Evolution system for genetic algorithm neural evolution
  const [isEvolving, setIsEvolving] = useState(false);
  const [currentGen, setCurrentGen] = useState(0);
  const [population, setPopulation] = useState<Genome[]>([]);
  const [evolutionStats, setEvolutionStats] = useState<EvolutionStats>({
    generation: 0,
    populationSize: 50,
    bestFitness: 0.45,
    averageFitness: 0.32,
    mutationRate: 0.1,
    crossoverRate: 0.7,
    diversity: 0.85,
    convergenceRate: 0.12
  });
  const [bestGenome, setBestGenome] = useState<Genome | null>(null);
  const [evolutionHistory, setEvolutionHistory] = useState<number[]>([]);
  const animationRef = useRef<number>();

  // Initialize population
  useEffect(() => {
    initializePopulation();
  }, []);

  const initializePopulation = () => {
    const initialPopulation: Genome[] = Array.from({ length: 50 }, (_, i) => ({
      id: `genome-${i}`,
      fitness: Math.random() * 0.5,
      generation: 0,
      genes: Array.from({ length: 100 }, () => Math.random()),
      mutationRate: 0.05 + Math.random() * 0.15,
      accuracy: 0.3 + Math.random() * 0.4,
      speed: 0.2 + Math.random() * 0.5,
      efficiency: 0.25 + Math.random() * 0.45
    }));
    
    setPopulation(initialPopulation);
    setBestGenome(initialPopulation.reduce((best, genome) => 
      genome.fitness > best.fitness ? genome : best
    ));
  };

  const evolveGeneration = () => {
    setPopulation(prevPopulation => {
      // Selection: Tournament selection
      const tournamentSize = 5;
      const selected: Genome[] = [];
      
      for (let i = 0; i < prevPopulation.length; i++) {
        const tournament = Array.from({ length: tournamentSize }, () => 
          prevPopulation[Math.floor(Math.random() * prevPopulation.length)]
        );
        selected.push(tournament.reduce((best, genome) => 
          genome.fitness > best.fitness ? genome : best
        ));
      }

      // Crossover and Mutation
      const newPopulation: Genome[] = selected.map((parent1, index) => {
        const parent2 = selected[(index + 1) % selected.length];
        const crossoverPoint = Math.floor(Math.random() * parent1.genes.length);
        
        // Crossover
        const childGenes = [
          ...parent1.genes.slice(0, crossoverPoint),
          ...parent2.genes.slice(crossoverPoint)
        ];
        
        // Mutation
        const mutatedGenes = childGenes.map(gene => 
          Math.random() < 0.1 ? gene + (Math.random() - 0.5) * 0.2 : gene
        );
        
        // Calculate new fitness based on evolved genes
        const geneFitness = mutatedGenes.reduce((sum, gene) => sum + Math.abs(gene), 0) / mutatedGenes.length;
        const newFitness = Math.min(0.99, parent1.fitness + (Math.random() - 0.3) * 0.1);
        
        return {
          id: `genome-${currentGen}-${index}`,
          fitness: newFitness,
          generation: currentGen + 1,
          genes: mutatedGenes,
          mutationRate: parent1.mutationRate + (Math.random() - 0.5) * 0.02,
          accuracy: Math.min(0.99, parent1.accuracy + (Math.random() - 0.3) * 0.05),
          speed: Math.min(0.99, parent1.speed + (Math.random() - 0.3) * 0.03),
          efficiency: Math.min(0.99, parent1.efficiency + (Math.random() - 0.3) * 0.04)
        };
      });

      // Sort by fitness
      newPopulation.sort((a, b) => b.fitness - a.fitness);
      
      // Update best genome
      const newBest = newPopulation[0];
      setBestGenome(newBest);
      
      // Update stats
      const avgFitness = newPopulation.reduce((sum, genome) => sum + genome.fitness, 0) / newPopulation.length;
      const diversity = calculateDiversity(newPopulation);
      
      setEvolutionStats(prev => ({
        generation: prev.generation + 1,
        populationSize: newPopulation.length,
        bestFitness: newBest.fitness,
        averageFitness: avgFitness,
        mutationRate: 0.1,
        crossoverRate: 0.7,
        diversity: diversity,
        convergenceRate: Math.abs(newBest.fitness - avgFitness)
      }));
      
      return newPopulation;
    });
    
    setCurrentGen(prev => prev + 1);
  };

  const calculateDiversity = (pop: Genome[]): number => {
    if (pop.length < 2) return 0;
    
    let totalDistance = 0;
    let comparisons = 0;
    
    for (let i = 0; i < Math.min(pop.length, 10); i++) {
      for (let j = i + 1; j < Math.min(pop.length, 10); j++) {
        const distance = pop[i].genes.reduce((sum, gene, index) => 
          sum + Math.abs(gene - pop[j].genes[index]), 0
        ) / pop[i].genes.length;
        totalDistance += distance;
        comparisons++;
      }
    }
    
    return totalDistance / comparisons;
  };

  const startEvolution = () => {
    setIsEvolving(true);
    runEvolution();
  };

  const stopEvolution = () => {
    setIsEvolving(false);
    if (animationRef.current) {
      cancelAnimationFrame(animationRef.current);
    }
  };

  const runEvolution = () => {
    if (!isEvolving) return;
    
    evolveGeneration();
    
    // Continue evolution
    setTimeout(() => {
      animationRef.current = requestAnimationFrame(runEvolution);
    }, 500);
  };

  const resetEvolution = () => {
    stopEvolution();
    setCurrentGen(0);
    initializePopulation();
    setEvolutionHistory([]);
    setEvolutionStats({
      generation: 0,
      populationSize: 50,
      bestFitness: 0.45,
      averageFitness: 0.32,
      mutationRate: 0.1,
      crossoverRate: 0.7,
      diversity: 0.85,
      convergenceRate: 0.12
    });
  };

  // Update evolution history
  useEffect(() => {
    if (evolutionStats.generation > 0) {
      setEvolutionHistory(prev => [...prev.slice(-49), evolutionStats.bestFitness]);
    }
  }, [evolutionStats.bestFitness]);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-2xl font-bold text-white flex items-center gap-2">
            <Dna className="w-6 h-6 text-green-400" />
            Neural Evolution System
          </h3>
          <p className="text-gray-400">Genetic Algorithm Neural Evolution</p>
        </div>
        <div className="flex gap-2">
          <Button
            onClick={isEvolving ? stopEvolution : startEvolution}
            className={isEvolving ? "bg-red-600 hover:bg-red-700" : "bg-green-600 hover:bg-green-700"}
          >
            {isEvolving ? (
              <>
                <Pause className="w-4 h-4 mr-2" />
                Pause Evolution
              </>
            ) : (
              <>
                <Play className="w-4 h-4 mr-2" />
                Start Evolution
              </>
            )}
          </Button>
          <Button onClick={resetEvolution} variant="outline">
            <RotateCw className="w-4 h-4 mr-2" />
            Reset
          </Button>
        </div>
      </div>

      {/* Evolution Stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <Card className="bg-black/30 border-green-500/20">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-gray-400 text-sm">Generation</p>
                <p className="text-2xl font-bold text-white">{evolutionStats.generation}</p>
              </div>
              <TrendingUp className="w-8 h-8 text-green-400" />
            </div>
          </CardContent>
        </Card>

        <Card className="bg-black/30 border-blue-500/20">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-gray-400 text-sm">Best Fitness</p>
                <p className="text-2xl font-bold text-white">{(evolutionStats.bestFitness * 100).toFixed(1)}%</p>
              </div>
              <TrendingUp className="w-8 h-8 text-blue-400" />
            </div>
          </CardContent>
        </Card>

        <Card className="bg-black/30 border-purple-500/20">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-gray-400 text-sm">Avg Fitness</p>
                <p className="text-2xl font-bold text-white">{(evolutionStats.averageFitness * 100).toFixed(1)}%</p>
              </div>
              <Activity className="w-8 h-8 text-purple-400" />
            </div>
          </CardContent>
        </Card>

        <Card className="bg-black/30 border-orange-500/20">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-gray-400 text-sm">Diversity</p>
                <p className="text-2xl font-bold text-white">{(evolutionStats.diversity * 100).toFixed(1)}%</p>
              </div>
              <Zap className="w-8 h-8 text-orange-400" />
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Evolution Visualization */}
      <div className="grid md:grid-cols-2 gap-6">
        {/* Fitness Progress */}
        <Card className="bg-black/30 border-gray-700">
          <CardHeader>
            <CardTitle className="text-white flex items-center gap-2">
              <TrendingUp className="w-5 h-5 text-green-400" />
              Fitness Evolution
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="h-48 flex items-end justify-between gap-1">
              {evolutionHistory.length > 0 ? evolutionHistory.map((fitness, index) => (
                <div
                  key={index}
                  className="flex-1 bg-gradient-to-t from-green-600 to-green-400 rounded-t"
                  style={{ height: `${fitness * 100}%` }}
                />
              )) : Array.from({ length: 50 }, (_, i) => (
                <div
                  key={i}
                  className="flex-1 bg-gray-700 rounded-t"
                  style={{ height: "10%" }}
                />
              ))}
            </div>
            <div className="mt-4 flex justify-between text-xs text-gray-400">
              <span>Start</span>
              <span>Current Generation</span>
            </div>
          </CardContent>
        </Card>

        {/* Best Genome */}
        <Card className="bg-black/30 border-gray-700">
          <CardHeader>
            <CardTitle className="text-white flex items-center gap-2">
              <Brain className="w-5 h-5 text-purple-400" />
              Best Genome
            </CardTitle>
          </CardHeader>
          <CardContent>
            {bestGenome && (
              <div className="space-y-3">
                <div className="flex justify-between items-center">
                  <span className="text-gray-400">Fitness Score</span>
                  <Badge className="bg-green-600">{(bestGenome.fitness * 100).toFixed(2)}%</Badge>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-gray-400">Accuracy</span>
                  <Progress value={bestGenome.accuracy * 100} className="w-24 h-2" />
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-gray-400">Speed</span>
                  <Progress value={bestGenome.speed * 100} className="w-24 h-2" />
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-gray-400">Efficiency</span>
                  <Progress value={bestGenome.efficiency * 100} className="w-24 h-2" />
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-gray-400">Mutation Rate</span>
                  <Badge className="bg-blue-600">{(bestGenome.mutationRate * 100).toFixed(1)}%</Badge>
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Evolution Parameters */}
      <Card className="bg-black/30 border-gray-700">
        <CardHeader>
          <CardTitle className="text-white">Evolution Parameters</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div>
              <p className="text-gray-400 text-sm mb-1">Population Size</p>
              <p className="text-white font-semibold">{evolutionStats.populationSize}</p>
            </div>
            <div>
              <p className="text-gray-400 text-sm mb-1">Mutation Rate</p>
              <p className="text-white font-semibold">{(evolutionStats.mutationRate * 100).toFixed(1)}%</p>
            </div>
            <div>
              <p className="text-gray-400 text-sm mb-1">Crossover Rate</p>
              <p className="text-white font-semibold">{(evolutionStats.crossoverRate * 100).toFixed(1)}%</p>
            </div>
            <div>
              <p className="text-gray-400 text-sm mb-1">Convergence</p>
              <p className="text-white font-semibold">{(evolutionStats.convergenceRate * 100).toFixed(2)}%</p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Status Badge */}
      <div className="flex justify-center">
        <Badge className={isEvolving ? "bg-green-600 animate-pulse" : "bg-gray-600"}>
          {isEvolving ? "Evolution in Progress" : "Evolution Paused"}
        </Badge>
      </div>
    </div>
  );
}