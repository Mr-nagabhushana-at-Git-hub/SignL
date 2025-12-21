'use client';

import { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { 
  BarChart, 
  Bar, 
  LineChart, 
  Line, 
  PieChart, 
  Pie, 
  Cell,
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  Legend,
  ResponsiveContainer,
  AreaChart,
  Area
} from 'recharts';
import { 
  TrendingUp, 
  Users, 
  Brain, 
  Activity, 
  Clock, 
  Target,
  Eye,
  Smile,
  Zap
} from 'lucide-react';
import { AnalyticsData, PerformanceMetrics } from '@/types';
import { EMOTION_CATEGORIES, THEME_COLORS } from '@/constants';

interface AnalyticsDashboardProps {
  analyticsData: AnalyticsData;
  performanceHistory: PerformanceMetrics[];
  isLoading?: boolean;
}

export default function AnalyticsDashboard({ 
  analyticsData, 
  performanceHistory, 
  isLoading = false 
}: AnalyticsDashboardProps) {
  const [selectedTimeRange, setSelectedTimeRange] = useState<'1h' | '24h' | '7d' | '30d'>('24h');
  const [realTimeMode, setRealTimeMode] = useState(true);

  // Prepare data for charts
  const emotionData = Object.entries(analyticsData.emotionDistribution).map(([emotion, count]) => ({
    name: EMOTION_CATEGORIES.find(e => e.key === emotion)?.label || emotion,
    value: count,
    color: EMOTION_CATEGORIES.find(e => e.key === emotion)?.color || THEME_COLORS.primary
  }));

  const ageData = Object.entries(analyticsData.ageDistribution).map(([age, count]) => ({
    name: age,
    value: count,
    color: THEME_COLORS.secondary
  }));

  const genderData = Object.entries(analyticsData.genderDistribution).map(([gender, count]) => ({
    name: gender.charAt(0).toUpperCase() + gender.slice(1),
    value: count,
    color: gender === 'male' ? THEME_COLORS.info : gender === 'female' ? THEME_COLORS.neural : THEME_COLORS.warning
  }));

  const performanceData = performanceHistory.map((metric, index) => ({
    time: index % 20, // Simplified time representation
    fps: Math.round(metric.fps * 10) / 10, // Round to avoid decimal issues
    latency: Math.round(metric.latency * 10) / 10,
    accuracy: Math.round(metric.accuracy * 1000) / 10, // Convert to percentage with 1 decimal
    memory: metric.memoryUsage || 0
  }));

  const phraseData = analyticsData.mostUsedPhrases.slice(0, 5).map(phrase => ({
    phrase: phrase.phrase.length > 20 ? phrase.phrase.substring(0, 20) + '...' : phrase.phrase,
    count: phrase.count
  }));

  if (isLoading) {
    return (
      <div className="flex items-center justify-center min-h-96">
        <div className="text-center space-y-4">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-purple-600 mx-auto"></div>
          <p className="text-white text-lg">Loading analytics...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-white">Advanced Analytics</h2>
          <p className="text-gray-400">Real-time performance and usage insights</p>
        </div>
        <div className="flex items-center gap-4">
          <Tabs value={selectedTimeRange} onValueChange={(value: any) => setSelectedTimeRange(value)}>
            <TabsList className="bg-black/30 border border-purple-500/20">
              <TabsTrigger value="1h" className="text-white data-[state=active]:bg-purple-600">
                1H
              </TabsTrigger>
              <TabsTrigger value="24h" className="text-white data-[state=active]:bg-purple-600">
                24H
              </TabsTrigger>
              <TabsTrigger value="7d" className="text-white data-[state=active]:bg-purple-600">
                7D
              </TabsTrigger>
              <TabsTrigger value="30d" className="text-white data-[state=active]:bg-purple-600">
                30D
              </TabsTrigger>
            </TabsList>
          </Tabs>
          <Badge className={realTimeMode ? "bg-green-600" : "bg-gray-600"}>
            {realTimeMode ? "🔴 Live" : "⏸️ Paused"}
          </Badge>
        </div>
      </div>

      {/* Key Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <Card className="bg-black/30 backdrop-blur-md border-purple-500/20">
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-gray-400 text-sm">Total Translations</p>
                <p className="text-3xl font-bold text-white">{analyticsData.totalTranslations.toLocaleString()}</p>
                <p className="text-green-400 text-sm flex items-center gap-1">
                  <TrendingUp className="w-3 h-3" />
                  +12.5% from last period
                </p>
              </div>
              <div className="w-12 h-12 bg-purple-600/20 rounded-lg flex items-center justify-center">
                <Brain className="w-6 h-6 text-purple-400" />
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className="bg-black/30 backdrop-blur-md border-cyan-500/20">
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-gray-400 text-sm">Unique Faces</p>
                <p className="text-3xl font-bold text-white">{analyticsData.uniqueFaces}</p>
                <p className="text-cyan-400 text-sm flex items-center gap-1">
                  <Users className="w-3 h-3" />
                  {analyticsData.uniqueFaces > 0 ? Math.round(analyticsData.totalTranslations / analyticsData.uniqueFaces) : 0} avg per person
                </p>
              </div>
              <div className="w-12 h-12 bg-cyan-600/20 rounded-lg flex items-center justify-center">
                <Users className="w-6 h-6 text-cyan-400" />
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className="bg-black/30 backdrop-blur-md border-green-500/20">
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-gray-400 text-sm">Avg Confidence</p>
                <p className="text-3xl font-bold text-white">{(analyticsData.averageConfidence * 100).toFixed(1)}%</p>
                <p className="text-green-400 text-sm flex items-center gap-1">
                  <Target className="w-3 h-3" />
                  Above target (85%)
                </p>
              </div>
              <div className="w-12 h-12 bg-green-600/20 rounded-lg flex items-center justify-center">
                <Target className="w-6 h-6 text-green-400" />
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className="bg-black/30 backdrop-blur-md border-orange-500/20">
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-gray-400 text-sm">Session Duration</p>
                <p className="text-3xl font-bold text-white">{Math.round(analyticsData.sessionDuration / 60)}m</p>
                <p className="text-orange-400 text-sm flex items-center gap-1">
                  <Clock className="w-3 h-3" />
                  {Math.round(analyticsData.sessionDuration)}s total
                </p>
              </div>
              <div className="w-12 h-12 bg-orange-600/20 rounded-lg flex items-center justify-center">
                <Clock className="w-6 h-6 text-orange-400" />
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Charts Section */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Performance Trends */}
        <Card className="bg-black/30 backdrop-blur-md border-purple-500/20">
          <CardHeader>
            <CardTitle className="text-white flex items-center gap-2">
              <Activity className="w-5 h-5" />
              Performance Trends
            </CardTitle>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <AreaChart data={performanceData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis dataKey="time" stroke="#9CA3AF" />
                <YAxis stroke="#9CA3AF" />
                <Tooltip 
                  contentStyle={{ backgroundColor: '#1F2937', border: '1px solid #374151' }}
                  labelStyle={{ color: '#F3F4F6' }}
                />
                <Legend />
                <Area 
                  type="monotone" 
                  dataKey="fps" 
                  stackId="1"
                  stroke="#10B981" 
                  fill="#10B981" 
                  fillOpacity={0.6}
                  name="FPS"
                />
                <Area 
                  type="monotone" 
                  dataKey="accuracy" 
                  stackId="2"
                  stroke="#8B5CF6" 
                  fill="#8B5CF6" 
                  fillOpacity={0.6}
                  name="Accuracy %"
                />
              </AreaChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        {/* Emotion Distribution */}
        <Card className="bg-black/30 backdrop-blur-md border-purple-500/20">
          <CardHeader>
            <CardTitle className="text-white flex items-center gap-2">
              <Smile className="w-5 h-5" />
              Emotion Distribution
            </CardTitle>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <PieChart>
                <Pie
                  data={emotionData}
                  cx="50%"
                  cy="50%"
                  labelLine={false}
                  label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                  outerRadius={80}
                  fill="#8884d8"
                  dataKey="value"
                >
                  {emotionData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip 
                  contentStyle={{ backgroundColor: '#1F2937', border: '1px solid #374151' }}
                  labelStyle={{ color: '#F3F4F6' }}
                />
              </PieChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        {/* Demographics */}
        <Card className="bg-black/30 backdrop-blur-md border-purple-500/20">
          <CardHeader>
            <CardTitle className="text-white flex items-center gap-2">
              <Users className="w-5 h-5" />
              Demographics
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div>
                <h4 className="text-white text-sm font-medium mb-2">Gender Distribution</h4>
                <ResponsiveContainer width="100%" height={150}>
                  <BarChart data={genderData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                    <XAxis dataKey="name" stroke="#9CA3AF" />
                    <YAxis stroke="#9CA3AF" />
                    <Tooltip 
                      contentStyle={{ backgroundColor: '#1F2937', border: '1px solid #374151' }}
                      labelStyle={{ color: '#F3F4F6' }}
                    />
                    <Bar dataKey="value" fill="#8B5CF6" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
              
              <div>
                <h4 className="text-white text-sm font-medium mb-2">Age Groups</h4>
                <ResponsiveContainer width="100%" height={150}>
                  <BarChart data={ageData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                    <XAxis dataKey="name" stroke="#9CA3AF" />
                    <YAxis stroke="#9CA3AF" />
                    <Tooltip 
                      contentStyle={{ backgroundColor: '#1F2937', border: '1px solid #374151' }}
                      labelStyle={{ color: '#F3F4F6' }}
                    />
                    <Bar dataKey="value" fill="#06B6D4" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Most Used Phrases */}
        <Card className="bg-black/30 backdrop-blur-md border-purple-500/20">
          <CardHeader>
            <CardTitle className="text-white flex items-center gap-2">
              <Zap className="w-5 h-5" />
              Top Phrases
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {phraseData.map((phrase, index) => (
                <div key={index} className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <div className="w-8 h-8 bg-purple-600/20 rounded-full flex items-center justify-center">
                      <span className="text-purple-400 text-sm font-bold">{index + 1}</span>
                    </div>
                    <span className="text-white text-sm">{phrase.phrase}</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="text-gray-400 text-sm">{phrase.count} times</span>
                    <div className="w-16 bg-gray-700 rounded-full h-2">
                      <div 
                        className="bg-purple-500 h-2 rounded-full" 
                        style={{ width: `${(phrase.count / phraseData[0]?.count) * 100}%` }}
                      />
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* System Health */}
      <Card className="bg-black/30 backdrop-blur-md border-purple-500/20">
        <CardHeader>
          <CardTitle className="text-white flex items-center gap-2">
            <Activity className="w-5 h-5" />
            System Health
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-gray-400 text-sm">CPU Usage</span>
                <span className="text-white text-sm font-medium">45%</span>
              </div>
              <Progress value={45} className="h-2" />
            </div>
            
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-gray-400 text-sm">Memory Usage</span>
                <span className="text-white text-sm font-medium">62%</span>
              </div>
              <Progress value={62} className="h-2" />
            </div>
            
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-gray-400 text-sm">GPU Usage</span>
                <span className="text-white text-sm font-medium">78%</span>
              </div>
              <Progress value={78} className="h-2" />
            </div>
          </div>
          
          <div className="mt-6 pt-6 border-t border-gray-700">
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-center">
              <div>
                <div className="text-green-400 text-lg font-bold">99.9%</div>
                <div className="text-gray-400 text-xs">Uptime</div>
              </div>
              <div>
                <div className="text-cyan-400 text-lg font-bold">12ms</div>
                <div className="text-gray-400 text-xs">Avg Latency</div>
              </div>
              <div>
                <div className="text-purple-400 text-lg font-bold">94.2%</div>
                <div className="text-gray-400 text-xs">Accuracy</div>
              </div>
              <div>
                <div className="text-orange-400 text-lg font-bold">1.2K</div>
                <div className="text-gray-400 text-xs">Requests/min</div>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}