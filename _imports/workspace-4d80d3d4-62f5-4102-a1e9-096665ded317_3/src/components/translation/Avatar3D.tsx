'use client';

import { useState, useEffect, useRef } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Slider } from '@/components/ui/slider';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { 
  User, 
  Settings, 
  RotateCcw, 
  Maximize2, 
  Volume2, 
  Eye,
  Smile,
  Zap,
  Brain
} from 'lucide-react';
import { FaceData, EmotionData } from '@/types';
import { EMOTION_CATEGORIES } from '@/constants';

interface Avatar3DProps {
  isActive: boolean;
  detectedFaces: FaceData[];
  currentTranslation: string;
  isSpeaking: boolean;
}

export default function Avatar3D({ 
  isActive, 
  detectedFaces, 
  currentTranslation, 
  isSpeaking 
}: Avatar3DProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationRef = useRef<number>();
  const [avatarStyle, setAvatarStyle] = useState<'realistic' | 'cartoon' | 'cyberpunk'>('cyberpunk');
  const [avatarGender, setAvatarGender] = useState<'neutral' | 'male' | 'female'>('neutral');
  const [avatarEmotion, setAvatarEmotion] = useState<string>('neutral');
  const [animationSpeed, setAnimationSpeed] = useState(1.0);
  const [showSkeleton, setShowSkeleton] = useState(false);
  const [autoEmotion, setAutoEmotion] = useState(true);

  // Avatar state
  const [avatarPosition, setAvatarPosition] = useState({ x: 0, y: 0, z: 0 });
  const [avatarRotation, setAvatarRotation] = useState({ x: 0, y: 0, z: 0 });
  const [lipSync, setLipSync] = useState(0);
  const [blinkState, setBlinkState] = useState(0);

  useEffect(() => {
    if (isActive && canvasRef.current) {
      initializeCanvas();
      startAnimation();
    } else {
      stopAnimation();
    }

    return () => stopAnimation();
  }, [isActive]);

  useEffect(() => {
    // Update avatar emotion based on detected faces
    if (autoEmotion && detectedFaces.length > 0) {
      const primaryFace = detectedFaces[0];
      if (primaryFace.emotion) {
        setAvatarEmotion(primaryFace.emotion.primary);
      }
    }
  }, [detectedFaces, autoEmotion]);

  useEffect(() => {
    // Trigger lip sync when speaking
    if (isSpeaking) {
      const interval = setInterval(() => {
        setLipSync(Math.random());
      }, 100);
      return () => clearInterval(interval);
    } else {
      setLipSync(0);
    }
  }, [isSpeaking]);

  useEffect(() => {
    // Random blinking
    const blinkInterval = setInterval(() => {
      if (Math.random() > 0.95) {
        setBlinkState(1);
        setTimeout(() => setBlinkState(0), 150);
      }
    }, 2000);
    return () => clearInterval(blinkInterval);
  }, []);

  const initializeCanvas = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Set canvas size
    canvas.width = canvas.offsetWidth;
    canvas.height = canvas.offsetHeight;
  };

  const startAnimation = () => {
    const animate = () => {
      drawAvatar();
      animationRef.current = requestAnimationFrame(animate);
    };
    animate();
  };

  const stopAnimation = () => {
    if (animationRef.current) {
      cancelAnimationFrame(animationRef.current);
    }
  };

  const drawAvatar = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Set up drawing context
    ctx.save();
    ctx.translate(canvas.width / 2, canvas.height / 2);

    // Draw based on style
    switch (avatarStyle) {
      case 'realistic':
        drawRealisticAvatar(ctx, canvas);
        break;
      case 'cartoon':
        drawCartoonAvatar(ctx, canvas);
        break;
      case 'cyberpunk':
        drawCyberpunkAvatar(ctx, canvas);
        break;
    }

    ctx.restore();
  };

  const drawCyberpunkAvatar = (ctx: CanvasRenderingContext2D, canvas: HTMLCanvasElement) => {
    const time = Date.now() * 0.001 * animationSpeed;
    const breathing = Math.sin(time * 2) * 5;
    
    // Head shape with glow effect
    ctx.shadowColor = '#8B5CF6';
    ctx.shadowBlur = 20;
    ctx.fillStyle = '#1a1a2e';
    ctx.beginPath();
    ctx.arc(0, -50 + breathing, 80, 0, Math.PI * 2);
    ctx.fill();
    
    // Cyberpunk face details
    ctx.shadowBlur = 0;
    
    // Eyes with LED effect
    const eyeGlow = Math.sin(time * 3) * 0.5 + 0.5;
    ctx.shadowColor = '#00ffff';
    ctx.shadowBlur = 10 * eyeGlow;
    
    // Left eye
    ctx.fillStyle = blinkState > 0.5 ? '#1a1a2e' : '#00ffff';
    ctx.beginPath();
    ctx.arc(-25, -60 + breathing, 12, 0, Math.PI * 2);
    ctx.fill();
    
    // Right eye
    ctx.beginPath();
    ctx.arc(25, -60 + breathing, 12, 0, Math.PI * 2);
    ctx.fill();
    
    // Mouth with lip sync
    ctx.shadowBlur = 0;
    ctx.strokeStyle = '#ff00ff';
    ctx.lineWidth = 2;
    ctx.beginPath();
    const mouthWidth = 30 + lipSync * 10;
    ctx.moveTo(-mouthWidth/2, -20 + breathing);
    ctx.lineTo(mouthWidth/2, -20 + breathing);
    ctx.stroke();
    
    // Neural circuit lines
    ctx.strokeStyle = '#8B5CF6';
    ctx.lineWidth = 1;
    ctx.globalAlpha = 0.6;
    
    for (let i = 0; i < 5; i++) {
      const angle = (time + i * 1.2) % (Math.PI * 2);
      const x = Math.cos(angle) * 100;
      const y = Math.sin(angle) * 100 - 50;
      
      ctx.beginPath();
      ctx.moveTo(0, -50 + breathing);
      ctx.lineTo(x, y);
      ctx.stroke();
    }
    
    ctx.globalAlpha = 1;
    
    // Emotion indicator
    const emotionData = EMOTION_CATEGORIES.find(e => e.key === avatarEmotion);
    if (emotionData) {
      ctx.fillStyle = emotionData.color;
      ctx.font = '24px Arial';
      ctx.fillText(emotionData.icon, -10, 50);
    }
    
    // Status indicators
    drawStatusIndicators(ctx, canvas);
  };

  const drawCartoonAvatar = (ctx: CanvasRenderingContext2D, canvas: HTMLCanvasElement) => {
    const time = Date.now() * 0.001 * animationSpeed;
    const breathing = Math.sin(time * 2) * 3;
    
    // Simple cartoon head
    ctx.fillStyle = '#fdbcb4';
    ctx.beginPath();
    ctx.arc(0, -50 + breathing, 70, 0, Math.PI * 2);
    ctx.fill();
    
    // Eyes
    ctx.fillStyle = '#ffffff';
    ctx.beginPath();
    ctx.arc(-20, -65 + breathing, 15, 0, Math.PI * 2);
    ctx.arc(20, -65 + breathing, 15, 0, Math.PI * 2);
    ctx.fill();
    
    // Pupils
    ctx.fillStyle = '#000000';
    if (blinkState <= 0.5) {
      ctx.beginPath();
      ctx.arc(-20, -65 + breathing, 8, 0, Math.PI * 2);
      ctx.arc(20, -65 + breathing, 8, 0, Math.PI * 2);
      ctx.fill();
    }
    
    // Mouth
    ctx.strokeStyle = '#000000';
    ctx.lineWidth = 3;
    ctx.beginPath();
    if (lipSync > 0.1) {
      ctx.arc(0, -25 + breathing, 15 + lipSync * 5, 0, Math.PI);
    } else {
      ctx.moveTo(-20, -25 + breathing);
      ctx.lineTo(20, -25 + breathing);
    }
    ctx.stroke();
    
    // Emotion
    const emotionData = EMOTION_CATEGORIES.find(e => e.key === avatarEmotion);
    if (emotionData) {
      ctx.font = '32px Arial';
      ctx.fillText(emotionData.icon, -15, 50);
    }
  };

  const drawRealisticAvatar = (ctx: CanvasRenderingContext2D, canvas: HTMLCanvasElement) => {
    const time = Date.now() * 0.001 * animationSpeed;
    const breathing = Math.sin(time * 1.5) * 2;
    
    // More realistic head shape
    ctx.fillStyle = '#f4c2a1';
    ctx.beginPath();
    ctx.ellipse(0, -50 + breathing, 75, 85, 0, 0, Math.PI * 2);
    ctx.fill();
    
    // Shading for depth
    const gradient = ctx.createRadialGradient(-20, -70, 10, 0, -50, 80);
    gradient.addColorStop(0, 'rgba(255, 255, 255, 0.3)');
    gradient.addColorStop(1, 'rgba(0, 0, 0, 0.1)');
    ctx.fillStyle = gradient;
    ctx.fill();
    
    // Realistic eyes
    if (blinkState <= 0.5) {
      // Eye whites
      ctx.fillStyle = '#ffffff';
      ctx.beginPath();
      ctx.ellipse(-25, -60 + breathing, 18, 12, 0, 0, Math.PI * 2);
      ctx.ellipse(25, -60 + breathing, 18, 12, 0, 0, Math.PI * 2);
      ctx.fill();
      
      // Irises
      ctx.fillStyle = '#4a90e2';
      ctx.beginPath();
      ctx.arc(-25, -60 + breathing, 10, 0, Math.PI * 2);
      ctx.arc(25, -60 + breathing, 10, 0, Math.PI * 2);
      ctx.fill();
      
      // Pupils
      ctx.fillStyle = '#000000';
      ctx.beginPath();
      ctx.arc(-25, -60 + breathing, 5, 0, Math.PI * 2);
      ctx.arc(25, -60 + breathing, 5, 0, Math.PI * 2);
      ctx.fill();
    } else {
      // Closed eyes
      ctx.strokeStyle = '#000000';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(-35, -60 + breathing);
      ctx.lineTo(-15, -60 + breathing);
      ctx.moveTo(15, -60 + breathing);
      ctx.lineTo(35, -60 + breathing);
      ctx.stroke();
    }
    
    // Nose
    ctx.strokeStyle = '#d4a574';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(0, -45 + breathing);
    ctx.lineTo(0, -30 + breathing);
    ctx.stroke();
    
    // Mouth with more realistic shape
    ctx.strokeStyle = '#d4a574';
    ctx.lineWidth = 3;
    ctx.beginPath();
    if (lipSync > 0.1) {
      ctx.arc(0, -20 + breathing, 12 + lipSync * 3, 0, Math.PI);
    } else {
      ctx.moveTo(-15, -20 + breathing);
      ctx.quadraticCurveTo(0, -15 + breathing, 15, -20 + breathing);
    }
    ctx.stroke();
  };

  const drawStatusIndicators = (ctx: CanvasRenderingContext2D, canvas: HTMLCanvasElement) => {
    // Connection status
    ctx.fillStyle = isActive ? '#10b981' : '#ef4444';
    ctx.beginPath();
    ctx.arc(canvas.width / 2 - 20, 20, 5, 0, Math.PI * 2);
    ctx.fill();
    
    // Speaking status
    if (isSpeaking) {
      ctx.fillStyle = '#f59e0b';
      ctx.beginPath();
      ctx.arc(canvas.width / 2, 20, 5, 0, Math.PI * 2);
      ctx.fill();
    }
    
    // Face detection status
    if (detectedFaces.length > 0) {
      ctx.fillStyle = '#8b5cf6';
      ctx.beginPath();
      ctx.arc(canvas.width / 2 + 20, 20, 5, 0, Math.PI * 2);
      ctx.fill();
    }
  };

  const resetAvatar = () => {
    setAvatarPosition({ x: 0, y: 0, z: 0 });
    setAvatarRotation({ x: 0, y: 0, z: 0 });
    setAvatarEmotion('neutral');
    setLipSync(0);
  };

  return (
    <Card className="bg-black/30 backdrop-blur-md border-purple-500/20">
      <CardHeader>
        <CardTitle className="text-white flex items-center gap-2">
          <User className="w-5 h-5" />
          3D Neural Avatar
        </CardTitle>
        <div className="flex items-center gap-2">
          <Badge className={isActive ? "bg-green-600" : "bg-red-600"}>
            {isActive ? "Active" : "Inactive"}
          </Badge>
          <Badge className={isSpeaking ? "bg-yellow-600" : "bg-gray-600"}>
            {isSpeaking ? "Speaking" : "Silent"}
          </Badge>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* 3D Avatar Canvas */}
        <div className="relative aspect-video bg-gradient-to-br from-purple-900/20 to-cyan-900/20 rounded-lg overflow-hidden">
          <canvas
            ref={canvasRef}
            className="w-full h-full"
            style={{ imageRendering: 'crisp-edges' }}
          />
          
          {/* Status Overlay */}
          <div className="absolute top-4 left-4 space-y-2">
            {detectedFaces.length > 0 && (
              <div className="bg-black/70 backdrop-blur-sm rounded-lg px-3 py-2">
                <div className="flex items-center gap-2 text-purple-400">
                  <Eye className="w-4 h-4" />
                  <span className="text-sm font-medium">{detectedFaces.length} Face{detectedFaces.length > 1 ? 's' : ''}</span>
                </div>
              </div>
            )}
            
            {currentTranslation && (
              <div className="bg-black/70 backdrop-blur-sm rounded-lg px-3 py-2 max-w-xs">
                <div className="text-cyan-400 text-sm font-medium">
                  "{currentTranslation}"
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Avatar Controls */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <label className="text-white text-sm font-medium mb-2 block">Avatar Style</label>
            <Select value={avatarStyle} onValueChange={(value: any) => setAvatarStyle(value)}>
              <SelectTrigger className="bg-black/20 border-gray-600 text-white">
                <SelectValue />
              </SelectTrigger>
              <SelectContent className="bg-black/90 border-gray-600">
                <SelectItem value="realistic" className="text-white hover:bg-purple-600/20">
                  🎭 Realistic
                </SelectItem>
                <SelectItem value="cartoon" className="text-white hover:bg-purple-600/20">
                  🎨 Cartoon
                </SelectItem>
                <SelectItem value="cyberpunk" className="text-white hover:bg-purple-600/20">
                  🤖 Cyberpunk
                </SelectItem>
              </SelectContent>
            </Select>
          </div>

          <div>
            <label className="text-white text-sm font-medium mb-2 block">Gender</label>
            <Select value={avatarGender} onValueChange={(value: any) => setAvatarGender(value)}>
              <SelectTrigger className="bg-black/20 border-gray-600 text-white">
                <SelectValue />
              </SelectTrigger>
              <SelectContent className="bg-black/90 border-gray-600">
                <SelectItem value="neutral" className="text-white hover:bg-purple-600/20">
                  ⚧ Neutral
                </SelectItem>
                <SelectItem value="male" className="text-white hover:bg-purple-600/20">
                  👨 Male
                </SelectItem>
                <SelectItem value="female" className="text-white hover:bg-purple-600/20">
                  👩 Female
                </SelectItem>
              </SelectContent>
            </Select>
          </div>
        </div>

        <div>
          <label className="text-white text-sm font-medium mb-2 block">
            Animation Speed: {animationSpeed.toFixed(1)}x
          </label>
          <Slider
            value={[animationSpeed]}
            onValueChange={([value]) => setAnimationSpeed(value)}
            max={3.0}
            min={0.1}
            step={0.1}
            className="w-full"
          />
        </div>

        <div>
          <label className="text-white text-sm font-medium mb-2 block">Emotion</label>
          <Select value={avatarEmotion} onValueChange={setAvatarEmotion}>
            <SelectTrigger className="bg-black/20 border-gray-600 text-white">
              <SelectValue />
            </SelectTrigger>
            <SelectContent className="bg-black/90 border-gray-600">
              {EMOTION_CATEGORIES.map((emotion) => (
                <SelectItem key={emotion.key} value={emotion.key} className="text-white hover:bg-purple-600/20">
                  <div className="flex items-center gap-2">
                    <span>{emotion.icon}</span>
                    <span>{emotion.label}</span>
                  </div>
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        {/* Control Buttons */}
        <div className="flex items-center justify-center gap-4">
          <Button
            variant="outline"
            size="sm"
            onClick={() => setShowSkeleton(!showSkeleton)}
          >
            <Brain className="w-4 h-4 mr-1" />
            {showSkeleton ? 'Hide' : 'Show'} Skeleton
          </Button>
          
          <Button
            variant="outline"
            size="sm"
            onClick={() => setAutoEmotion(!autoEmotion)}
          >
            <Smile className="w-4 h-4 mr-1" />
            Auto Emotion
          </Button>
          
          <Button
            variant="outline"
            size="sm"
            onClick={resetAvatar}
          >
            <RotateCcw className="w-4 h-4 mr-1" />
            Reset
          </Button>
          
          <Button
            variant="outline"
            size="sm"
          >
            <Maximize2 className="w-4 h-4 mr-1" />
            Fullscreen
          </Button>
        </div>

        {/* Performance Stats */}
        <div className="grid grid-cols-3 gap-4 text-center">
          <div className="bg-black/20 rounded-lg p-3">
            <div className="text-cyan-400 text-lg font-bold">
              {detectedFaces.length}
            </div>
            <div className="text-gray-400 text-xs">Faces Tracked</div>
          </div>
          <div className="bg-black/20 rounded-lg p-3">
            <div className="text-purple-400 text-lg font-bold">
              {avatarEmotion}
            </div>
            <div className="text-gray-400 text-xs">Current Emotion</div>
          </div>
          <div className="bg-black/20 rounded-lg p-3">
            <div className="text-green-400 text-lg font-bold">
              {animationSpeed.toFixed(1)}x
            </div>
            <div className="text-gray-400 text-xs">Anim Speed</div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}