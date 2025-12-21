'use client';

import { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Switch } from '@/components/ui/switch';
import { Slider } from '@/components/ui/slider';
import { Badge } from '@/components/ui/badge';
import { Settings, Camera, Volume2, Vibrate, Globe, Brain } from 'lucide-react';
import { SUPPORTED_LANGUAGES, CAMERA_RESOLUTIONS, FRAME_RATES, QUALITY_SETTINGS, ADVANCED_AI_MODELS } from '@/constants';
import { UserSettings, SignLanguage } from '@/types';

interface SettingsPanelProps {
  settings: UserSettings;
  onSettingsChange: (settings: UserSettings) => void;
}

export default function SettingsPanel({ settings, onSettingsChange }: SettingsPanelProps) {
  const [selectedLanguage, setSelectedLanguage] = useState<SignLanguage>(
    SUPPORTED_LANGUAGES.find(lang => lang.code === settings.preferredLanguage) || SUPPORTED_LANGUAGES[0]
  );

  const updateSetting = <K extends keyof UserSettings>(key: K, value: UserSettings[K]) => {
    onSettingsChange({ ...settings, [key]: value });
  };

  const updateCameraSetting = <K extends keyof UserSettings['camera']>(key: K, value: UserSettings['camera'][K]) => {
    updateSetting('camera', { ...settings.camera, [key]: value });
  };

  return (
    <div className="space-y-6">
      {/* Language Settings */}
      <Card className="bg-black/30 backdrop-blur-md border-purple-500/20">
        <CardHeader>
          <CardTitle className="text-white flex items-center gap-2">
            <Globe className="w-5 h-5" />
            Language Settings
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div>
            <label className="text-white text-sm font-medium mb-2 block">
              Primary Sign Language
            </label>
            <Select value={selectedLanguage.code} onValueChange={(value) => {
              const lang = SUPPORTED_LANGUAGES.find(l => l.code === value);
              if (lang) {
                setSelectedLanguage(lang);
                updateSetting('preferredLanguage', lang.code);
              }
            }}>
              <SelectTrigger className="bg-black/20 border-gray-600 text-white">
                <SelectValue />
              </SelectTrigger>
              <SelectContent className="bg-black/90 border-gray-600">
                {SUPPORTED_LANGUAGES.map((lang) => (
                  <SelectItem key={lang.code} value={lang.code} className="text-white hover:bg-purple-600/20">
                    <div className="flex items-center gap-2">
                      <span>{lang.flag}</span>
                      <span>{lang.name}</span>
                    </div>
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          <div className="flex items-center justify-between">
            <div>
              <div className="text-white font-medium">Auto-translate</div>
              <div className="text-gray-400 text-sm">Automatically translate detected signs</div>
            </div>
            <Switch
              checked={settings.autoTranslate}
              onCheckedChange={(checked) => updateSetting('autoTranslate', checked)}
            />
          </div>
        </CardContent>
      </Card>

      {/* Camera Settings */}
      <Card className="bg-black/30 backdrop-blur-md border-purple-500/20">
        <CardHeader>
          <CardTitle className="text-white flex items-center gap-2">
            <Camera className="w-5 h-5" />
            Camera Settings
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div>
            <label className="text-white text-sm font-medium mb-2 block">
              Resolution
            </label>
            <Select value={settings.camera.resolution} onValueChange={(value) => updateCameraSetting('resolution', value)}>
              <SelectTrigger className="bg-black/20 border-gray-600 text-white">
                <SelectValue />
              </SelectTrigger>
              <SelectContent className="bg-black/90 border-gray-600">
                {CAMERA_RESOLUTIONS.map((resolution) => (
                  <SelectItem key={resolution.value} value={resolution.value} className="text-white hover:bg-purple-600/20">
                    {resolution.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          <div>
            <label className="text-white text-sm font-medium mb-2 block">
              Frame Rate: {settings.camera.frameRate} FPS
            </label>
            <Slider
              value={[settings.camera.frameRate]}
              onValueChange={([value]) => updateCameraSetting('frameRate', value)}
              max={120}
              min={15}
              step={15}
              className="w-full"
            />
          </div>

          <div>
            <label className="text-white text-sm font-medium mb-2 block">
              Quality: {Math.round(settings.camera.quality * 100)}%
            </label>
            <Slider
              value={[settings.camera.quality]}
              onValueChange={([value]) => updateCameraSetting('quality', value)}
              max={1}
              min={0.5}
              step={0.1}
              className="w-full"
            />
          </div>
        </CardContent>
      </Card>

      {/* Audio & Haptic Settings */}
      <Card className="bg-black/30 backdrop-blur-md border-purple-500/20">
        <CardHeader>
          <CardTitle className="text-white flex items-center gap-2">
            <Volume2 className="w-5 h-5" />
            Audio & Haptic
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <div className="text-white font-medium">Sound Effects</div>
              <div className="text-gray-400 text-sm">Play sounds for translations</div>
            </div>
            <Switch
              checked={settings.soundEnabled}
              onCheckedChange={(checked) => updateSetting('soundEnabled', checked)}
            />
          </div>

          <div className="flex items-center justify-between">
            <div>
              <div className="text-white font-medium flex items-center gap-2">
                <Vibrate className="w-4 h-4" />
                Vibration
              </div>
              <div className="text-gray-400 text-sm">Vibrate on successful translation</div>
            </div>
            <Switch
              checked={settings.vibrationEnabled}
              onCheckedChange={(checked) => updateSetting('vibrationEnabled', checked)}
            />
          </div>
        </CardContent>
      </Card>

      {/* AI Models */}
      <Card className="bg-black/30 backdrop-blur-md border-purple-500/20">
        <CardHeader>
          <CardTitle className="text-white flex items-center gap-2">
            <Brain className="w-5 h-5" />
            Neural Network Models
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            {ADVANCED_AI_MODELS.map((model) => (
              <div key={model.id} className="flex items-center justify-between p-3 bg-black/20 rounded-lg border border-gray-600">
                <div className="flex items-center gap-3">
                  <Brain className="w-4 h-4 text-purple-400" />
                  <div>
                    <div className="text-white font-medium text-sm">{model.name}</div>
                    <div className="text-gray-400 text-xs">v{model.version}</div>
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  <div className="text-right">
                    <div className="text-cyan-400 text-xs">{(model.accuracy * 100).toFixed(1)}%</div>
                    <div className="text-green-400 text-xs">{(model.speed * 100).toFixed(1)}% speed</div>
                  </div>
                  <Badge className={model.loaded ? "bg-green-600" : "bg-gray-600"} variant="secondary">
                    {model.loaded ? "Active" : "Offline"}
                  </Badge>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Reset Button */}
      <div className="flex justify-center">
        <Button variant="outline" className="border-gray-600 text-white hover:bg-gray-600/20">
          Reset to Default Settings
        </Button>
      </div>
    </div>
  );
}