import React, { useEffect, useState } from 'react';
import { Activity, Cpu, Wifi } from 'lucide-react';
import { AVATAR_PLACEHOLDER } from '../constants';

interface HyperAvatarProps {
  isSigning: boolean;
  currentGloss: string;
}

export const HyperAvatar: React.FC<HyperAvatarProps> = ({ isSigning, currentGloss }) => {
  const [engineLoad, setEngineLoad] = useState(34);

  useEffect(() => {
    const interval = setInterval(() => {
      setEngineLoad(prev => isSigning ? Math.min(99, prev + Math.random() * 10) : Math.max(30, prev - Math.random() * 5));
    }, 500);
    return () => clearInterval(interval);
  }, [isSigning]);

  return (
    <div className="relative w-full h-full bg-slate-900 overflow-hidden rounded-2xl border border-slate-700 shadow-2xl group">
      {/* Background Grid */}
      <div className="absolute inset-0 bg-[linear-gradient(rgba(15,23,42,0.9)_1px,transparent_1px),linear-gradient(90deg,rgba(15,23,42,0.9)_1px,transparent_1px)] bg-[size:40px_40px] opacity-20 pointer-events-none"></div>

      {/* Avatar Visual */}
      <div className="absolute inset-0 flex items-center justify-center">
        <div className={`relative w-full h-full transition-transform duration-700 ${isSigning ? 'scale-105' : 'scale-100'}`}>
          <img 
            src={AVATAR_PLACEHOLDER} 
            alt="Neural Avatar" 
            className="w-full h-full object-cover opacity-80 mix-blend-normal filter contrast-125 saturate-50"
            style={{
                objectPosition: "center 20%"
            }}
          />
          {/* Holographic Overlay */}
          <div className="absolute inset-0 bg-gradient-to-t from-slate-900 via-transparent to-cyan-900/20 mix-blend-overlay"></div>
          
          {/* Breathing Animation / Interference */}
          <div className="absolute inset-0 bg-cyan-500/5 animate-pulse mix-blend-overlay"></div>

          {/* Scanning Line */}
          {isSigning && (
             <div className="absolute inset-0 border-b-2 border-cyan-500/30 animate-scan pointer-events-none bg-gradient-to-b from-transparent to-cyan-500/10"></div>
          )}
        </div>
      </div>

      {/* Gloss Overlay */}
      <div className="absolute bottom-10 left-0 right-0 flex justify-center z-10">
        {currentGloss ? (
          <div className="bg-black/80 backdrop-blur-md border border-cyan-500/50 px-8 py-4 rounded-2xl shadow-[0_0_25px_rgba(6,182,212,0.5)] transform transition-all duration-300 animate-in fade-in slide-in-from-bottom-4">
             <span className="text-3xl font-mono font-bold text-cyan-400 tracking-[0.2em] uppercase drop-shadow-[0_0_10px_rgba(6,182,212,0.8)]">
               {currentGloss}
             </span>
          </div>
        ) : (
          <div className="text-cyan-500/30 font-mono text-xs tracking-widest animate-pulse">
              NEURAL AVATAR IDLE
          </div>
        )}
      </div>

      {/* HUD Elements */}
      <div className="absolute top-4 left-4 flex flex-col gap-2 z-10">
        <div className="flex items-center gap-2 px-3 py-1 bg-black/40 backdrop-blur rounded border border-slate-700">
          <Cpu className="w-4 h-4 text-cyan-400" />
          <span className="text-xs font-mono text-cyan-200">NEURAL ENGINE V10</span>
        </div>
        <div className="flex items-center gap-2 px-3 py-1 bg-black/40 backdrop-blur rounded border border-slate-700">
          <Activity className="w-4 h-4 text-emerald-400" />
          <span className="text-xs font-mono text-emerald-200">LATENCY: 12ms</span>
        </div>
      </div>

      <div className="absolute top-4 right-4 text-right z-10">
        <div className="text-xs font-mono text-slate-400 mb-1">RENDERING LOAD</div>
        <div className="w-32 h-2 bg-slate-800 rounded-full overflow-hidden border border-slate-700">
          <div 
            className={`h-full transition-all duration-500 ${engineLoad > 80 ? 'bg-red-500 shadow-[0_0_10px_red]' : 'bg-cyan-500 shadow-[0_0_10px_cyan]'}`}
            style={{ width: `${engineLoad}%` }}
          ></div>
        </div>
      </div>

      <div className="absolute bottom-4 right-4 flex items-center gap-2 text-xs font-mono text-slate-500 z-10">
        <Wifi className="w-3 h-3" />
        SECURE LINK ESTABLISHED
      </div>
    </div>
  );
};