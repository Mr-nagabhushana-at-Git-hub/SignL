import React, { useRef, useEffect, useState } from 'react';
import { Camera, RefreshCw, Zap, Lock, User, AlertTriangle } from 'lucide-react';

interface LiveFeedProps {
  onMotionDetected: (intensity: number) => void;
  isActive: boolean;
  selectedDeviceId: string;
}

export const LiveFeed: React.FC<LiveFeedProps> = ({ onMotionDetected, isActive, selectedDeviceId }) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [streamReady, setStreamReady] = useState(false);
  const [permissionError, setPermissionError] = useState(false);
  const [detectedUser, setDetectedUser] = useState<string | null>(null);

  // Initialize Camera
  useEffect(() => {
    let currentStream: MediaStream | null = null;
    let isMounted = true;

    const startCamera = async (retryWithDefault = false) => {
      setStreamReady(false);
      setPermissionError(false);
      
      try {
        const constraints: MediaStreamConstraints = {
          video: retryWithDefault 
            ? true 
            : (selectedDeviceId ? { deviceId: { exact: selectedDeviceId } } : true)
        };

        console.log("Requesting camera with constraints:", constraints);
        const stream = await navigator.mediaDevices.getUserMedia(constraints);
        
        if (!isMounted) {
          stream.getTracks().forEach(t => t.stop());
          return;
        }

        currentStream = stream;

        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          // Explicitly try to play
          try {
            await videoRef.current.play();
          } catch (playErr) {
            console.error("Video play error:", playErr);
          }
        }
      } catch (err: any) {
        console.error("Camera Error:", err);
        // If exact device ID fails, try default
        if (!retryWithDefault && err.name === 'OverconstrainedError') {
          console.log("Constraint failed, retrying with default camera...");
          startCamera(true);
          return;
        }
        if (isMounted) setPermissionError(true);
      }
    };

    if (isActive) {
      startCamera();
    } else {
      // Cleanup if inactive
      if (videoRef.current) {
        if (videoRef.current.srcObject) {
           const tracks = (videoRef.current.srcObject as MediaStream).getTracks();
           tracks.forEach(t => t.stop());
        }
        videoRef.current.srcObject = null;
      }
      setStreamReady(false);
    }
    
    return () => {
      isMounted = false;
      if (currentStream) {
        currentStream.getTracks().forEach(t => t.stop());
      }
    };
  }, [isActive, selectedDeviceId]);

  // Motion Detection Loop
  useEffect(() => {
    if (!streamReady || !isActive) return;

    let animationFrameId: number;
    const ctx = canvasRef.current?.getContext('2d', { willReadFrequently: true });
    const width = 100; // Low res for analysis
    const height = 75;

    let lastFrame: ImageData | null = null;
    let frameCount = 0;

    const processFrame = () => {
      if (!videoRef.current || !ctx) return;

      // Ensure video is actually playing and has dimensions
      if (videoRef.current.readyState === 4 && videoRef.current.videoWidth > 0) {
          try {
            ctx.drawImage(videoRef.current, 0, 0, width, height);
            const currentFrame = ctx.getImageData(0, 0, width, height);

            if (lastFrame) {
              let diff = 0;
              const len = currentFrame.data.length;
              for (let i = 0; i < len; i += 4) {
                // Simple green channel diff
                if (Math.abs(currentFrame.data[i + 1] - lastFrame.data[i + 1]) > 20) {
                  diff++;
                }
              }
              
              const motion = diff / (width * height);
              onMotionDetected(motion);
              
              // Simulating Face ID randomly
              frameCount++;
              if (frameCount % 100 === 0 && motion > 0.01) {
                  setDetectedUser(Math.random() > 0.5 ? "Guest (Scanning...)" : "Nagabhushana Raju S");
              }
            }

            lastFrame = currentFrame;
          } catch (e) {
            console.error("Frame processing error", e);
          }
      }
      animationFrameId = requestAnimationFrame(processFrame);
    };

    processFrame();

    return () => cancelAnimationFrame(animationFrameId);
  }, [streamReady, isActive, onMotionDetected]);

  return (
    <div className="relative h-full w-full bg-black rounded-xl overflow-hidden border border-slate-700 group">
      {/* Hidden processing canvas */}
      <canvas ref={canvasRef} width="100" height="75" className="hidden" />

      {permissionError ? (
        <div className="absolute inset-0 flex flex-col items-center justify-center text-red-500 bg-slate-900/90 z-20 p-4 text-center">
          <Lock className="w-12 h-12 mb-2" />
          <p className="font-mono text-sm">CAMERA ACCESS DENIED</p>
          <p className="text-xs text-slate-400 mt-2">Please check browser permissions.</p>
          <button 
            onClick={() => window.location.reload()} 
            className="mt-4 px-4 py-2 bg-slate-800 rounded border border-slate-700 text-xs hover:bg-slate-700 transition-colors"
          >
            RETRY
          </button>
        </div>
      ) : !isActive ? (
         <div className="absolute inset-0 flex flex-col items-center justify-center text-slate-500 bg-slate-900/90 z-20">
            <AlertTriangle className="w-10 h-10 mb-2 opacity-50" />
            <p className="font-mono text-xs">SENSOR OFFLINE</p>
         </div>
      ) : !streamReady && (
        <div className="absolute inset-0 flex flex-col items-center justify-center text-cyan-500/50 bg-black z-20">
          <RefreshCw className="w-8 h-8 animate-spin mb-2" />
          <p className="font-mono text-xs">INITIALIZING SENSOR ARRAY...</p>
        </div>
      )}

      {/* Video Element */}
      <video 
        ref={videoRef} 
        autoPlay 
        playsInline 
        muted 
        onCanPlay={() => {
          console.log("Video can play");
          setStreamReady(true);
        }}
        className={`w-full h-full object-cover scale-x-[-1] transition-opacity duration-500 ${streamReady ? 'opacity-80' : 'opacity-0'}`} 
      />
      
      {/* AR Overlay (Only show when ready) */}
      {streamReady && (
        <div className="absolute inset-0 pointer-events-none z-10">
          {/* Corners */}
          <div className="absolute top-4 left-4 w-8 h-8 border-t-2 border-l-2 border-cyan-500"></div>
          <div className="absolute top-4 right-4 w-8 h-8 border-t-2 border-r-2 border-cyan-500"></div>
          <div className="absolute bottom-4 left-4 w-8 h-8 border-b-2 border-l-2 border-cyan-500"></div>
          <div className="absolute bottom-4 right-4 w-8 h-8 border-b-2 border-r-2 border-cyan-500"></div>

          {/* Simulated Tracking Box - Center */}
          <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-48 h-48 border border-cyan-500/30 rounded-full animate-pulse-fast"></div>
          
          {/* Metadata Tags */}
          <div className="absolute top-4 left-14 bg-black/60 backdrop-blur px-2 py-1 rounded border border-cyan-500/20 text-[10px] text-cyan-400 font-mono flex items-center gap-2">
            <Camera className="w-3 h-3" />
            INPUT: ACTIVE
          </div>

          {detectedUser && (
               <div className="absolute bottom-10 left-1/2 -translate-x-1/2 bg-cyan-950/80 border border-cyan-400 px-4 py-1 rounded text-cyan-300 font-mono text-xs flex items-center gap-2 animate-in slide-in-from-bottom-2">
                  <User className="w-3 h-3" />
                  IDENTITY: {detectedUser.toUpperCase()}
               </div>
          )}

          {/* Motion Waveform Simulation */}
          <div className="absolute bottom-0 left-0 right-0 h-16 bg-gradient-to-t from-black to-transparent flex items-end justify-center pb-2 gap-1">
               {[...Array(20)].map((_, i) => (
                   <div key={i} className="w-1 bg-cyan-500/50 animate-pulse" style={{ height: `${Math.random() * 100}%`, animationDelay: `${i * 0.05}s` }}></div>
               ))}
          </div>
        </div>
      )}
    </div>
  );
};