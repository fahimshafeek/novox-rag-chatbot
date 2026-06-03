import React, { useEffect, useState } from 'react';
import './Splash.css';
import novoxLogo from './assets/novox2.png';

interface SplashProps {
  onComplete: () => void;
}

const Splash: React.FC<SplashProps> = ({ onComplete }) => {
  const [phase, setPhase] = useState<'joining' | 'showing' | 'splitting' | 'zooming' | 'migrating'>('joining');

  useEffect(() => {
    const timers = [
      setTimeout(() => setPhase('showing'), 100),
      setTimeout(() => setPhase('splitting'), 2000),
      setTimeout(() => setPhase('zooming'), 2800),
      setTimeout(() => setPhase('migrating'), 3800),
      setTimeout(() => onComplete(), 5500),
    ];
    return () => timers.forEach(clearTimeout);
  }, [onComplete]);

  const NovoxAvatar = () => (
    <svg viewBox="0 0 100 100" className="splash-avatar-svg">
      <defs>
        <filter id="electric-glow-splash" x="-20%" y="-20%" width="140%" height="140%">
          <feGaussianBlur stdDeviation="2" result="blur" />
          <feDropShadow dx="0" dy="0" stdDeviation="3" floodColor="#7DF9FF" />
        </filter>
      </defs>
      <circle cx="50" cy="50" r="48" fill="#0a0c10" stroke="#1a1c22" strokeWidth="1" />
      <path 
        d="M50 50 L75 35 L90 50 L75 65 Z" 
        fill="#0047AB" 
        stroke="#7DF9FF" 
        strokeWidth="1.2"
        filter="url(#electric-glow-splash)"
      />
      <path 
        d="M10 50 L25 35 L50 50 L25 65 Z" 
        fill="#0047AB" 
        stroke="#7DF9FF" 
        strokeWidth="1.5"
        filter="url(#electric-glow-splash)"
      />
    </svg>
  );

  return (
    <div className={`splash-container phase-${phase}`}>
      <div className="logo-wrapper">
        <div className="logo-half top-half">
           <img src={novoxLogo} alt="Novox Logo" className="splash-logo-img" />
        </div>
        <div className="logo-half bottom-half">
           <img src={novoxLogo} alt="Novox Logo" className="splash-logo-img" />
        </div>
      </div>

      {(phase === 'zooming' || phase === 'migrating') && (
        <div className="avatar-migration-container">
          <NovoxAvatar />
        </div>
      )}
      
      <div className="cyber-grid"></div>
    </div>
  );
};

export default Splash;
