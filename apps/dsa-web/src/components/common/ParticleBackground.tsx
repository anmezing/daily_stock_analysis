import type React from 'react';

/**
 * Lightweight animated background for the login page.
 * Kept dependency-free (CSS gradients + a slow pulse) instead of a canvas loop.
 */
export const ParticleBackground: React.FC = () => (
  <div aria-hidden className="pointer-events-none fixed inset-0 overflow-hidden">
    <div className="absolute inset-0 bg-[radial-gradient(circle_at_20%_20%,rgba(34,211,238,0.12),transparent_45%),radial-gradient(circle_at_80%_30%,rgba(168,85,247,0.10),transparent_40%),radial-gradient(circle_at_50%_80%,rgba(34,211,238,0.08),transparent_45%)]" />
    <div className="absolute inset-0 opacity-40 [background-image:radial-gradient(rgba(255,255,255,0.18)_1px,transparent_1px)] [background-size:28px_28px]" />
    <div className="absolute -left-24 top-1/4 h-72 w-72 rounded-full bg-cyan/10 blur-3xl animate-pulse" />
    <div className="absolute -right-24 bottom-1/4 h-72 w-72 rounded-full bg-purple/10 blur-3xl animate-pulse [animation-delay:2s]" />
  </div>
);
