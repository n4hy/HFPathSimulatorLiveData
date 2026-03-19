/**
 * Main dashboard component
 */

import React, { useState, useEffect } from 'react';
import apiClient, { GPUInfo, HealthStatus } from '../api/client';
import { useChannelState } from '../hooks/useChannelState';
import ChannelDisplay from './ChannelDisplay';
import Spectrum from './Spectrum';
import ControlTabs from './ControlTabs';
import './Dashboard.css';

export const Dashboard: React.FC = () => {
  const { state, loading, error, reset } = useChannelState();
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [gpuInfo, setGPUInfo] = useState<GPUInfo | null>(null);

  useEffect(() => {
    const fetchSystemInfo = async () => {
      try {
        const [h, g] = await Promise.all([
          apiClient.getHealth(),
          apiClient.getGPUInfo(),
        ]);
        setHealth(h);
        setGPUInfo(g);
      } catch (e) {
        console.error('Failed to fetch system info:', e);
      }
    };

    fetchSystemInfo();
    const interval = setInterval(fetchSystemInfo, 5000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="dashboard">
      <header className="dashboard-header">
        <div className="header-title">
          <h1>HF Path Simulator</h1>
          <span className="version">v{health?.version || '0.1.0'}</span>
        </div>
        <div className="header-status">
          {gpuInfo?.available && (
            <span className="gpu-badge">
              GPU: {gpuInfo.name}
            </span>
          )}
          {health && (
            <span className="uptime">
              Uptime: {formatUptime(health.uptime_seconds)}
            </span>
          )}
        </div>
      </header>

      {error && (
        <div className="error-banner">
          Error: {error}
          <button onClick={reset} className="secondary">Retry</button>
        </div>
      )}

      <main className="dashboard-main">
        <div className="dashboard-left">
          <Spectrum title="Output Spectrum" height={280} />
          <ChannelDisplay state={state} loading={loading} />
        </div>

        <div className="dashboard-right">
          <ControlTabs />
        </div>
      </main>
    </div>
  );
};

function formatUptime(seconds: number): string {
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const secs = Math.floor(seconds % 60);

  if (hours > 0) {
    return `${hours}h ${minutes}m`;
  }
  if (minutes > 0) {
    return `${minutes}m ${secs}s`;
  }
  return `${secs}s`;
}

export default Dashboard;
