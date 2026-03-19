/**
 * Real-time spectrum display using Plotly
 */

import React, { useMemo } from 'react';
import Plot from 'react-plotly.js';
import { useSpectrumStream } from '../hooks/useWebSocket';

interface SpectrumProps {
  title?: string;
  height?: number;
}

export const Spectrum: React.FC<SpectrumProps> = ({
  title = 'Spectrum',
  height = 250,
}) => {
  const { spectrum, connected } = useSpectrumStream(100);

  const plotData = useMemo(() => {
    if (!spectrum) {
      return [{
        x: [],
        y: [],
        type: 'scatter' as const,
        mode: 'lines' as const,
        line: { color: '#00d4ff', width: 1 },
      }];
    }

    // Convert Hz to MHz for display
    const freqMHz = spectrum.freq_axis_hz.map(f => f / 1e6);

    return [{
      x: freqMHz,
      y: spectrum.spectrum_db,
      type: 'scatter' as const,
      mode: 'lines' as const,
      fill: 'tozeroy',
      line: { color: '#00d4ff', width: 1 },
      fillcolor: 'rgba(0, 212, 255, 0.2)',
    }];
  }, [spectrum]);

  const layout = useMemo(() => ({
    title: {
      text: title,
      font: { color: '#e6e6e6', size: 14 },
    },
    xaxis: {
      title: 'Frequency (MHz)',
      color: '#a0a0a0',
      gridcolor: '#2d3748',
      zerolinecolor: '#2d3748',
    },
    yaxis: {
      title: 'Power (dB)',
      color: '#a0a0a0',
      gridcolor: '#2d3748',
      zerolinecolor: '#2d3748',
      range: [-100, 0],
    },
    paper_bgcolor: 'transparent',
    plot_bgcolor: '#1f2940',
    margin: { t: 40, r: 20, b: 50, l: 60 },
    showlegend: false,
    autosize: true,
  }), [title]);

  const config = useMemo(() => ({
    displayModeBar: false,
    responsive: true,
  }), []);

  return (
    <div className="card">
      <div className="card-header">
        <h4>{title}</h4>
        <span className={`status-indicator`}>
          <span className={`status-dot ${connected ? 'running' : 'stopped'}`} />
          {connected ? 'Live' : 'Disconnected'}
        </span>
      </div>
      <Plot
        data={plotData}
        layout={layout}
        config={config}
        style={{ width: '100%', height }}
        useResizeHandler
      />
    </div>
  );
};

export default Spectrum;
