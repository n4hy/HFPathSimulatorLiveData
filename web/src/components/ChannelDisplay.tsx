/**
 * Channel state display component
 */

import React from 'react';
import { ChannelState } from '../api/client';
import './ChannelDisplay.css';

interface ChannelDisplayProps {
  state: ChannelState | null;
  loading?: boolean;
}

const MODEL_LABELS: Record<string, string> = {
  vogler: 'Vogler IPM',
  watterson: 'Watterson TDL',
  vogler_hoffmeyer: 'Vogler-Hoffmeyer',
  passthrough: 'Passthrough',
};

export const ChannelDisplay: React.FC<ChannelDisplayProps> = ({
  state,
  loading = false,
}) => {
  if (loading) {
    return (
      <div className="card channel-display">
        <div className="card-header">
          <h4>Channel State</h4>
        </div>
        <div className="loading">Loading...</div>
      </div>
    );
  }

  if (!state) {
    return (
      <div className="card channel-display">
        <div className="card-header">
          <h4>Channel State</h4>
        </div>
        <div className="error">No data</div>
      </div>
    );
  }

  return (
    <div className="card channel-display">
      <div className="card-header">
        <h4>Channel State</h4>
        <span className="status-indicator">
          <span className={`status-dot ${state.running ? 'running' : 'stopped'}`} />
          {state.running ? 'Processing' : 'Stopped'}
        </span>
      </div>

      <div className="channel-info">
        <div className="info-row">
          <span className="label">Model</span>
          <span className="value">{MODEL_LABELS[state.model] || state.model}</span>
        </div>
        <div className="info-row">
          <span className="label">Blocks Processed</span>
          <span className="value mono">{state.blocks_processed.toLocaleString()}</span>
        </div>
        <div className="info-row">
          <span className="label">Samples Processed</span>
          <span className="value mono">{state.total_samples_processed.toLocaleString()}</span>
        </div>
      </div>

      {state.vogler && (
        <div className="channel-section">
          <h5>Vogler Parameters</h5>
          <div className="params-grid">
            <div className="param">
              <span className="label">foF2</span>
              <span className="value mono">{state.vogler.foF2?.toFixed(1)} MHz</span>
            </div>
            <div className="param">
              <span className="label">hmF2</span>
              <span className="value mono">{state.vogler.hmF2?.toFixed(0)} km</span>
            </div>
            <div className="param">
              <span className="label">foE</span>
              <span className="value mono">{state.vogler.foE?.toFixed(1)} MHz</span>
            </div>
            <div className="param">
              <span className="label">hmE</span>
              <span className="value mono">{state.vogler.hmE?.toFixed(0)} km</span>
            </div>
            <div className="param">
              <span className="label">Doppler Spread</span>
              <span className="value mono">{state.vogler.doppler_spread_hz?.toFixed(1)} Hz</span>
            </div>
            <div className="param">
              <span className="label">Delay Spread</span>
              <span className="value mono">{state.vogler.delay_spread_ms?.toFixed(1)} ms</span>
            </div>
          </div>
        </div>
      )}

      <div className="channel-section">
        <h5>Impairments</h5>
        <div className="impairment-meters">
          <div className="impairment">
            <div className="impairment-header">
              <span>AGC</span>
              <span className={state.agc_enabled ? 'enabled' : 'disabled'}>
                {state.agc_enabled ? 'ON' : 'OFF'}
              </span>
            </div>
            {state.agc_enabled && (
              <>
                <div className="meter">
                  <div
                    className="meter-fill"
                    style={{ width: `${Math.min(100, Math.abs(state.agc_gain_db) + 50)}%` }}
                  />
                </div>
                <span className="meter-value mono">{state.agc_gain_db.toFixed(1)} dB</span>
              </>
            )}
          </div>

          <div className="impairment">
            <div className="impairment-header">
              <span>Limiter</span>
              <span className={state.limiter_enabled ? 'enabled' : 'disabled'}>
                {state.limiter_enabled ? 'ON' : 'OFF'}
              </span>
            </div>
            {state.limiter_enabled && (
              <>
                <div className="meter">
                  <div
                    className={`meter-fill ${state.limiter_reduction_db < -3 ? 'warning' : ''}`}
                    style={{ width: `${Math.min(100, Math.abs(state.limiter_reduction_db) * 10)}%` }}
                  />
                </div>
                <span className="meter-value mono">{state.limiter_reduction_db.toFixed(1)} dB</span>
              </>
            )}
          </div>

          <div className="impairment">
            <div className="impairment-header">
              <span>Freq Offset</span>
              <span className={state.freq_offset_enabled ? 'enabled' : 'disabled'}>
                {state.freq_offset_enabled ? 'ON' : 'OFF'}
              </span>
            </div>
            {state.freq_offset_enabled && (
              <span className="meter-value mono">{state.current_freq_offset_hz.toFixed(1)} Hz</span>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default ChannelDisplay;
