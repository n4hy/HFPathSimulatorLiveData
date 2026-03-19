/**
 * Tabbed control panel for channel configuration
 */

import React, { useState } from 'react';
import { useChannelState } from '../hooks/useChannelState';
import './ControlTabs.css';

type TabId = 'channel' | 'noise' | 'impairments';

export const ControlTabs: React.FC = () => {
  const [activeTab, setActiveTab] = useState<TabId>('channel');
  const {
    state,
    configureVogler,
    configureWatterson,
    configureNoise,
    configureAGC,
    configureLimiter,
    configureFreqOffset,
    reset,
  } = useChannelState();

  // Local state for form inputs
  const [vogler, setVogler] = useState({
    foF2: state?.vogler?.foF2 || 7.5,
    hmF2: state?.vogler?.hmF2 || 300,
    doppler_spread_hz: state?.vogler?.doppler_spread_hz || 1.0,
    delay_spread_ms: state?.vogler?.delay_spread_ms || 2.0,
  });

  const [noise, setNoise] = useState({
    snr_db: 20,
    enable_atmospheric: false,
    enable_manmade: false,
  });

  const [agc, setAgc] = useState({
    enabled: state?.agc_enabled || false,
    target_level_db: -10,
  });

  const [limiter, setLimiter] = useState({
    enabled: state?.limiter_enabled || false,
    threshold_db: -3,
  });

  const [freqOffset, setFreqOffset] = useState({
    enabled: state?.freq_offset_enabled || false,
    offset_hz: 0,
  });

  const handleApplyVogler = () => {
    configureVogler(vogler);
  };

  const handleApplyNoise = () => {
    configureNoise(noise);
  };

  const handleApplyAGC = () => {
    configureAGC(agc);
  };

  const handleApplyLimiter = () => {
    configureLimiter(limiter);
  };

  const handleApplyFreqOffset = () => {
    configureFreqOffset(freqOffset);
  };

  return (
    <div className="card control-tabs">
      <div className="tabs">
        <button
          className={`tab ${activeTab === 'channel' ? 'active' : ''}`}
          onClick={() => setActiveTab('channel')}
        >
          Channel
        </button>
        <button
          className={`tab ${activeTab === 'noise' ? 'active' : ''}`}
          onClick={() => setActiveTab('noise')}
        >
          Noise
        </button>
        <button
          className={`tab ${activeTab === 'impairments' ? 'active' : ''}`}
          onClick={() => setActiveTab('impairments')}
        >
          Impairments
        </button>
      </div>

      <div className="tab-content">
        {activeTab === 'channel' && (
          <div className="control-panel">
            <h4>Channel Model</h4>
            <div className="model-select">
              <button
                className={`model-btn ${state?.model === 'vogler' ? 'active' : ''}`}
                onClick={() => configureVogler({})}
              >
                Vogler IPM
              </button>
              <button
                className={`model-btn ${state?.model === 'watterson' ? 'active' : ''}`}
                onClick={() => configureWatterson('moderate')}
              >
                Watterson
              </button>
            </div>

            {state?.model === 'vogler' && (
              <div className="vogler-controls">
                <div className="control-group">
                  <label>foF2 (MHz)</label>
                  <input
                    type="range"
                    min="1"
                    max="20"
                    step="0.1"
                    value={vogler.foF2}
                    onChange={(e) => setVogler({ ...vogler, foF2: parseFloat(e.target.value) })}
                  />
                  <span className="value mono">{vogler.foF2.toFixed(1)}</span>
                </div>

                <div className="control-group">
                  <label>hmF2 (km)</label>
                  <input
                    type="range"
                    min="150"
                    max="500"
                    step="10"
                    value={vogler.hmF2}
                    onChange={(e) => setVogler({ ...vogler, hmF2: parseFloat(e.target.value) })}
                  />
                  <span className="value mono">{vogler.hmF2.toFixed(0)}</span>
                </div>

                <div className="control-group">
                  <label>Doppler Spread (Hz)</label>
                  <input
                    type="range"
                    min="0.1"
                    max="20"
                    step="0.1"
                    value={vogler.doppler_spread_hz}
                    onChange={(e) => setVogler({ ...vogler, doppler_spread_hz: parseFloat(e.target.value) })}
                  />
                  <span className="value mono">{vogler.doppler_spread_hz.toFixed(1)}</span>
                </div>

                <div className="control-group">
                  <label>Delay Spread (ms)</label>
                  <input
                    type="range"
                    min="0.1"
                    max="10"
                    step="0.1"
                    value={vogler.delay_spread_ms}
                    onChange={(e) => setVogler({ ...vogler, delay_spread_ms: parseFloat(e.target.value) })}
                  />
                  <span className="value mono">{vogler.delay_spread_ms.toFixed(1)}</span>
                </div>

                <button onClick={handleApplyVogler} className="apply-btn">
                  Apply
                </button>
              </div>
            )}

            {state?.model === 'watterson' && (
              <div className="watterson-controls">
                <div className="control-group">
                  <label>ITU Condition</label>
                  <select onChange={(e) => configureWatterson(e.target.value)}>
                    <option value="quiet">Quiet</option>
                    <option value="moderate" selected>Moderate</option>
                    <option value="disturbed">Disturbed</option>
                    <option value="flutter">Flutter</option>
                  </select>
                </div>
              </div>
            )}

            <button onClick={reset} className="secondary reset-btn">
              Reset Channel
            </button>
          </div>
        )}

        {activeTab === 'noise' && (
          <div className="control-panel">
            <h4>Noise Configuration</h4>

            <div className="control-group">
              <label>SNR (dB)</label>
              <input
                type="range"
                min="-10"
                max="40"
                step="1"
                value={noise.snr_db}
                onChange={(e) => setNoise({ ...noise, snr_db: parseFloat(e.target.value) })}
              />
              <span className="value mono">{noise.snr_db}</span>
            </div>

            <div className="control-group checkbox-group">
              <label>
                <input
                  type="checkbox"
                  checked={noise.enable_atmospheric}
                  onChange={(e) => setNoise({ ...noise, enable_atmospheric: e.target.checked })}
                />
                Atmospheric Noise
              </label>
            </div>

            <div className="control-group checkbox-group">
              <label>
                <input
                  type="checkbox"
                  checked={noise.enable_manmade}
                  onChange={(e) => setNoise({ ...noise, enable_manmade: e.target.checked })}
                />
                Man-Made Noise
              </label>
            </div>

            <button onClick={handleApplyNoise} className="apply-btn">
              Apply Noise
            </button>
          </div>
        )}

        {activeTab === 'impairments' && (
          <div className="control-panel">
            <h4>Impairments</h4>

            <div className="impairment-section">
              <div className="impairment-header">
                <label>
                  <input
                    type="checkbox"
                    checked={agc.enabled}
                    onChange={(e) => setAgc({ ...agc, enabled: e.target.checked })}
                  />
                  AGC
                </label>
              </div>
              {agc.enabled && (
                <div className="control-group">
                  <label>Target Level (dBFS)</label>
                  <input
                    type="range"
                    min="-40"
                    max="0"
                    step="1"
                    value={agc.target_level_db}
                    onChange={(e) => setAgc({ ...agc, target_level_db: parseFloat(e.target.value) })}
                  />
                  <span className="value mono">{agc.target_level_db}</span>
                </div>
              )}
              <button onClick={handleApplyAGC} className="apply-btn small">Apply</button>
            </div>

            <div className="impairment-section">
              <div className="impairment-header">
                <label>
                  <input
                    type="checkbox"
                    checked={limiter.enabled}
                    onChange={(e) => setLimiter({ ...limiter, enabled: e.target.checked })}
                  />
                  Limiter
                </label>
              </div>
              {limiter.enabled && (
                <div className="control-group">
                  <label>Threshold (dBFS)</label>
                  <input
                    type="range"
                    min="-20"
                    max="0"
                    step="0.5"
                    value={limiter.threshold_db}
                    onChange={(e) => setLimiter({ ...limiter, threshold_db: parseFloat(e.target.value) })}
                  />
                  <span className="value mono">{limiter.threshold_db}</span>
                </div>
              )}
              <button onClick={handleApplyLimiter} className="apply-btn small">Apply</button>
            </div>

            <div className="impairment-section">
              <div className="impairment-header">
                <label>
                  <input
                    type="checkbox"
                    checked={freqOffset.enabled}
                    onChange={(e) => setFreqOffset({ ...freqOffset, enabled: e.target.checked })}
                  />
                  Frequency Offset
                </label>
              </div>
              {freqOffset.enabled && (
                <div className="control-group">
                  <label>Offset (Hz)</label>
                  <input
                    type="range"
                    min="-1000"
                    max="1000"
                    step="10"
                    value={freqOffset.offset_hz}
                    onChange={(e) => setFreqOffset({ ...freqOffset, offset_hz: parseFloat(e.target.value) })}
                  />
                  <span className="value mono">{freqOffset.offset_hz}</span>
                </div>
              )}
              <button onClick={handleApplyFreqOffset} className="apply-btn small">Apply</button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default ControlTabs;
