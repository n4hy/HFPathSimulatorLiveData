/**
 * React hook for channel state management
 */

import { useState, useEffect, useCallback } from 'react';
import apiClient, { ChannelState, VoglerConfig } from '../api/client';

export function useChannelState() {
  const [state, setState] = useState<ChannelState | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      setError(null);
      const newState = await apiClient.getChannelState();
      setState(newState);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refresh();
    // Poll every 500ms when not using WebSocket
    const interval = setInterval(refresh, 500);
    return () => clearInterval(interval);
  }, [refresh]);

  const configureVogler = useCallback(async (config: VoglerConfig) => {
    try {
      setError(null);
      const newState = await apiClient.configureVogler(config);
      setState(newState);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Unknown error');
    }
  }, []);

  const configureWatterson = useCallback(async (condition: string) => {
    try {
      setError(null);
      const newState = await apiClient.configureWatterson(condition);
      setState(newState);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Unknown error');
    }
  }, []);

  const configureVH = useCallback(async (condition: string, spreadF?: boolean) => {
    try {
      setError(null);
      const newState = await apiClient.configureVH(condition, spreadF);
      setState(newState);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Unknown error');
    }
  }, []);

  const configureNoise = useCallback(async (config: {
    snr_db?: number;
    enable_atmospheric?: boolean;
    enable_manmade?: boolean;
    enable_impulse?: boolean;
  }) => {
    try {
      setError(null);
      const newState = await apiClient.configureNoise(config);
      setState(newState);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Unknown error');
    }
  }, []);

  const configureAGC = useCallback(async (config: {
    enabled: boolean;
    target_level_db?: number;
    max_gain_db?: number;
  }) => {
    try {
      setError(null);
      const newState = await apiClient.configureAGC(config);
      setState(newState);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Unknown error');
    }
  }, []);

  const configureLimiter = useCallback(async (config: {
    enabled: boolean;
    threshold_db?: number;
  }) => {
    try {
      setError(null);
      const newState = await apiClient.configureLimiter(config);
      setState(newState);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Unknown error');
    }
  }, []);

  const configureFreqOffset = useCallback(async (config: {
    enabled: boolean;
    offset_hz?: number;
    drift_hz_per_sec?: number;
  }) => {
    try {
      setError(null);
      const newState = await apiClient.configureFreqOffset(config);
      setState(newState);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Unknown error');
    }
  }, []);

  const reset = useCallback(async () => {
    try {
      setError(null);
      const newState = await apiClient.resetChannel();
      setState(newState);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Unknown error');
    }
  }, []);

  return {
    state,
    loading,
    error,
    refresh,
    configureVogler,
    configureWatterson,
    configureVH,
    configureNoise,
    configureAGC,
    configureLimiter,
    configureFreqOffset,
    reset,
  };
}
