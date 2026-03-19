/**
 * HF Path Simulator API Client
 */

const API_BASE = import.meta.env.VITE_API_URL || '/api/v1';

export interface ChannelState {
  model: 'vogler' | 'watterson' | 'vogler_hoffmeyer' | 'passthrough';
  running: boolean;
  total_samples_processed: number;
  blocks_processed: number;
  vogler?: VoglerConfig;
  agc_enabled: boolean;
  agc_gain_db: number;
  limiter_enabled: boolean;
  limiter_reduction_db: number;
  freq_offset_enabled: boolean;
  current_freq_offset_hz: number;
}

export interface VoglerConfig {
  foF2?: number;
  hmF2?: number;
  foE?: number;
  hmE?: number;
  doppler_spread_hz?: number;
  delay_spread_ms?: number;
  frequency_mhz?: number;
  path_length_km?: number;
}

export interface GPUInfo {
  available: boolean;
  name?: string;
  compute_capability?: string;
  total_memory_gb?: number;
  multiprocessors?: number;
  backend?: string;
}

export interface HealthStatus {
  status: string;
  version: string;
  uptime_seconds: number;
}

class ApiClient {
  private baseUrl: string;

  constructor(baseUrl: string = API_BASE) {
    this.baseUrl = baseUrl;
  }

  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    const url = `${this.baseUrl}${endpoint}`;
    const response = await fetch(url, {
      ...options,
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.detail || `HTTP ${response.status}`);
    }

    return response.json();
  }

  // Health endpoints
  async getHealth(): Promise<HealthStatus> {
    return this.request('/health');
  }

  async getGPUInfo(): Promise<GPUInfo> {
    return this.request('/gpu');
  }

  // Channel endpoints
  async getChannelState(): Promise<ChannelState> {
    return this.request('/channel/state');
  }

  async configureVogler(config: VoglerConfig): Promise<ChannelState> {
    return this.request('/channel/vogler', {
      method: 'POST',
      body: JSON.stringify(config),
    });
  }

  async configureWatterson(condition: string): Promise<ChannelState> {
    return this.request('/channel/watterson', {
      method: 'POST',
      body: JSON.stringify({ condition }),
    });
  }

  async configureVH(
    condition: string,
    spreadFEnabled?: boolean
  ): Promise<ChannelState> {
    return this.request('/channel/vh', {
      method: 'POST',
      body: JSON.stringify({
        condition,
        spread_f_enabled: spreadFEnabled,
      }),
    });
  }

  async configureNoise(config: {
    snr_db?: number;
    enable_atmospheric?: boolean;
    enable_manmade?: boolean;
    enable_impulse?: boolean;
  }): Promise<ChannelState> {
    return this.request('/channel/noise', {
      method: 'POST',
      body: JSON.stringify(config),
    });
  }

  async disableNoise(): Promise<ChannelState> {
    return this.request('/channel/noise/disable', { method: 'POST' });
  }

  async configureAGC(config: {
    enabled: boolean;
    target_level_db?: number;
    max_gain_db?: number;
    min_gain_db?: number;
  }): Promise<ChannelState> {
    return this.request('/channel/impairments/agc', {
      method: 'POST',
      body: JSON.stringify(config),
    });
  }

  async configureLimiter(config: {
    enabled: boolean;
    threshold_db?: number;
  }): Promise<ChannelState> {
    return this.request('/channel/impairments/limiter', {
      method: 'POST',
      body: JSON.stringify(config),
    });
  }

  async configureFreqOffset(config: {
    enabled: boolean;
    offset_hz?: number;
    drift_hz_per_sec?: number;
  }): Promise<ChannelState> {
    return this.request('/channel/impairments/offset', {
      method: 'POST',
      body: JSON.stringify(config),
    });
  }

  async resetChannel(): Promise<ChannelState> {
    return this.request('/channel/reset', { method: 'POST' });
  }

  // Processing endpoints
  async startProcessing(): Promise<void> {
    return this.request('/processing/start', { method: 'POST' });
  }

  async stopProcessing(): Promise<void> {
    return this.request('/processing/stop', { method: 'POST' });
  }
}

export const apiClient = new ApiClient();
export default apiClient;
