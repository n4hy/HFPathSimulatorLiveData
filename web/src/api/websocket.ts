/**
 * WebSocket client for real-time streaming
 */

const WS_BASE = import.meta.env.VITE_WS_URL ||
  `ws://${window.location.host}/api/v1/stream`;

export interface StateMessage {
  type: 'state';
  timestamp: number;
  running: boolean;
  blocks_processed: number;
  total_samples_processed: number;
  agc_gain_db: number;
  limiter_reduction_db: number;
  current_freq_offset_hz: number;
}

export interface SpectrumMessage {
  type: 'spectrum';
  timestamp: number;
  spectrum_db: number[];
  freq_axis_hz: number[];
  fft_size: number;
}

export interface KeepaliveMessage {
  type: 'keepalive';
  timestamp: number;
}

export type WSMessage = StateMessage | SpectrumMessage | KeepaliveMessage;

export class WebSocketClient {
  private ws: WebSocket | null = null;
  private url: string;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectDelay = 1000;
  private onMessage: (msg: WSMessage) => void;
  private onStatusChange: (connected: boolean) => void;

  constructor(
    endpoint: string,
    onMessage: (msg: WSMessage) => void,
    onStatusChange: (connected: boolean) => void = () => {}
  ) {
    this.url = `${WS_BASE}/${endpoint}`;
    this.onMessage = onMessage;
    this.onStatusChange = onStatusChange;
  }

  connect(): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      return;
    }

    this.ws = new WebSocket(this.url);

    this.ws.onopen = () => {
      console.log(`WebSocket connected: ${this.url}`);
      this.reconnectAttempts = 0;
      this.onStatusChange(true);
    };

    this.ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data) as WSMessage;
        this.onMessage(msg);
      } catch (e) {
        console.error('Failed to parse WebSocket message:', e);
      }
    };

    this.ws.onclose = () => {
      console.log('WebSocket disconnected');
      this.onStatusChange(false);
      this.attemptReconnect();
    };

    this.ws.onerror = (error) => {
      console.error('WebSocket error:', error);
    };
  }

  disconnect(): void {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
  }

  send(data: object): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(data));
    }
  }

  private attemptReconnect(): void {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      console.log('Max reconnect attempts reached');
      return;
    }

    this.reconnectAttempts++;
    const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1);

    console.log(`Reconnecting in ${delay}ms (attempt ${this.reconnectAttempts})`);
    setTimeout(() => this.connect(), delay);
  }

  get isConnected(): boolean {
    return this.ws?.readyState === WebSocket.OPEN;
  }
}

// Factory functions for specific streams
export function createStateStream(
  onMessage: (msg: StateMessage) => void,
  onStatusChange?: (connected: boolean) => void,
  intervalMs = 100
): WebSocketClient {
  return new WebSocketClient(
    `state?interval_ms=${intervalMs}`,
    (msg) => {
      if (msg.type === 'state') {
        onMessage(msg as StateMessage);
      }
    },
    onStatusChange
  );
}

export function createSpectrumStream(
  onMessage: (msg: SpectrumMessage) => void,
  onStatusChange?: (connected: boolean) => void,
  intervalMs = 100
): WebSocketClient {
  return new WebSocketClient(
    `spectrum?interval_ms=${intervalMs}`,
    (msg) => {
      if (msg.type === 'spectrum') {
        onMessage(msg as SpectrumMessage);
      }
    },
    onStatusChange
  );
}
