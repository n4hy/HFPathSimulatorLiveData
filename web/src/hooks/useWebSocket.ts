/**
 * React hooks for WebSocket streams
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import {
  createStateStream,
  createSpectrumStream,
  StateMessage,
  SpectrumMessage,
  WebSocketClient,
} from '../api/websocket';

export function useStateStream(intervalMs = 100) {
  const [state, setState] = useState<StateMessage | null>(null);
  const [connected, setConnected] = useState(false);
  const clientRef = useRef<WebSocketClient | null>(null);

  useEffect(() => {
    const client = createStateStream(
      (msg) => setState(msg),
      (status) => setConnected(status),
      intervalMs
    );

    clientRef.current = client;
    client.connect();

    return () => {
      client.disconnect();
    };
  }, [intervalMs]);

  return { state, connected };
}

export function useSpectrumStream(intervalMs = 100) {
  const [spectrum, setSpectrum] = useState<SpectrumMessage | null>(null);
  const [connected, setConnected] = useState(false);
  const clientRef = useRef<WebSocketClient | null>(null);

  useEffect(() => {
    const client = createSpectrumStream(
      (msg) => setSpectrum(msg),
      (status) => setConnected(status),
      intervalMs
    );

    clientRef.current = client;
    client.connect();

    return () => {
      client.disconnect();
    };
  }, [intervalMs]);

  return { spectrum, connected };
}
