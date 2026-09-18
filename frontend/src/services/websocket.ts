const WS_URL = import.meta.env.VITE_WS_URL || "ws://localhost:8000";

export type WSStatus = "connecting" | "open" | "closed" | "error";

export interface ManagedSocketOptions<T> {
  path: string;
  onMessage: (data: T) => void;
  onStatusChange?: (status: WSStatus) => void;
  reconnectDelayMs?: number;
}

/**
 * A small reconnecting WebSocket wrapper. The backend is the single source
 * of truth for simulation/evaluation state; this client never invents or
 * interpolates values beyond what the server sent (interpolation of
 * *positions between ticks* happens purely visually in the 3D scene, see
 * simulation/interpolation.ts).
 */
export class ManagedSocket<T> {
  private socket: WebSocket | null = null;
  private shouldReconnect = true;
  private reconnectDelayMs: number;

  constructor(private options: ManagedSocketOptions<T>) {
    this.reconnectDelayMs = options.reconnectDelayMs ?? 2000;
  }

  connect() {
    this.shouldReconnect = true;
    this.open();
  }

  private open() {
    this.options.onStatusChange?.("connecting");
    const socket = new WebSocket(`${WS_URL}${this.options.path}`);
    this.socket = socket;

    socket.onopen = () => this.options.onStatusChange?.("open");

    socket.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data) as T;
        this.options.onMessage(data);
      } catch (err) {
        console.error("Failed to parse WebSocket message", err);
      }
    };

    socket.onerror = () => this.options.onStatusChange?.("error");

    socket.onclose = () => {
      this.options.onStatusChange?.("closed");
      if (this.shouldReconnect) {
        setTimeout(() => this.open(), this.reconnectDelayMs);
      }
    };
  }

  send(data: unknown) {
    if (this.socket?.readyState === WebSocket.OPEN) {
      this.socket.send(JSON.stringify(data));
    }
  }

  close() {
    this.shouldReconnect = false;
    this.socket?.close();
  }
}

export { WS_URL };
