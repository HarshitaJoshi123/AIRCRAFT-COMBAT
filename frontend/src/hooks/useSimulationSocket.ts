import { useEffect, useRef, useState, useCallback } from "react";
import { ManagedSocket, WSStatus } from "../services/websocket";
import {
  SimulationSocketMessage,
  SimulationStateMessage,
  SimulationEventMessage,
} from "../types/simulation";

const MAX_EVENTS = 60;

export function useSimulationSocket() {
  const [connectionStatus, setConnectionStatus] = useState<WSStatus>("connecting");
  const [latestState, setLatestState] = useState<SimulationStateMessage | null>(null);
  const [events, setEvents] = useState<SimulationEventMessage[]>([]);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const socketRef = useRef<ManagedSocket<SimulationSocketMessage> | null>(null);

  useEffect(() => {
    const socket = new ManagedSocket<SimulationSocketMessage>({
      path: "/ws/simulation",
      onStatusChange: setConnectionStatus,
      onMessage: (msg) => {
        if (msg.type === "state") {
          setLatestState(msg);
          setErrorMessage(null);
        } else if (msg.type === "event") {
          setEvents((prev) => [msg, ...prev].slice(0, MAX_EVENTS));
        } else if (msg.type === "error") {
          setErrorMessage(msg.message);
        }
        // "status" snapshots are informational only; state updates arrive
        // via "state" messages once the sim loop starts ticking.
      },
    });
    socket.connect();
    socketRef.current = socket;
    return () => socket.close();
  }, []);

  const clearEvents = useCallback(() => setEvents([]), []);

  return { connectionStatus, latestState, events, errorMessage, clearEvents };
}
