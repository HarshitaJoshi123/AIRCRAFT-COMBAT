import { useEffect, useRef, useState, useCallback } from "react";
import { ManagedSocket, WSStatus } from "../services/websocket";
import { EvaluationResultMessage, EvaluationSocketMessage } from "../types/simulation";

export function useEvaluationSocket() {
  const [connectionStatus, setConnectionStatus] = useState<WSStatus>("connecting");
  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState({ completed: 0, total: 0, wins: 0, losses: 0 });
  const [result, setResult] = useState<EvaluationResultMessage | null>(null);
  const [error, setError] = useState<string | null>(null);
  const socketRef = useRef<ManagedSocket<EvaluationSocketMessage> | null>(null);

  useEffect(() => {
    const socket = new ManagedSocket<EvaluationSocketMessage>({
      path: "/ws/evaluation",
      onStatusChange: setConnectionStatus,
      onMessage: (msg) => {
        if (msg.type === "progress") {
          setRunning(true);
          setProgress({ completed: msg.completed, total: msg.total, wins: msg.wins, losses: msg.losses });
        } else if (msg.type === "result") {
          setRunning(false);
          setResult(msg);
        } else if (msg.type === "status") {
          setRunning(msg.running);
          if (msg.total > 0) {
            setProgress({ completed: msg.completed, total: msg.total, wins: msg.wins, losses: msg.losses });
          }
        } else if (msg.type === "error") {
          setRunning(false);
          setError(msg.message);
        }
      },
    });
    socket.connect();
    socketRef.current = socket;
    return () => socket.close();
  }, []);

  const runEvaluation = useCallback((episodes: number) => {
    setResult(null);
    setError(null);
    setProgress({ completed: 0, total: episodes, wins: 0, losses: 0 });
    socketRef.current?.send({ action: "run", episodes });
  }, []);

  return { connectionStatus, running, progress, result, error, runEvaluation };
}
