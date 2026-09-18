import { useCallback, useState } from "react";
import { CombatScene } from "../components/CombatScene";
import { TelemetryPanel } from "../components/TelemetryPanel";
import { ControlsPanel } from "../components/ControlsPanel";
import { EventLog } from "../components/EventLog";
import { ResultOverlay } from "../components/ResultOverlay";
import { useSimulationSocket } from "../hooks/useSimulationSocket";
import { api } from "../services/api";
import { CameraMode } from "../types/simulation";

const TERMINAL_STATUSES = new Set(["VICTORY", "DEFEAT", "TIMEOUT"]);

type SimSocket = ReturnType<typeof useSimulationSocket>;

export function DashboardPage({ sim }: { sim: SimSocket }) {
  const { latestState, events, errorMessage } = sim;
  const [cameraMode, setCameraMode] = useState<CameraMode>("follow_agent");
  const [busy, setBusy] = useState(false);
  const [apiError, setApiError] = useState<string | null>(null);

  const runAction = useCallback(async (fn: () => Promise<unknown>) => {
    setBusy(true);
    setApiError(null);
    try {
      await fn();
    } catch (err) {
      setApiError(err instanceof Error ? err.message : "Request failed");
    } finally {
      setBusy(false);
    }
  }, []);

  const isTerminal = latestState ? TERMINAL_STATUSES.has(latestState.status) : false;

  return (
    <div className="grid grid-cols-1 lg:grid-cols-[1fr_320px] gap-3 p-3 h-full min-h-0">
      <div className="relative rounded-lg overflow-hidden border border-aero-border min-h-[420px]">
        <CombatScene state={latestState} cameraMode={cameraMode} />

        {latestState && isTerminal && (
          <ResultOverlay state={latestState} onRunAgain={() => runAction(api.resetSimulation).then(() => runAction(api.startSimulation))} />
        )}

        {(errorMessage || apiError) && (
          <div className="absolute top-3 left-3 right-3 bg-aero-danger/20 border border-aero-danger/60 text-aero-danger text-xs font-mono px-3 py-2 rounded-md">
            {errorMessage || apiError}
          </div>
        )}

        <div className="absolute bottom-3 left-3 hud-panel rounded-md px-3 py-1.5 text-xs font-mono text-slate-300">
          Status: <span className="text-aero-accent">{latestState?.status ?? "IDLE"}</span>
        </div>
      </div>

      <div className="flex flex-col gap-3 min-h-0 overflow-y-auto pr-1">
        <ControlsPanel
          status={latestState?.status}
          busy={busy}
          onStart={() => runAction(api.startSimulation)}
          onPause={() => runAction(api.pauseSimulation)}
          onResume={() => runAction(api.resumeSimulation)}
          onReset={() => runAction(api.resetSimulation)}
          onStop={() => runAction(api.stopSimulation)}
          cameraMode={cameraMode}
          onCameraModeChange={setCameraMode}
        />
        <TelemetryPanel state={latestState} />
        <div className="flex-1 min-h-[180px]">
          <EventLog events={events} />
        </div>
      </div>
    </div>
  );
}
