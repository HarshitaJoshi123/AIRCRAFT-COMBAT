import { SimulationStatus, CameraMode } from "../types/simulation";

interface ControlsPanelProps {
  status: SimulationStatus | undefined;
  onStart: () => void;
  onPause: () => void;
  onResume: () => void;
  onReset: () => void;
  onStop: () => void;
  cameraMode: CameraMode;
  onCameraModeChange: (mode: CameraMode) => void;
  busy?: boolean;
}

export function ControlsPanel({
  status,
  onStart,
  onPause,
  onResume,
  onReset,
  onStop,
  cameraMode,
  onCameraModeChange,
  busy,
}: ControlsPanelProps) {
  const isActive = status === "ACTIVE";
  const isPaused = status === "PAUSED";

  return (
    <div className="hud-panel rounded-lg p-3 flex flex-col gap-3">
      <div>
        <h3 className="text-xs font-semibold tracking-widest text-aero-accent mb-2">SIMULATION</h3>
        <div className="grid grid-cols-2 gap-2">
          <ControlButton label="Start" onClick={onStart} disabled={busy || isActive || isPaused} variant="primary" />
          {isPaused ? (
            <ControlButton label="Resume" onClick={onResume} disabled={busy} variant="primary" />
          ) : (
            <ControlButton label="Pause" onClick={onPause} disabled={busy || !isActive} />
          )}
          <ControlButton label="Reset" onClick={onReset} disabled={busy || isActive} />
          <ControlButton label="Stop" onClick={onStop} disabled={busy || (!isActive && !isPaused)} variant="danger" />
        </div>
      </div>

      <div>
        <h3 className="text-xs font-semibold tracking-widest text-aero-accent mb-2">CAMERA</h3>
        <div className="grid grid-cols-2 gap-2">
          <CameraButton label="Follow Agent" mode="follow_agent" active={cameraMode} onClick={onCameraModeChange} />
          <CameraButton label="Follow Enemy" mode="follow_enemy" active={cameraMode} onClick={onCameraModeChange} />
          <CameraButton label="Free Camera" mode="free" active={cameraMode} onClick={onCameraModeChange} />
          <CameraButton label="Top View" mode="top" active={cameraMode} onClick={onCameraModeChange} />
        </div>
      </div>
    </div>
  );
}

function ControlButton({
  label,
  onClick,
  disabled,
  variant,
}: {
  label: string;
  onClick: () => void;
  disabled?: boolean;
  variant?: "primary" | "danger";
}) {
  const base = "text-xs font-semibold py-2 rounded-md transition-colors disabled:opacity-30 disabled:cursor-not-allowed";
  const style =
    variant === "primary"
      ? "bg-aero-accent/20 text-aero-accent border border-aero-accent/50 hover:bg-aero-accent/30"
      : variant === "danger"
      ? "bg-aero-danger/20 text-aero-danger border border-aero-danger/50 hover:bg-aero-danger/30"
      : "bg-aero-border/40 text-slate-200 border border-aero-border hover:bg-aero-border/70";
  return (
    <button className={`${base} ${style}`} onClick={onClick} disabled={disabled}>
      {label}
    </button>
  );
}

function CameraButton({
  label,
  mode,
  active,
  onClick,
}: {
  label: string;
  mode: CameraMode;
  active: CameraMode;
  onClick: (mode: CameraMode) => void;
}) {
  const isActive = active === mode;
  return (
    <button
      className={`text-xs font-medium py-2 rounded-md border transition-colors ${
        isActive
          ? "bg-aero-accent/20 border-aero-accent text-aero-accent"
          : "bg-aero-border/20 border-aero-border text-slate-300 hover:bg-aero-border/40"
      }`}
      onClick={() => onClick(mode)}
    >
      {label}
    </button>
  );
}
