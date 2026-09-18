import { SimulationStateMessage } from "../types/simulation";

interface ResultOverlayProps {
  state: SimulationStateMessage;
  onRunAgain: () => void;
}

const RESULT_CONFIG: Record<string, { label: string; color: string; glow: string }> = {
  VICTORY: { label: "VICTORY", color: "text-aero-success", glow: "shadow-[0_0_60px_rgba(34,197,94,0.35)]" },
  DEFEAT: { label: "DEFEAT", color: "text-aero-danger", glow: "shadow-[0_0_60px_rgba(239,68,68,0.35)]" },
  TIMEOUT: { label: "TIMEOUT", color: "text-aero-warning", glow: "shadow-[0_0_60px_rgba(245,158,11,0.35)]" },
};

export function ResultOverlay({ state, onRunAgain }: ResultOverlayProps) {
  const config = RESULT_CONFIG[state.status];
  if (!config) return null;

  const distance = Math.sqrt(
    (state.agent.position[0] - state.enemy.position[0]) ** 2 +
      (state.agent.position[1] - state.enemy.position[1]) ** 2 +
      (state.agent.position[2] - state.enemy.position[2]) ** 2
  );

  return (
    <div className="absolute inset-0 flex items-center justify-center bg-black/60 backdrop-blur-sm z-20">
      <div className={`hud-panel rounded-xl px-10 py-8 flex flex-col items-center gap-4 ${config.glow}`}>
        <h2 className={`text-5xl font-black tracking-widest ${config.color} glow-cyan`}>{config.label}</h2>

        <div className="grid grid-cols-2 gap-x-8 gap-y-2 font-mono text-sm mt-2">
          <Stat label="Total Steps" value={state.step} />
          <Stat label="Total Reward" value={state.episode_reward.toFixed(2)} />
          <Stat label="Final Distance" value={distance.toFixed(1)} />
          <Stat label="Final Status" value={state.status} />
          <Stat label="Agent Hit" value={state.agent_hit ? "YES" : "no"} />
          <Stat label="Enemy Hit" value={state.enemy_hit ? "YES" : "no"} />
        </div>

        <button
          onClick={onRunAgain}
          className="mt-4 px-6 py-2 rounded-md bg-aero-accent/20 border border-aero-accent text-aero-accent font-semibold text-sm hover:bg-aero-accent/30 transition-colors"
        >
          RUN AGAIN
        </button>
      </div>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="flex justify-between gap-6">
      <span className="text-slate-400">{label}</span>
      <span className="text-slate-100">{value}</span>
    </div>
  );
}
