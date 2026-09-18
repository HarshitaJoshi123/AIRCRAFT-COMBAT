import { SimulationEventMessage } from "../types/simulation";

const LEVEL_COLOR: Record<SimulationEventMessage["level"], string> = {
  info: "text-slate-300",
  success: "text-aero-success",
  warning: "text-aero-warning",
  critical: "text-aero-danger",
};

export function EventLog({ events }: { events: SimulationEventMessage[] }) {
  return (
    <div className="hud-panel rounded-lg p-3 flex flex-col h-full min-h-0">
      <h3 className="text-xs font-semibold tracking-widest text-aero-accent mb-2 shrink-0">COMBAT EVENT LOG</h3>
      <div className="flex flex-col-reverse gap-1 overflow-y-auto pr-1 flex-1 min-h-0">
        {events.length === 0 && (
          <div className="text-xs text-slate-500 font-mono">No events yet. Start the simulation.</div>
        )}
        {[...events].reverse().map((e, i) => (
          <div key={`${e.step}-${i}-${e.message}`} className="font-mono text-xs flex gap-2">
            <span className="text-slate-500 shrink-0">[{String(e.step).padStart(3, "0")}]</span>
            <span className={LEVEL_COLOR[e.level]}>{e.message}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
