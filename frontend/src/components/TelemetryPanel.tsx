import { SimulationStateMessage } from "../types/simulation";

function distanceBetween(a: [number, number, number], b: [number, number, number]) {
  return Math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2);
}

function speed(v: [number, number, number]) {
  return Math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2);
}

export function TelemetryPanel({ state }: { state: SimulationStateMessage | null }) {
  const step = state?.step ?? 0;
  const maxSteps = state?.max_steps ?? 250;
  const distance = state?.distance ?? (state ? distanceBetween(state.agent.position, state.enemy.position) : 0);
  const agentSpeed = state ? speed(state.agent.velocity) : 0;
  const enemySpeed = state ? speed(state.enemy.velocity) : 0;

  return (
    <div className="flex flex-col gap-3 text-sm">
      <Section title="COMBAT STATUS">
        <Row label="Agent" value={state?.agent_hit ? "HIT" : "ACTIVE"} tone={state?.agent_hit ? "danger" : "success"} />
        <Row label="Enemy" value={state?.enemy_hit ? "HIT" : "ACTIVE"} tone={state?.enemy_hit ? "success" : "danger"} />
        <Row label="Distance" value={distance.toFixed(1)} />
        <Row label="Agent Speed" value={agentSpeed.toFixed(2)} />
        <Row label="Enemy Speed" value={enemySpeed.toFixed(2)} />
        <Row label="Step" value={`${step} / ${maxSteps}`} />
      </Section>

      <Section title="MISSILES">
        <Row label="Agent Missile" value={state?.missile.active ? "FIRED" : "READY"} tone={state?.missile.active ? "warning" : "success"} />
        <Row label="Enemy Missile" value={state?.enemy_missile.active ? "FIRED" : "READY"} tone={state?.enemy_missile.active ? "warning" : "success"} />
      </Section>

      <Section title="REWARD">
        <Row label="Current Reward" value={(state?.reward ?? 0).toFixed(2)} />
        <Row label="Episode Reward" value={(state?.episode_reward ?? 0).toFixed(2)} />
      </Section>

      <Section title="PPO ACTION">
        <ActionBar label="Throttle" value={state?.action.throttle ?? 0} range={[0, 1]} />
        <ActionBar label="Pitch" value={state?.action.pitch ?? 0} range={[-1, 1]} />
        <ActionBar label="Yaw" value={state?.action.yaw ?? 0} range={[-1, 1]} />
        <Row label="Fire" value={state?.action.fire ? "TRUE" : "false"} tone={state?.action.fire ? "warning" : undefined} />
      </Section>
    </div>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="hud-panel rounded-lg p-3">
      <h3 className="text-xs font-semibold tracking-widest text-aero-accent mb-2">{title}</h3>
      <div className="flex flex-col gap-1.5">{children}</div>
    </div>
  );
}

function Row({ label, value, tone }: { label: string; value: string | number; tone?: "success" | "danger" | "warning" }) {
  const toneClass =
    tone === "success" ? "text-aero-success" : tone === "danger" ? "text-aero-danger" : tone === "warning" ? "text-aero-warning" : "text-slate-200";
  return (
    <div className="flex justify-between font-mono text-xs">
      <span className="text-slate-400">{label}</span>
      <span className={toneClass}>{value}</span>
    </div>
  );
}

function ActionBar({ label, value, range }: { label: string; value: number; range: [number, number] }) {
  const [min, max] = range;
  const pct = ((value - min) / (max - min)) * 100;
  return (
    <div className="flex flex-col gap-0.5">
      <div className="flex justify-between font-mono text-xs">
        <span className="text-slate-400">{label}</span>
        <span className="text-slate-200">{value.toFixed(2)}</span>
      </div>
      <div className="h-1.5 w-full rounded-full bg-aero-border overflow-hidden">
        <div className="h-full bg-aero-accent" style={{ width: `${Math.min(100, Math.max(0, pct))}%` }} />
      </div>
    </div>
  );
}
