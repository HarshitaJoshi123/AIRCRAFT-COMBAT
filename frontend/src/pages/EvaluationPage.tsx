import { useMemo, useState } from "react";
import {
  BarChart,
  Bar,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from "recharts";
import { useEvaluationSocket } from "../hooks/useEvaluationSocket";

const EPISODE_PRESETS = [50, 100, 300, 500];

export function EvaluationPage() {
  const { running, progress, result, error, runEvaluation } = useEvaluationSocket();
  const [episodes, setEpisodes] = useState(300);

  const rewardHistogram = useMemo(() => {
    if (!result) return [];
    const buckets = 12;
    const min = result.min_reward;
    const max = result.max_reward;
    const width = (max - min) / buckets || 1;
    const counts = Array.from({ length: buckets }, (_, i) => ({
      bucket: `${(min + i * width).toFixed(0)}`,
      count: 0,
    }));
    result.episode_rewards.forEach((r) => {
      const idx = Math.min(buckets - 1, Math.max(0, Math.floor((r - min) / width)));
      counts[idx].count += 1;
    });
    return counts;
  }, [result]);

  const rewardSeries = useMemo(() => {
    if (!result) return [];
    return result.episode_rewards.map((r, i) => ({ episode: i + 1, reward: r }));
  }, [result]);

  const winLossData = result
    ? [
        { name: "Wins", value: result.wins, fill: "#22c55e" },
        { name: "Losses", value: result.losses, fill: "#ef4444" },
      ]
    : [];

  return (
    <div className="p-6 max-w-6xl mx-auto overflow-y-auto h-full flex flex-col gap-4">
      <div>
        <h2 className="text-2xl font-bold text-aero-accent mb-1">Model Evaluation</h2>
        <p className="text-slate-400 text-sm">
          Runs the actual trained PPO model against fresh episodes of <code>AircraftCombatEnv</code>.
          Numbers below are real, computed on demand &mdash; nothing is precomputed or faked.
        </p>
      </div>

      <div className="hud-panel rounded-lg p-4 flex flex-wrap items-center gap-4">
        <label className="text-xs text-slate-400 font-mono">Episodes</label>
        <input
          type="number"
          min={1}
          max={5000}
          value={episodes}
          onChange={(e) => setEpisodes(Math.max(1, Math.min(5000, Number(e.target.value) || 1)))}
          className="w-24 bg-aero-bg border border-aero-border rounded-md px-2 py-1 text-sm font-mono text-slate-100"
        />
        <div className="flex gap-1">
          {EPISODE_PRESETS.map((p) => (
            <button
              key={p}
              onClick={() => setEpisodes(p)}
              className="text-xs px-2 py-1 rounded-md border border-aero-border text-slate-300 hover:bg-aero-border/40"
            >
              {p}
            </button>
          ))}
        </div>
        <button
          onClick={() => runEvaluation(episodes)}
          disabled={running}
          className="ml-auto px-5 py-2 rounded-md bg-aero-accent/20 border border-aero-accent text-aero-accent font-semibold text-sm hover:bg-aero-accent/30 disabled:opacity-40 disabled:cursor-not-allowed"
        >
          {running ? "RUNNING\u2026" : "RUN EVALUATION"}
        </button>
      </div>

      {error && (
        <div className="bg-aero-danger/20 border border-aero-danger/60 text-aero-danger text-xs font-mono px-3 py-2 rounded-md">
          {error}
        </div>
      )}

      {running && (
        <div className="hud-panel rounded-lg p-4">
          <div className="flex justify-between text-xs font-mono text-slate-400 mb-2">
            <span>
              Episode {progress.completed} / {progress.total}
            </span>
            <span>
              Wins: <span className="text-aero-success">{progress.wins}</span> &middot; Losses:{" "}
              <span className="text-aero-danger">{progress.losses}</span>
            </span>
          </div>
          <div className="h-2 w-full rounded-full bg-aero-border overflow-hidden">
            <div
              className="h-full bg-aero-accent transition-all"
              style={{ width: `${progress.total ? (progress.completed / progress.total) * 100 : 0}%` }}
            />
          </div>
        </div>
      )}

      {result && (
        <>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            <StatCard label="Win Rate" value={`${result.win_rate.toFixed(1)}%`} tone="success" />
            <StatCard label="Loss Rate" value={`${result.loss_rate.toFixed(1)}%`} tone="danger" />
            <StatCard label="Avg Reward" value={result.avg_reward.toFixed(2)} />
            <StatCard label="Max / Min Reward" value={`${result.max_reward.toFixed(1)} / ${result.min_reward.toFixed(1)}`} />
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <ChartCard title="Win / Loss Distribution">
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={winLossData}>
                  <CartesianGrid stroke="#1e2a3d" strokeDasharray="3 3" />
                  <XAxis dataKey="name" stroke="#64748b" fontSize={12} />
                  <YAxis stroke="#64748b" fontSize={12} />
                  <Tooltip contentStyle={{ background: "#111826", border: "1px solid #1e2a3d" }} />
                  <Bar dataKey="value">
                    {winLossData.map((entry, i) => (
                      <Cell key={i} fill={entry.fill} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </ChartCard>

            <ChartCard title="Reward Distribution">
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={rewardHistogram}>
                  <CartesianGrid stroke="#1e2a3d" strokeDasharray="3 3" />
                  <XAxis dataKey="bucket" stroke="#64748b" fontSize={10} />
                  <YAxis stroke="#64748b" fontSize={12} />
                  <Tooltip contentStyle={{ background: "#111826", border: "1px solid #1e2a3d" }} />
                  <Bar dataKey="count" fill="#22d3ee" />
                </BarChart>
              </ResponsiveContainer>
            </ChartCard>

            <ChartCard title="Episode Rewards" className="md:col-span-2">
              <ResponsiveContainer width="100%" height={240}>
                <LineChart data={rewardSeries}>
                  <CartesianGrid stroke="#1e2a3d" strokeDasharray="3 3" />
                  <XAxis dataKey="episode" stroke="#64748b" fontSize={11} />
                  <YAxis stroke="#64748b" fontSize={11} />
                  <Tooltip contentStyle={{ background: "#111826", border: "1px solid #1e2a3d" }} />
                  <Line type="monotone" dataKey="reward" stroke="#22d3ee" dot={false} strokeWidth={1.5} />
                </LineChart>
              </ResponsiveContainer>
            </ChartCard>
          </div>
        </>
      )}
    </div>
  );
}

function StatCard({ label, value, tone }: { label: string; value: string; tone?: "success" | "danger" }) {
  const toneClass = tone === "success" ? "text-aero-success" : tone === "danger" ? "text-aero-danger" : "text-aero-accent";
  return (
    <div className="hud-panel rounded-lg p-4 flex flex-col gap-1">
      <span className="text-xs text-slate-400 font-mono">{label}</span>
      <span className={`text-2xl font-bold ${toneClass}`}>{value}</span>
    </div>
  );
}

function ChartCard({ title, children, className = "" }: { title: string; children: React.ReactNode; className?: string }) {
  return (
    <div className={`hud-panel rounded-lg p-4 ${className}`}>
      <h3 className="text-xs font-semibold tracking-widest text-aero-accent mb-2">{title.toUpperCase()}</h3>
      {children}
    </div>
  );
}
