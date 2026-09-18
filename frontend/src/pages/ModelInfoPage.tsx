export function ModelInfoPage() {
  return (
    <div className="p-6 max-w-4xl mx-auto overflow-y-auto h-full">
      <h2 className="text-2xl font-bold text-aero-accent mb-1">Model Information</h2>
      <p className="text-slate-400 text-sm mb-6">
        Specifications below are taken directly from the training notebook and the saved model artifact
        (<code className="text-slate-300">ppo_aircraft_model.zip</code>) - nothing here is invented.
      </p>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <InfoCard title="AI Controller">
          <p className="text-sm text-slate-300">
            <span className="text-aero-accent font-semibold">PPO</span> &mdash; Proximal Policy Optimization,
            from Stable-Baselines3, using an <code>MlpPolicy</code> actor-critic network.
          </p>
        </InfoCard>

        <InfoCard title="Environment">
          <p className="text-sm text-slate-300">
            <code>AircraftCombatEnv</code> &mdash; a custom Gymnasium environment simulating a 1v1 dogfight
            inside a 100&times;100&times;100 bounded arena.
          </p>
        </InfoCard>

        <InfoCard title="Observation Space">
          <p className="text-sm text-slate-300 mb-2">13 continuous features:</p>
          <ul className="text-xs font-mono text-slate-400 space-y-1 list-disc list-inside">
            <li>agent position (3)</li>
            <li>agent velocity (3)</li>
            <li>enemy position (3)</li>
            <li>enemy velocity (3)</li>
            <li>distance to enemy (1)</li>
          </ul>
        </InfoCard>

        <InfoCard title="Action Space">
          <p className="text-sm text-slate-300 mb-2">4 continuous actions, each in [-1, 1]:</p>
          <ul className="text-xs font-mono text-slate-400 space-y-1 list-disc list-inside">
            <li>throttle &mdash; mapped to [0, 1]</li>
            <li>pitch &mdash; rotation delta</li>
            <li>yaw &mdash; rotation delta</li>
            <li>fire &mdash; mapped to [0, 1], fires when &gt; 0.5</li>
          </ul>
        </InfoCard>

        <InfoCard title="Training Configuration">
          <ul className="text-xs font-mono text-slate-400 space-y-1">
            <li>Algorithm: PPO (MlpPolicy)</li>
            <li>Total timesteps: 500,000 (notebook default)</li>
            <li>n_steps: 2048, batch_size: 64 (SB3 PPO defaults, unmodified)</li>
            <li>Saved model timesteps: 501,760</li>
            <li>Framework: stable-baselines3 2.9.0 / gymnasium 1.3.0 / torch 2.11.0</li>
          </ul>
        </InfoCard>

        <InfoCard title="Episode Termination">
          <ul className="text-xs font-mono text-slate-400 space-y-1">
            <li>Terminated: agent missile hits enemy (+100 reward, VICTORY)</li>
            <li>Terminated: enemy missile hits agent (-100 reward, DEFEAT)</li>
            <li>Truncated: 250 steps reached (TIMEOUT)</li>
          </ul>
        </InfoCard>
      </div>

      <InfoCard title="How PPO Interacts With the Environment" className="mt-4">
        <p className="text-sm text-slate-300 leading-relaxed">
          On every simulation tick, the backend reads the current 13-dimensional observation from{" "}
          <code>AircraftCombatEnv</code> and passes it to the loaded PPO policy via{" "}
          <code>model.predict(obs, deterministic=True)</code>. The resulting 4-dimensional action
          (throttle, pitch, yaw, fire) is fed back into <code>env.step(action)</code>, which advances
          the aircraft&apos;s position and orientation, resolves missile firing/impact logic, computes the
          shaped reward, and returns the next observation. That real transition &mdash; not a scripted
          or randomized one &mdash; is what gets serialized and streamed to the 3D viewport over
          WebSocket.
        </p>
      </InfoCard>
    </div>
  );
}

function InfoCard({ title, children, className = "" }: { title: string; children: React.ReactNode; className?: string }) {
  return (
    <div className={`hud-panel rounded-lg p-4 ${className}`}>
      <h3 className="text-xs font-semibold tracking-widest text-aero-accent mb-2">{title.toUpperCase()}</h3>
      {children}
    </div>
  );
}
