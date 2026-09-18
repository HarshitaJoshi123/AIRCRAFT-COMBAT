type Page = "dashboard" | "model" | "evaluation";

interface NavBarProps {
  page: Page;
  onNavigate: (page: Page) => void;
  wsConnected: boolean;
}

export function NavBar({ page, onNavigate, wsConnected }: NavBarProps) {
  return (
    <header className="hud-panel flex items-center justify-between px-5 py-3 border-b border-aero-border">
      <div className="flex items-center gap-3">
        <div className="h-2.5 w-2.5 rounded-full bg-aero-accent shadow-[0_0_8px_rgba(34,211,238,0.8)]" />
        <h1 className="font-bold tracking-wide text-slate-100 text-sm md:text-base">
          AI AIRCRAFT COMBAT SIMULATOR
        </h1>
      </div>

      <nav className="flex items-center gap-1">
        <NavLink label="Dashboard" active={page === "dashboard"} onClick={() => onNavigate("dashboard")} />
        <NavLink label="Model Info" active={page === "model"} onClick={() => onNavigate("model")} />
        <NavLink label="Evaluation" active={page === "evaluation"} onClick={() => onNavigate("evaluation")} />
      </nav>

      <div className="flex items-center gap-2 text-xs font-mono text-slate-400">
        <span className={`h-2 w-2 rounded-full ${wsConnected ? "bg-aero-success" : "bg-aero-danger"}`} />
        {wsConnected ? "LIVE" : "DISCONNECTED"}
      </div>
    </header>
  );
}

function NavLink({ label, active, onClick }: { label: string; active: boolean; onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      className={`px-3 py-1.5 text-xs font-semibold rounded-md transition-colors ${
        active ? "bg-aero-accent/20 text-aero-accent" : "text-slate-400 hover:text-slate-200"
      }`}
    >
      {label}
    </button>
  );
}
