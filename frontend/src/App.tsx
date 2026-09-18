import { useState } from "react";
import { NavBar } from "./components/NavBar";
import { DashboardPage } from "./pages/DashboardPage";
import { ModelInfoPage } from "./pages/ModelInfoPage";
import { EvaluationPage } from "./pages/EvaluationPage";
import { useSimulationSocket } from "./hooks/useSimulationSocket";

type Page = "dashboard" | "model" | "evaluation";

export default function App() {
  const [page, setPage] = useState<Page>("dashboard");
  // A single shared simulation socket, lifted here so switching tabs never
  // opens a second WebSocket connection or loses in-flight state.
  const simSocket = useSimulationSocket();

  return (
    <div className="h-screen w-screen flex flex-col bg-aero-bg">
      <NavBar page={page} onNavigate={setPage} wsConnected={simSocket.connectionStatus === "open"} />
      <main className="flex-1 min-h-0">
        {page === "dashboard" && <DashboardPage sim={simSocket} />}
        {page === "model" && <ModelInfoPage />}
        {page === "evaluation" && <EvaluationPage />}
      </main>
    </div>
  );
}
