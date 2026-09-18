const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

async function post(path: string) {
  const res = await fetch(`${API_URL}${path}`, { method: "POST" });
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(body.detail || `Request to ${path} failed (${res.status})`);
  }
  return res.json();
}

async function get(path: string) {
  const res = await fetch(`${API_URL}${path}`);
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(body.detail || `Request to ${path} failed (${res.status})`);
  }
  return res.json();
}

export const api = {
  health: () => get("/api/health"),
  startSimulation: () => post("/api/simulation/start"),
  pauseSimulation: () => post("/api/simulation/pause"),
  resumeSimulation: () => post("/api/simulation/resume"),
  resetSimulation: () => post("/api/simulation/reset"),
  stopSimulation: () => post("/api/simulation/stop"),
  simulationStatus: () => get("/api/simulation/status"),
  evaluationStatus: () => get("/api/evaluation/status"),
};

export { API_URL };
