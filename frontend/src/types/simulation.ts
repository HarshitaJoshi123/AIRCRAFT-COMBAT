export type SimulationStatus =
  | "IDLE"
  | "ACTIVE"
  | "PAUSED"
  | "VICTORY"
  | "DEFEAT"
  | "TIMEOUT"
  | "STOPPED";

export interface EntityState {
  position: [number, number, number];
  velocity: [number, number, number];
  direction: [number, number, number];
}

export interface MissileState {
  active: boolean;
  position: [number, number, number] | null;
}

export interface ActionState {
  throttle: number;
  pitch: number;
  yaw: number;
  fire: boolean;
}

export interface SimulationStateMessage {
  type: "state";
  step: number;
  max_steps: number;
  status: SimulationStatus;
  reward: number;
  episode_reward: number;
  distance: number;
  agent: EntityState;
  enemy: EntityState;
  missile: MissileState;
  enemy_missile: MissileState;
  action: ActionState;
  agent_hit: boolean;
  enemy_hit: boolean;
  terminated: boolean;
  truncated: boolean;
}

export interface SimulationEventMessage {
  type: "event";
  step: number;
  message: string;
  level: "info" | "warning" | "critical" | "success";
}

export interface SimulationErrorMessage {
  type: "error";
  message: string;
}

export interface SimulationStatusSnapshot {
  type: "status";
  status: SimulationStatus;
  running: boolean;
  step: number;
  max_steps: number;
  episode_reward: number;
}

export type SimulationSocketMessage =
  | SimulationStateMessage
  | SimulationEventMessage
  | SimulationErrorMessage
  | SimulationStatusSnapshot;

export interface EvaluationProgressMessage {
  type: "progress";
  completed: number;
  total: number;
  wins: number;
  losses: number;
}

export interface EvaluationResultMessage {
  type: "result";
  episodes: number;
  wins: number;
  losses: number;
  win_rate: number;
  loss_rate: number;
  avg_reward: number;
  max_reward: number;
  min_reward: number;
  episode_rewards: number[];
  episode_outcomes: ("win" | "loss")[];
}

export interface EvaluationStatusMessage {
  type: "status";
  running: boolean;
  completed: number;
  total: number;
  wins: number;
  losses: number;
  done: boolean;
  error: string | null;
}

export interface EvaluationErrorMessage {
  type: "error";
  message: string;
}

export type EvaluationSocketMessage =
  | EvaluationProgressMessage
  | EvaluationResultMessage
  | EvaluationStatusMessage
  | EvaluationErrorMessage;

export type CameraMode = "follow_agent" | "follow_enemy" | "free" | "top";
