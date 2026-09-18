import { SimulationStateMessage } from "../types/simulation";

export type Vec3 = [number, number, number];

function lerp3(a: Vec3, b: Vec3, t: number): Vec3 {
  return [
    a[0] + (b[0] - a[0]) * t,
    a[1] + (b[1] - a[1]) * t,
    a[2] + (b[2] - a[2]) * t,
  ];
}

/**
 * Pure visual interpolation between the last two REAL backend states.
 * This never invents new simulation outcomes - it only smooths the camera
 * and mesh positions between two authoritative snapshots so the ~12Hz
 * backend tick rate doesn't look choppy at 60fps.
 */
export function interpolateState(
  prev: SimulationStateMessage | null,
  next: SimulationStateMessage,
  t: number
): SimulationStateMessage {
  if (!prev || prev.step === next.step) return next;
  const clampedT = Math.min(1, Math.max(0, t));

  return {
    ...next,
    agent: {
      ...next.agent,
      position: lerp3(prev.agent.position, next.agent.position, clampedT),
    },
    enemy: {
      ...next.enemy,
      position: lerp3(prev.enemy.position, next.enemy.position, clampedT),
    },
    missile:
      next.missile.active && prev.missile.active && prev.missile.position && next.missile.position
        ? { active: true, position: lerp3(prev.missile.position, next.missile.position, clampedT) }
        : next.missile,
    enemy_missile:
      next.enemy_missile.active &&
      prev.enemy_missile.active &&
      prev.enemy_missile.position &&
      next.enemy_missile.position
        ? {
            active: true,
            position: lerp3(prev.enemy_missile.position, next.enemy_missile.position, clampedT),
          }
        : next.enemy_missile,
  };
}
