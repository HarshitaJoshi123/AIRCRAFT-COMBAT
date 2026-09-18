import { Canvas } from "@react-three/fiber";
import { Arena } from "./Arena";
import { Aircraft } from "./Aircraft";
import { Missile } from "./Missile";
import { Explosion } from "./Explosion";
import { CameraRig } from "./CameraRig";
import { SimulationStateMessage, CameraMode } from "../types/simulation";

interface CombatSceneProps {
  state: SimulationStateMessage | null;
  cameraMode: CameraMode;
}

export function CombatScene({ state, cameraMode }: CombatSceneProps) {
  const agentPos = state?.agent.position ?? [20, 50, 20];
  const enemyPos = state?.enemy.position ?? [80, 50, 80];
  const agentDir = state?.agent.direction ?? [0, 0, 1];
  const enemyDir = state?.enemy.direction ?? [0, 0, -1];

  const showAgentExplosion = state?.agent_hit ?? false;
  const showEnemyExplosion = state?.enemy_hit ?? false;

  return (
    <Canvas
      shadows
      camera={{ position: [30, 22, 30], fov: 50, near: 0.1, far: 2000 }}
      gl={{ antialias: true }}
    >
      <Arena />

      <Aircraft position={agentPos} direction={agentDir} variant="agent" label="AGENT" hit={showAgentExplosion} />
      <Aircraft position={enemyPos} direction={enemyDir} variant="enemy" label="ENEMY" hit={showEnemyExplosion} />

      {state?.missile.active && state.missile.position && (
        <Missile position={state.missile.position} color="#00e5ff" />
      )}
      {state?.enemy_missile.active && state.enemy_missile.position && (
        <Missile position={state.enemy_missile.position} color="#ff5500" />
      )}

      {showAgentExplosion && <Explosion position={agentPos} color="#ff5555" />}
      {showEnemyExplosion && <Explosion position={enemyPos} color="#ffd166" />}

      <CameraRig
        mode={cameraMode}
        agentPosition={agentPos}
        enemyPosition={enemyPos}
        agentDirection={agentDir}
        enemyDirection={enemyDir}
      />
    </Canvas>
  );
}
