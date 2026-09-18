import { useRef } from "react";
import { useFrame, useThree } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import * as THREE from "three";
import { CameraMode } from "../types/simulation";

interface CameraRigProps {
  mode: CameraMode;
  agentPosition: [number, number, number];
  enemyPosition: [number, number, number];
  agentDirection: [number, number, number];
  enemyDirection: [number, number, number];
}

const SPACE_LIMIT = 100;

/**
 * Chase-cam style follow: the camera sits behind and above the tracked
 * aircraft ALONG ITS ACTUAL HEADING (the real `direction` vector from the
 * backend), not a fixed world-space offset. That's what makes the other
 * aircraft appear naturally "ahead" in frame, cinematic-dogfight style,
 * instead of the camera ending up beside or behind the action at a
 * arbitrary angle.
 */
export function CameraRig({ mode, agentPosition, enemyPosition, agentDirection, enemyDirection }: CameraRigProps) {
  const { camera } = useThree();
  const controlsRef = useRef<any>(null);
  const desiredPos = useRef(new THREE.Vector3(30, 22, 30));
  const desiredTarget = useRef(new THREE.Vector3(50, 50, 50));

  useFrame(() => {
    if (mode === "follow_agent" || mode === "follow_enemy") {
      const pos = mode === "follow_agent" ? agentPosition : enemyPosition;
      const dir = mode === "follow_agent" ? agentDirection : enemyDirection;

      const p = new THREE.Vector3(...pos);
      let d = new THREE.Vector3(...dir);
      if (d.lengthSq() < 1e-6) d.set(0, 0, 1);
      d.normalize();

      const behindDistance = 9;
      const heightOffset = 3.3;
      const lookAheadDistance = 10;

      desiredPos.current.copy(p).addScaledVector(d, -behindDistance);
      desiredPos.current.y += heightOffset;

      desiredTarget.current.copy(p).addScaledVector(d, lookAheadDistance);
    } else if (mode === "top") {
      desiredPos.current.set(50, 260, 50.001);
      desiredTarget.current.set(50, 0, 50);
    } else {
      // free: leave camera under OrbitControls' control entirely
      return;
    }

    camera.position.lerp(desiredPos.current, 0.09);
    if (controlsRef.current) {
      controlsRef.current.target.lerp(desiredTarget.current, 0.09);
      controlsRef.current.update();
    } else {
      camera.lookAt(desiredTarget.current);
    }
  });

  return (
    <OrbitControls
      ref={controlsRef}
      enabled={mode === "free"}
      minDistance={8}
      maxDistance={350}
      target={[SPACE_LIMIT / 2, SPACE_LIMIT / 2, SPACE_LIMIT / 2]}
    />
  );
}
