import { useMemo, useRef } from "react";
import { useFrame } from "@react-three/fiber";
import * as THREE from "three";

interface ExplosionProps {
  position: [number, number, number];
  color?: string;
}

const PARTICLE_COUNT = 60;

/**
 * A professional-looking hit effect: an expanding flash sphere plus a
 * particle burst. Rendered only when the backend's real agent_hit /
 * enemy_hit flag is true for the current state - never on a timer or
 * randomly.
 */
export function Explosion({ position, color = "#ffb020" }: ExplosionProps) {
  const pointsRef = useRef<THREE.Points>(null);
  const flashRef = useRef<THREE.Mesh>(null);
  const age = useRef(0);

  const velocities = useMemo(
    () =>
      Array.from({ length: PARTICLE_COUNT }, () => {
        const v = new THREE.Vector3(Math.random() - 0.5, Math.random() - 0.5, Math.random() - 0.5);
        return v.normalize().multiplyScalar(0.08 + Math.random() * 0.28);
      }),
    []
  );
  const positions = useMemo(() => new Float32Array(PARTICLE_COUNT * 3), []);

  useFrame((_, delta) => {
    age.current += delta;

    const geom = pointsRef.current?.geometry;
    if (geom) {
      const arr = geom.attributes.position.array as Float32Array;
      for (let i = 0; i < PARTICLE_COUNT; i++) {
        arr[i * 3] += velocities[i].x;
        arr[i * 3 + 1] += velocities[i].y;
        arr[i * 3 + 2] += velocities[i].z;
      }
      geom.attributes.position.needsUpdate = true;
      const material = pointsRef.current?.material as THREE.PointsMaterial;
      if (material) material.opacity = Math.max(0, 1 - age.current / 1.4);
    }

    if (flashRef.current) {
      const s = 1 + age.current * 8;
      flashRef.current.scale.setScalar(s);
      const mat = flashRef.current.material as THREE.MeshBasicMaterial;
      mat.opacity = Math.max(0, 0.9 - age.current * 2.2);
    }
  });

  return (
    <group position={position}>
      <mesh ref={flashRef}>
        <sphereGeometry args={[0.9, 16, 16]} />
        <meshBasicMaterial color="#fff3c4" transparent opacity={0.9} toneMapped={false} />
      </mesh>
      <pointLight color={color} intensity={14} distance={20} decay={2} />
      <points ref={pointsRef}>
        <bufferGeometry>
          <bufferAttribute attach="attributes-position" args={[positions, 3]} />
        </bufferGeometry>
        <pointsMaterial color={color} size={0.9} transparent opacity={1} sizeAttenuation />
      </points>
    </group>
  );
}
