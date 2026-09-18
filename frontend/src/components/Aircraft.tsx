import { useMemo, useRef } from "react";
import { useFrame } from "@react-three/fiber";
import { Html } from "@react-three/drei";
import * as THREE from "three";

interface AircraftProps {
  position: [number, number, number];
  direction: [number, number, number];
  variant: "agent" | "enemy";
  label?: string;
  hit?: boolean;
  scale?: number;
}

const PALETTE = {
  agent: {
    body: "#31404f",
    belly: "#1b2530",
    accent: "#38bdf8",
    emissive: "#38bdf8",
    tag: "#0369a1",
    canopy: "#0b1520",
  },
  enemy: {
    body: "#4a1414",
    belly: "#241010",
    accent: "#f87171",
    emissive: "#ef4444",
    tag: "#991b1b",
    canopy: "#1a0a0a",
  },
};

/**
 * A more visually polished procedural jet - still pure Three.js primitives
 * (no external model files, per the project's constraints), but composed
 * from a smoother capsule fuselage, a glossy clearcoat finish, swept wings,
 * canted twin tails, wingtip navigation lights, and a glowing afterburner
 * nozzle. Purely cosmetic: geometry/materials only, no change to how
 * position/direction are consumed from the real backend state.
 */
export function Aircraft({ position, direction, variant, label, hit, scale = 3 }: AircraftProps) {
  const group = useRef<THREE.Group>(null);
  const palette = PALETTE[variant];

  const quaternion = useMemo(() => {
    const dir = new THREE.Vector3(...direction).normalize();
    if (dir.lengthSq() === 0) return new THREE.Quaternion();
    const forward = new THREE.Vector3(0, 0, 1);
    return new THREE.Quaternion().setFromUnitVectors(forward, dir);
  }, [direction[0], direction[1], direction[2]]);

  useFrame(() => {
    if (group.current) {
      group.current.quaternion.slerp(quaternion, 0.35);
    }
  });

  const isEnemy = variant === "enemy";
  const bodyColor = hit ? "#ffdd55" : palette.body;
  const bodyEmissive = hit ? "#ff8800" : palette.emissive;
  const bodyEmissiveIntensity = hit ? 1.8 : 0.18;

  return (
    <group ref={group} position={position} scale={scale}>
      {/* Fuselage - smooth capsule for a rounded, aerodynamic body */}
      <mesh rotation={[Math.PI / 2, 0, 0]} position={[0, 0, 0.1]} castShadow>
        <capsuleGeometry args={[0.42, 1.9, 6, 14]} />
        <meshPhysicalMaterial
          color={bodyColor}
          emissive={bodyEmissive}
          emissiveIntensity={bodyEmissiveIntensity}
          metalness={0.75}
          roughness={0.28}
          clearcoat={0.4}
          clearcoatRoughness={0.25}
        />
      </mesh>

      {/* Tapered nose cone */}
      <mesh rotation={[Math.PI / 2, 0, 0]} position={[0, -0.02, 1.55]} castShadow>
        <coneGeometry args={[0.38, 1.0, 14]} />
        <meshPhysicalMaterial color={bodyColor} metalness={0.75} roughness={0.3} clearcoat={0.4} />
      </mesh>

      {/* Belly / underside panel line for visual depth */}
      <mesh position={[0, -0.38, 0.1]}>
        <boxGeometry args={[0.5, 0.12, 1.9]} />
        <meshStandardMaterial color={palette.belly} metalness={0.6} roughness={0.5} />
      </mesh>

      {/* Canopy */}
      <mesh position={[0, 0.34, 0.75]} castShadow>
        <sphereGeometry args={[0.3, 12, 12, 0, Math.PI * 2, 0, Math.PI / 1.7]} />
        <meshPhysicalMaterial
          color={palette.canopy}
          metalness={0.3}
          roughness={0.05}
          transmission={0.15}
          transparent
          opacity={0.92}
          clearcoat={1}
        />
      </mesh>

      {/* Main wings - swept trapezoid pairs, angled back for a sleek look */}
      <mesh position={[1.05, -0.05, -0.55]} rotation={[0, 0, -0.12]} castShadow>
        <boxGeometry args={[1.9, 0.07, 1.15]} />
        <meshStandardMaterial color={palette.accent} metalness={0.55} roughness={0.35} />
      </mesh>
      <mesh position={[-1.05, -0.05, -0.55]} rotation={[0, 0, 0.12]} castShadow>
        <boxGeometry args={[1.9, 0.07, 1.15]} />
        <meshStandardMaterial color={palette.accent} metalness={0.55} roughness={0.35} />
      </mesh>

      {/* Small forward canards for extra fighter-jet silhouette detail */}
      <mesh position={[0.55, 0.02, 0.85]} rotation={[0, 0, -0.08]}>
        <boxGeometry args={[0.7, 0.05, 0.35]} />
        <meshStandardMaterial color={palette.body} metalness={0.6} roughness={0.4} />
      </mesh>
      <mesh position={[-0.55, 0.02, 0.85]} rotation={[0, 0, 0.08]}>
        <boxGeometry args={[0.7, 0.05, 0.35]} />
        <meshStandardMaterial color={palette.body} metalness={0.6} roughness={0.4} />
      </mesh>

      {/* Wingtip navigation lights - red on left/port, green on right/starboard,
          a classic authentic aircraft touch (purely decorative) */}
      <mesh position={[1.95, -0.05, -0.9]}>
        <sphereGeometry args={[0.06, 8, 8]} />
        <meshBasicMaterial color="#22c55e" toneMapped={false} />
      </mesh>
      <mesh position={[-1.95, -0.05, -0.9]}>
        <sphereGeometry args={[0.06, 8, 8]} />
        <meshBasicMaterial color="#ef4444" toneMapped={false} />
      </mesh>

      {/* Twin canted tail fins */}
      <mesh position={[0.32, 0.5, -1.15]} rotation={[0.2, 0, -0.28]}>
        <boxGeometry args={[0.07, 0.85, 0.6]} />
        <meshStandardMaterial color={palette.body} metalness={0.65} roughness={0.32} />
      </mesh>
      <mesh position={[-0.32, 0.5, -1.15]} rotation={[0.2, 0, 0.28]}>
        <boxGeometry args={[0.07, 0.85, 0.6]} />
        <meshStandardMaterial color={palette.body} metalness={0.65} roughness={0.32} />
      </mesh>

      {/* Enemy-only intake details for a slightly different, recognizable
          silhouette from the agent aircraft */}
      {isEnemy && (
        <>
          <mesh position={[0.35, -0.15, 0.3]}>
            <boxGeometry args={[0.22, 0.22, 0.7]} />
            <meshStandardMaterial color={palette.belly} metalness={0.6} roughness={0.4} />
          </mesh>
          <mesh position={[-0.35, -0.15, 0.3]}>
            <boxGeometry args={[0.22, 0.22, 0.7]} />
            <meshStandardMaterial color={palette.belly} metalness={0.6} roughness={0.4} />
          </mesh>
        </>
      )}

      {/* Nose sensor light */}
      <mesh position={[0, 0, 1.95]}>
        <sphereGeometry args={[0.1, 8, 8]} />
        <meshBasicMaterial color={palette.emissive} toneMapped={false} />
      </mesh>

      {/* Afterburner nozzle + glow */}
      <mesh rotation={[Math.PI / 2, 0, 0]} position={[0, -0.02, -1.35]}>
        <cylinderGeometry args={[0.22, 0.3, 0.35, 14]} />
        <meshStandardMaterial color="#1c1c1c" metalness={0.8} roughness={0.4} />
      </mesh>
      <mesh position={[0, -0.02, -1.65]}>
        <sphereGeometry args={[0.2, 10, 10]} />
        <meshBasicMaterial color="#ffb347" toneMapped={false} />
      </mesh>
      <pointLight position={[0, -0.02, -1.7]} color="#ffb347" intensity={2.5} distance={5} />

      {hit && <pointLight color="#ff6a3d" intensity={12} distance={14} />}

      {label && (
        <Html position={[0, 1.5, 0]} center distanceFactor={40} occlude={false}>
          <div
            style={{
              background: palette.tag,
              color: "white",
              fontFamily: "monospace",
              fontSize: "11px",
              fontWeight: 700,
              padding: "2px 8px",
              borderRadius: "4px",
              whiteSpace: "nowrap",
              boxShadow: "0 0 8px rgba(0,0,0,0.4)",
              pointerEvents: "none",
            }}
          >
            {label}
          </div>
        </Html>
      )}
    </group>
  );
}
