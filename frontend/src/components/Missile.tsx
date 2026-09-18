import { Trail } from "@react-three/drei";

interface MissileProps {
  position: [number, number, number];
  color: string;
}

/**
 * A high-visibility missile marker: a saturated, unlit (toneMapped=false)
 * core so it stays vivid instead of getting dimmed by the scene's
 * lighting/tone mapping, a soft outer glow, a point light for local
 * illumination, and a bright, wide trail. Position/active-state come
 * straight from the backend's real missile state.
 */
export function Missile({ position, color }: MissileProps) {
  return (
    <Trail width={5} length={12} color={color} attenuation={(t) => t * t} decay={1}>
      <group position={position}>
        {/* Bright solid core */}
        <mesh>
          <sphereGeometry args={[0.65, 14, 14]} />
          <meshBasicMaterial color={color} toneMapped={false} />
        </mesh>
        {/* Soft outer glow */}
        <mesh>
          <sphereGeometry args={[1.3, 14, 14]} />
          <meshBasicMaterial color={color} toneMapped={false} transparent opacity={0.35} />
        </mesh>
        <mesh>
          <sphereGeometry args={[1.9, 14, 14]} />
          <meshBasicMaterial color={color} toneMapped={false} transparent opacity={0.15} />
        </mesh>
        <pointLight color={color} intensity={8} distance={16} />
      </group>
    </Trail>
  );
}
