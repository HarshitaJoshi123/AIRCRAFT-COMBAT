import { Cloud, Grid } from "@react-three/drei";
import { useMemo } from "react";
import * as THREE from "three";

const SPACE_LIMIT = 100;
const CENTER: [number, number, number] = [SPACE_LIMIT / 2, SPACE_LIMIT / 2, SPACE_LIMIT / 2];

/**
 * A hand-tuned gradient sky dome instead of a physically-based atmosphere
 * shader - the PBR "Sky" model blows out to near-white very easily under
 * normal tone mapping, which is what made the previous pass look washed
 * out. A simple vertical-gradient shader gives direct, predictable color
 * control: a clear mid-blue at the top, a lighter (but not white) blue at
 * the horizon.
 */
function GradientSky() {
  const uniforms = useMemo(
    () => ({
      topColor: { value: new THREE.Color("#3f7fd1") },
      bottomColor: { value: new THREE.Color("#a9cdec") },
      offset: { value: 15 },
      exponent: { value: 0.65 },
    }),
    []
  );

  return (
    <mesh position={CENTER}>
      <sphereGeometry args={[850, 32, 16]} />
      <shaderMaterial
        uniforms={uniforms}
        side={THREE.BackSide}
        fog={false}
        vertexShader={`
          varying vec3 vWorldPosition;
          void main() {
            vec4 worldPosition = modelMatrix * vec4(position, 1.0);
            vWorldPosition = worldPosition.xyz;
            gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
          }
        `}
        fragmentShader={`
          uniform vec3 topColor;
          uniform vec3 bottomColor;
          uniform float offset;
          uniform float exponent;
          varying vec3 vWorldPosition;
          void main() {
            float h = normalize(vWorldPosition + vec3(0.0, offset, 0.0)).y;
            gl_FragColor = vec4(mix(bottomColor, topColor, max(pow(max(h, 0.0), exponent), 0.0)), 1.0);
          }
        `}
      />
    </mesh>
  );
}

export function Arena() {
  const half = SPACE_LIMIT / 2;

  // Fixed, deterministic cloud placements (visual only - not derived from
  // any simulation data), kept sparse and semi-transparent so they add
  // depth without turning the sky white.
  const cloudPuffs = useMemo(
    () => [
      { pos: [-10, 40, 40] as [number, number, number], scale: 5, opacity: 0.3 },
      { pos: [110, 55, 20] as [number, number, number], scale: 6, opacity: 0.25 },
      { pos: [40, 15, 100] as [number, number, number], scale: 5.5, opacity: 0.28 },
      { pos: [90, 25, -10] as [number, number, number], scale: 5, opacity: 0.22 },
    ],
    []
  );

  return (
    <group>
      <GradientSky />

      {cloudPuffs.map((c, i) => (
        <Cloud key={i} position={c.pos} scale={c.scale} opacity={c.opacity} speed={0.04} segments={16} color="#ffffff" />
      ))}

      {/* Moderate lighting - bright enough to read as daytime, restrained
          enough that aircraft materials keep visible shading/contrast
          instead of blowing out to flat white. */}
      <hemisphereLight args={["#bcdcff", "#4a5568", 0.55]} />
      <ambientLight intensity={0.28} />
      <directionalLight
        position={[70, 110, 30]}
        intensity={1.3}
        color="#fff3d6"
        castShadow
        shadow-mapSize-width={1024}
        shadow-mapSize-height={1024}
      />
      {/* Gentle fill from below/behind so the underside of aircraft isn't pure black */}
      <directionalLight position={[-40, -20, -60]} intensity={0.25} color="#8fb8de" />

      {/* Light atmospheric fog - pushed far enough out that it only affects
          distant objects, not the aircraft themselves */}
      <fog attach="fog" args={["#a9cdec", 220, 700]} />

      {/* Floor grid - sparse and subtle, secondary to the aircraft */}
      <Grid
        position={[half, 0, half]}
        args={[SPACE_LIMIT * 1.4, SPACE_LIMIT * 1.4]}
        cellSize={12}
        cellColor="#c3d9ee"
        cellThickness={0.25}
        sectionSize={48}
        sectionColor="#7fa9d1"
        sectionThickness={0.5}
        fadeDistance={240}
        fadeStrength={1.8}
        infiniteGrid={false}
      />

      {/* Wireframe boundary box representing the env's space_limit cube */}
      <mesh position={[half, half, half]}>
        <boxGeometry args={[SPACE_LIMIT, SPACE_LIMIT, SPACE_LIMIT]} />
        <meshBasicMaterial color="#5b9bd5" wireframe transparent opacity={0.15} />
      </mesh>
    </group>
  );
}
