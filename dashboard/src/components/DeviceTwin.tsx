import { Canvas, useFrame } from '@react-three/fiber'
import { OrbitControls, RoundedBox } from '@react-three/drei'
import { useEffect, useRef, useState } from 'react'
import * as THREE from 'three'
import type { DevicePose, InferenceEvent } from '../lib/types'

type DeviceState = 'local' | 'cloud' | 'failover' | 'tracking' | 'offline'

function ageMs(receivedAt?: string | null): number {
  if (!receivedAt) return Number.POSITIVE_INFINITY
  const parsed = new Date(receivedAt).getTime()
  return Number.isFinite(parsed) ? Math.max(0, Date.now() - parsed) : Number.POSITIVE_INFINITY
}

function StatusHalo({ state }: { state: DeviceState }) {
  const ring = useRef<THREE.Mesh>(null)
  const palette: Record<DeviceState, string> = {
    local: '#47e6a5',
    cloud: '#7c9cff',
    failover: '#ff9f66',
    tracking: '#55c9ff',
    offline: '#5f6b7a',
  }
  useFrame(({ clock }) => {
    if (!ring.current) return
    const pulse = 1 + Math.sin(clock.elapsedTime * 2.3) * 0.025
    ring.current.scale.setScalar(pulse)
  })
  return (
    <mesh ref={ring} rotation={[Math.PI / 2, 0, 0]} position={[0, -0.48, 0]}>
      <torusGeometry args={[2.35, 0.035, 20, 120]} />
      <meshStandardMaterial color={palette[state]} emissive={palette[state]} emissiveIntensity={1.4} />
    </mesh>
  )
}

function PinRow({ z }: { z: number }) {
  return (
    <group>
      {Array.from({ length: 14 }).map((_, index) => (
        <mesh key={index} position={[-2.2 + index * 0.34, -0.18, z]}>
          <cylinderGeometry args={[0.035, 0.035, 0.32, 10]} />
          <meshStandardMaterial color="#d8b35a" metalness={0.7} roughness={0.28} />
        </mesh>
      ))}
    </group>
  )
}

function BoardGeometry({ accent }: { accent: string }) {
  return (
    <>
      <RoundedBox args={[4.7, 0.24, 2.55]} radius={0.12} smoothness={4} position={[0, 0, 0]}>
        <meshStandardMaterial color="#0f4c3f" roughness={0.48} metalness={0.22} />
      </RoundedBox>
      <RoundedBox args={[1.35, 0.28, 1.35]} radius={0.08} smoothness={4} position={[0.25, 0.28, 0.05]}>
        <meshStandardMaterial color="#202834" roughness={0.32} metalness={0.72} />
      </RoundedBox>
      <RoundedBox args={[1.7, 0.22, 0.9]} radius={0.04} smoothness={3} position={[-1.45, 0.24, 0.05]}>
        <meshStandardMaterial color="#c8c8c8" roughness={0.22} metalness={0.9} />
      </RoundedBox>
      <RoundedBox args={[0.62, 0.2, 0.48]} radius={0.035} smoothness={3} position={[1.62, 0.22, 0.65]}>
        <meshStandardMaterial color="#151c25" roughness={0.4} metalness={0.42} />
      </RoundedBox>
      <PinRow z={-1.2} />
      <PinRow z={1.2} />

      <group position={[1.72, 0.48, -0.45]} rotation={[0, 0.04, 0]}>
        <RoundedBox args={[1.28, 0.12, 1.05]} radius={0.05} smoothness={3}>
          <meshStandardMaterial color="#145aa6" roughness={0.55} metalness={0.15} />
        </RoundedBox>
        <RoundedBox args={[0.46, 0.18, 0.46]} radius={0.035} smoothness={3} position={[0, 0.14, 0]}>
          <meshStandardMaterial color="#252c36" roughness={0.35} metalness={0.5} />
        </RoundedBox>
        {Array.from({ length: 8 }).map((_, index) => (
          <mesh key={index} position={[-0.5 + index * 0.145, 0.04, -0.46]}>
            <cylinderGeometry args={[0.025, 0.025, 0.22, 8]} />
            <meshStandardMaterial color="#d4ad54" metalness={0.7} />
          </mesh>
        ))}
      </group>

      <mesh position={[0.24, 0.44, 0.05]}>
        <sphereGeometry args={[0.06, 18, 18]} />
        <meshStandardMaterial color={accent} emissive={accent} emissiveIntensity={3} />
      </mesh>
    </>
  )
}

function PoseDrivenBoard({ pose, accent }: { pose: DevicePose | null; accent: string }) {
  const poseGroup = useRef<THREE.Group>(null)
  const targetQuaternion = useRef(new THREE.Quaternion())

  useEffect(() => {
    if (!pose) {
      targetQuaternion.current.identity()
      return
    }

    // orientation-v1 home pose has sensor +Z upward. The rendered board lies in
    // Three.js X/Z with +Y as its normal, so sensor axes map to scene rotations as:
    // roll(+X) -> scene X, pitch(+Y) -> -scene Z, yaw(+Z) -> scene Y.
    const roll = THREE.MathUtils.degToRad(pose.roll_deg_est)
    const pitch = THREE.MathUtils.degToRad(pose.pitch_deg_est)
    const yaw = THREE.MathUtils.degToRad(pose.yaw_rel_deg_est)
    const targetEuler = new THREE.Euler(roll, yaw, -pitch, 'YXZ')
    targetQuaternion.current.setFromEuler(targetEuler)
  }, [pose])

  useFrame((_, delta) => {
    if (!poseGroup.current) return
    const blend = 1 - Math.exp(-10 * delta)
    poseGroup.current.quaternion.slerp(targetQuaternion.current, blend)
  })

  return (
    <group ref={poseGroup}>
      <BoardGeometry accent={accent} />
    </group>
  )
}

function DeviceModel({ latest, pose, poseFresh }: {
  latest: InferenceEvent | null
  pose: DevicePose | null
  poseFresh: boolean
}) {
  const decisionFresh = latest ? ageMs(latest.received_at) <= 5000 : false
  const deviceFresh = poseFresh || decisionFresh
  const state: DeviceState = !deviceFresh
    ? 'offline'
    : !latest
      ? 'tracking'
      : latest.failover
        ? 'failover'
        : latest.execution_mode === 'CLOUD'
          ? 'cloud'
          : 'local'

  const accent = state === 'cloud'
    ? '#7c9cff'
    : state === 'failover'
      ? '#ff9f66'
      : state === 'offline'
        ? '#5f6b7a'
        : state === 'tracking'
          ? '#55c9ff'
          : '#47e6a5'

  return (
    <group rotation={[-0.12, -0.22, 0]}>
      <StatusHalo state={state} />
      <PoseDrivenBoard pose={pose} accent={accent} />
    </group>
  )
}

function formatAngle(value?: number | null): string {
  if (value == null || !Number.isFinite(value)) return '—'
  return `${value >= 0 ? '+' : ''}${value.toFixed(1)}°`
}

function formatAge(milliseconds: number): string {
  if (!Number.isFinite(milliseconds)) return '—'
  if (milliseconds < 1000) return `${Math.round(milliseconds)} ms`
  return `${(milliseconds / 1000).toFixed(1)} s`
}

export function DeviceTwin({ latest, pose, poseConnected }: {
  latest: InferenceEvent | null
  pose: DevicePose | null
  poseConnected: boolean
}) {
  const [clockTick, setClockTick] = useState(0)
  useEffect(() => {
    const timer = window.setInterval(() => setClockTick((value) => value + 1), 250)
    return () => window.clearInterval(timer)
  }, [])

  // clockTick intentionally participates in this calculation so pose freshness is
  // reevaluated even when no new telemetry arrives.
  void clockTick
  const poseAge = ageMs(pose?.received_at)
  const poseFresh = pose != null && poseAge <= 2000
  const poseStatus = poseFresh ? 'LIVE ATTITUDE' : pose ? 'POSE STALE' : 'AWAITING POSE'

  return (
    <div className="device-twin-canvas">
      <Canvas camera={{ position: [5.5, 4.4, 6.2], fov: 38 }} dpr={[1, 1.7]} shadows>
        <color attach="background" args={['#0a0f18']} />
        <ambientLight intensity={1.4} />
        <directionalLight position={[4, 6, 4]} intensity={2.5} color="#dfe8ff" />
        <pointLight position={[-5, 1, -3]} intensity={16} distance={12} color="#2e76ff" />
        <pointLight position={[4, 2, 4]} intensity={12} distance={10} color="#3effbc" />
        <DeviceModel latest={latest} pose={pose} poseFresh={poseFresh} />
        <OrbitControls
          enablePan={false}
          minDistance={5.5}
          maxDistance={10}
          minPolarAngle={0.65}
          maxPolarAngle={1.5}
          autoRotate={false}
        />
      </Canvas>

      <div className={`pose-stream-badge ${poseFresh ? 'live' : pose ? 'stale' : 'offline'}`}>
        <span>{poseStatus}</span>
        <strong>{poseConnected ? 'POSE WS' : 'REST FALLBACK'}</strong>
        <small>{pose ? `age ${formatAge(poseAge)}` : 'no pose rows yet'}</small>
      </div>

      <div className="pose-readout">
        <div><span>Roll est.</span><strong>{formatAngle(pose?.roll_deg_est)}</strong></div>
        <div><span>Pitch est.</span><strong>{formatAngle(pose?.pitch_deg_est)}</strong></div>
        <div><span>Yaw rel.</span><strong>{formatAngle(pose?.yaw_rel_deg_est)}</strong></div>
      </div>

      <div className="twin-note">
        Sensor-driven attitude · roll/pitch complementary estimate · yaw is boot-relative and drift-prone · geometry is representative
      </div>
    </div>
  )
}
