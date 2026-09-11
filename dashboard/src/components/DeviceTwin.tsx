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
    const pulse = 1 + Math.sin(clock.elapsedTime * 2.2) * 0.014
    ring.current.scale.setScalar(pulse)
  })
  return (
    <mesh ref={ring} rotation={[Math.PI / 2, 0, 0]} position={[0, -0.48, 0]}>
      <torusGeometry args={[2.28, 0.024, 18, 120]} />
      <meshStandardMaterial color={palette[state]} emissive={palette[state]} emissiveIntensity={1.35} />
    </mesh>
  )
}

function GoldPinRail({ x }: { x: number }) {
  const pins = Array.from({ length: 10 }, (_, index) => index)
  return (
    <group>
      {pins.map((index) => (
        <mesh key={`${x}-${index}`} position={[x, 0.28, -1.05 + index * 0.235]}>
          <boxGeometry args={[0.035, 0.2, 0.055]} />
          <meshStandardMaterial color="#d6aa45" roughness={0.28} metalness={0.78} />
        </mesh>
      ))}
    </group>
  )
}

function UsbLead() {
  return (
    <group position={[0, 0.22, 2.15]}>
      <RoundedBox args={[0.5, 0.19, 0.32]} radius={0.035} smoothness={3}>
        <meshStandardMaterial color="#9fa9b5" roughness={0.24} metalness={0.84} />
      </RoundedBox>
      <mesh position={[0, -0.02, 0.72]} rotation={[Math.PI / 2, 0, 0]}>
        <cylinderGeometry args={[0.07, 0.07, 1.15, 16]} />
        <meshStandardMaterial color="#a8b2c0" roughness={0.52} metalness={0.12} />
      </mesh>
    </group>
  )
}

function AxisMarker() {
  return (
    <group position={[-1.38, 0.72, -1.55]}>
      <arrowHelper args={[new THREE.Vector3(1, 0, 0), new THREE.Vector3(0, 0, 0), 0.72, '#ff6b76', 0.14, 0.08]} />
      <arrowHelper args={[new THREE.Vector3(0, 0, -1), new THREE.Vector3(0, 0, 0), 0.72, '#56d28f', 0.14, 0.08]} />
      <arrowHelper args={[new THREE.Vector3(0, 1, 0), new THREE.Vector3(0, 0, 0), 0.72, '#57a7ff', 0.14, 0.08]} />
    </group>
  )
}

function ReferenceRigGeometry({ accent }: { accent: string }) {
  return (
    <>
      {/* Stylized physical carrier aligned to Orientations.png.
          Home Pose is flat with sensor +Z normal to the top surface. */}
      <RoundedBox args={[2.34, 0.16, 4.45]} radius={0.13} smoothness={5} position={[0, -0.02, 0]}>
        <meshStandardMaterial color="#151d29" roughness={0.42} metalness={0.2} />
      </RoundedBox>
      <RoundedBox args={[2.12, 0.08, 4.16]} radius={0.1} smoothness={4} position={[0, 0.085, 0]}>
        <meshStandardMaterial color="#202b3a" roughness={0.58} metalness={0.08} />
      </RoundedBox>

      {/* Subtle rails preserve the long-axis silhouette visible in the physical reference. */}
      <RoundedBox args={[0.08, 0.08, 3.82]} radius={0.025} smoothness={2} position={[-0.93, 0.15, 0]}>
        <meshStandardMaterial color="#2e4155" roughness={0.4} metalness={0.18} />
      </RoundedBox>
      <RoundedBox args={[0.08, 0.08, 3.82]} radius={0.025} smoothness={2} position={[0.93, 0.15, 0]}>
        <meshStandardMaterial color="#2e4155" roughness={0.4} metalness={0.18} />
      </RoundedBox>

      {/* ESP32-S3 development board: dominant dark PCB centered on the carrier. */}
      <RoundedBox args={[1.28, 0.13, 2.7]} radius={0.07} smoothness={4} position={[-0.12, 0.22, 0.18]}>
        <meshStandardMaterial color="#0b2f35" roughness={0.38} metalness={0.2} />
      </RoundedBox>
      <GoldPinRail x={-0.78} />
      <GoldPinRail x={0.54} />

      {/* ESP32-S3 module can and SoC area. */}
      <RoundedBox args={[0.9, 0.16, 0.86]} radius={0.045} smoothness={3} position={[-0.12, 0.35, -0.28]}>
        <meshStandardMaterial color="#9fa9b5" roughness={0.22} metalness={0.86} />
      </RoundedBox>
      <RoundedBox args={[0.42, 0.13, 0.42]} radius={0.035} smoothness={3} position={[-0.12, 0.35, 0.66]}>
        <meshStandardMaterial color="#121821" roughness={0.28} metalness={0.48} />
      </RoundedBox>

      {/* GY-521 / MPU6050, offset on the same plane to make orientation visually asymmetric. */}
      <group position={[0.72, 0.31, -1.42]}>
        <RoundedBox args={[0.82, 0.105, 0.72]} radius={0.055} smoothness={4}>
          <meshStandardMaterial color="#1268c4" roughness={0.38} metalness={0.18} />
        </RoundedBox>
        <RoundedBox args={[0.3, 0.13, 0.3]} radius={0.025} smoothness={3} position={[0, 0.115, 0]}>
          <meshStandardMaterial color="#151a22" roughness={0.28} metalness={0.5} />
        </RoundedBox>
        <mesh position={[-0.25, 0.12, 0.22]}>
          <sphereGeometry args={[0.042, 14, 14]} />
          <meshStandardMaterial color="#dce8f6" emissive="#dce8f6" emissiveIntensity={0.5} />
        </mesh>
      </group>

      <UsbLead />
      <AxisMarker />

      {/* Status LED is intentionally small; status color must not dominate physical pose. */}
      <mesh position={[-0.43, 0.39, 0.96]}>
        <sphereGeometry args={[0.052, 16, 16]} />
        <meshStandardMaterial color={accent} emissive={accent} emissiveIntensity={2.6} />
      </mesh>

      {/* Front marker on the USB side makes +Z/-Z and ±X/±Y views easier to disambiguate. */}
      <RoundedBox args={[0.78, 0.04, 0.09]} radius={0.02} smoothness={2} position={[0, 0.19, 1.93]}>
        <meshStandardMaterial color="#4f9cff" emissive="#245b98" emissiveIntensity={0.7} roughness={0.32} />
      </RoundedBox>
    </>
  )
}

function PoseDrivenRig({ pose, accent }: { pose: DevicePose | null; accent: string }) {
  const poseGroup = useRef<THREE.Group>(null)
  const targetQuaternion = useRef(new THREE.Quaternion())

  useEffect(() => {
    if (!pose) {
      targetQuaternion.current.identity()
      return
    }

    // orientation-v1 physical convention:
    // Home Pose = sensor +Z upward.
    // sensor +X -> scene +X, sensor +Y -> scene -Z, sensor +Z -> scene +Y.
    const roll = THREE.MathUtils.degToRad(pose.roll_deg_est)
    const pitch = THREE.MathUtils.degToRad(pose.pitch_deg_est)
    const yaw = THREE.MathUtils.degToRad(pose.yaw_rel_deg_est)
    targetQuaternion.current.setFromEuler(new THREE.Euler(roll, yaw, -pitch, 'YXZ'))
  }, [pose])

  useFrame((_, delta) => {
    if (!poseGroup.current) return
    // With the 10 Hz pose target in r1-pose-v1.1, this keeps transitions smooth
    // while converging quickly enough to avoid the visible step-and-wait behavior
    // observed with the former 5 Hz + aggressive interpolation combination.
    const blend = 1 - Math.exp(-24 * delta)
    poseGroup.current.quaternion.slerp(targetQuaternion.current, blend)
  })

  return (
    <group ref={poseGroup}>
      <ReferenceRigGeometry accent={accent} />
    </group>
  )
}

function DeviceModel({ latest, pose, poseFresh }: {
  latest: InferenceEvent | null
  pose: DevicePose | null
  poseFresh: boolean
}) {
  const decisionFresh = latest ? ageMs(latest.received_at) <= 5000 : false
  const state: DeviceState = decisionFresh && latest
    ? latest.failover
      ? 'failover'
      : latest.execution_mode === 'CLOUD'
        ? 'cloud'
        : 'local'
    : poseFresh
      ? 'tracking'
      : 'offline'

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
    <group>
      <StatusHalo state={state} />
      <PoseDrivenRig pose={pose} accent={accent} />
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
    const timer = window.setInterval(() => setClockTick((value) => value + 1), 100)
    return () => window.clearInterval(timer)
  }, [])

  void clockTick
  const poseAge = ageMs(pose?.received_at)
  const poseFresh = pose != null && poseAge <= 2000
  const poseStatus = poseFresh ? 'LIVE ATTITUDE' : pose ? 'POSE STALE' : 'AWAITING POSE'

  return (
    <div className="device-twin-canvas">
      <Canvas camera={{ position: [4.7, 3.25, 5.35], fov: 35 }} dpr={[1, 1.6]} shadows>
        <color attach="background" args={['#090e16']} />
        <ambientLight intensity={1.25} />
        <directionalLight position={[4, 6, 4]} intensity={2.1} color="#dce7ff" />
        <pointLight position={[-4, 2, -3]} intensity={8} distance={11} color="#287dff" />
        <pointLight position={[4, 2, 4]} intensity={6} distance={10} color="#38dba1" />
        <DeviceModel latest={latest} pose={pose} poseFresh={poseFresh} />
        <OrbitControls
          enablePan={false}
          enableRotate={false}
          enableZoom
          minDistance={6.1}
          maxDistance={9.2}
          autoRotate={false}
        />
      </Canvas>

      <div className={`pose-stream-badge ${poseFresh ? 'live' : pose ? 'stale' : 'offline'}`}>
        <span>{poseStatus}</span>
        <strong>{poseConnected ? 'POSE WS' : 'REST FALLBACK'}</strong>
        <small>{pose ? `age ${formatAge(poseAge)}` : 'no pose rows yet'}</small>
      </div>

      <div className="orientation-reference">
        <strong>REFERENCE · HOME +Z</strong>
        <span><i className="axis-x" /> +X right</span>
        <span><i className="axis-y" /> +Y forward</span>
        <span><i className="axis-z" /> +Z up</span>
      </div>

      <div className="pose-readout">
        <div><span>Roll est.</span><strong>{formatAngle(pose?.roll_deg_est)}</strong></div>
        <div><span>Pitch est.</span><strong>{formatAngle(pose?.pitch_deg_est)}</strong></div>
        <div><span>Yaw rel.</span><strong>{formatAngle(pose?.yaw_rel_deg_est)}</strong></div>
      </div>

      <div className="twin-note">
        Sensor-driven reference rig · +Z Home · roll/pitch estimated · yaw boot-relative and drift-prone
      </div>
    </div>
  )
}
