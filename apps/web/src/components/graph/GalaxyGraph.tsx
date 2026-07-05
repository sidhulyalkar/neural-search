import { useEffect, useRef, useCallback } from 'react'
import * as THREE from 'three'
// @ts-ignore
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import type { GalaxyPoint } from '../../types/graph'

interface GalaxyGraphProps {
  points: GalaxyPoint[]
  highlightIds?: Set<string>
  onPointClick: (id: string, label: string) => void
  onPointHover?: (point: GalaxyPoint | null) => void
}

export function GalaxyGraph({ points, highlightIds, onPointClick, onPointHover }: GalaxyGraphProps) {
  const mountRef = useRef<HTMLDivElement>(null)
  const labelCanvasRef = useRef<HTMLCanvasElement>(null)
  const clickRef = useRef(onPointClick)
  const hoverRef = useRef(onPointHover)
  const highlightRef = useRef(highlightIds)
  clickRef.current = onPointClick
  hoverRef.current = onPointHover
  highlightRef.current = highlightIds

  // Expose update function for highlights without remounting
  const sceneRef = useRef<{
    updateHighlights: (ids: Set<string> | undefined) => void
  } | null>(null)

  const setupScene = useCallback(() => {
    const mount = mountRef.current
    const labelCanvas = labelCanvasRef.current
    if (!mount || !labelCanvas || points.length === 0) return

    const width = mount.clientWidth
    const height = mount.clientHeight

    const scene = new THREE.Scene()
    const camera = new THREE.PerspectiveCamera(60, width / height, 0.1, 20000)
    camera.position.set(0, 0, 800)

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false })
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    renderer.setSize(width, height)
    renderer.setClearColor(0x020b14)
    mount.appendChild(renderer.domElement)

    const controls = new OrbitControls(camera, renderer.domElement)
    controls.autoRotate = true
    controls.autoRotateSpeed = 0.3
    controls.enableDamping = true
    controls.dampingFactor = 0.05
    controls.minDistance = 50
    controls.maxDistance = 3000

    // ── Point cloud geometry ────────────────────────────────────────────────
    const geo = new THREE.BufferGeometry()
    const positions = new Float32Array(points.length * 3)
    const baseColors = new Float32Array(points.length * 3)
    const activeColors = new Float32Array(points.length * 3)
    const sizes = new Float32Array(points.length)

    points.forEach((p, i) => {
      positions[i * 3] = p.x
      positions[i * 3 + 1] = p.y
      positions[i * 3 + 2] = p.z
      const c = new THREE.Color(p.color)
      baseColors[i * 3] = c.r
      baseColors[i * 3 + 1] = c.g
      baseColors[i * 3 + 2] = c.b
      activeColors[i * 3] = c.r
      activeColors[i * 3 + 1] = c.g
      activeColors[i * 3 + 2] = c.b
      sizes[i] = Math.max(2, (p.size ?? 4) * 0.6)
    })

    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3))
    const colorAttr = new THREE.BufferAttribute(activeColors.slice(), 3)
    geo.setAttribute('color', colorAttr)

    const mat = new THREE.PointsMaterial({
      size: 5,
      vertexColors: true,
      transparent: true,
      opacity: 0.9,
      sizeAttenuation: true,
    })

    const cloud = new THREE.Points(geo, mat)
    scene.add(cloud)

    // ── Highlight update ────────────────────────────────────────────────────
    const updateHighlights = (ids: Set<string> | undefined) => {
      const hasHighlight = ids && ids.size > 0
      for (let i = 0; i < points.length; i++) {
        const isHit = hasHighlight ? ids!.has(points[i].id) : true
        const factor = isHit ? 1.0 : 0.12
        colorAttr.array[i * 3] = baseColors[i * 3] * factor
        colorAttr.array[i * 3 + 1] = baseColors[i * 3 + 1] * factor
        colorAttr.array[i * 3 + 2] = baseColors[i * 3 + 2] * factor
      }
      colorAttr.needsUpdate = true
    }

    sceneRef.current = { updateHighlights }
    updateHighlights(highlightRef.current)

    // ── Label canvas ────────────────────────────────────────────────────────
    labelCanvas.width = width
    labelCanvas.height = height
    labelCanvas.style.width = width + 'px'
    labelCanvas.style.height = height + 'px'
    const ctx2d = labelCanvas.getContext('2d')!

    // Project a 3D point to 2D screen coordinates
    const project = (x: number, y: number, z: number): [number, number, boolean] => {
      const v = new THREE.Vector3(x, y, z)
      v.project(camera)
      const sx = (v.x + 1) / 2 * width
      const sy = (-v.y + 1) / 2 * height
      const inFront = v.z < 1
      return [sx, sy, inFront]
    }

    // ── Raycaster ──────────────────────────────────────────────────────────
    const raycaster = new THREE.Raycaster()
    raycaster.params.Points = { threshold: 10 }
    const mouse = new THREE.Vector2(-9999, -9999)
    let hoveredIndex = -1

    const handleMouseMove = (e: MouseEvent) => {
      const rect = renderer.domElement.getBoundingClientRect()
      mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1
      mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1
    }

    const handleClick = (e: MouseEvent) => {
      const rect = renderer.domElement.getBoundingClientRect()
      const m = new THREE.Vector2(
        ((e.clientX - rect.left) / rect.width) * 2 - 1,
        -((e.clientY - rect.top) / rect.height) * 2 + 1
      )
      raycaster.setFromCamera(m, camera)
      const hits = raycaster.intersectObject(cloud)
      if (hits.length > 0 && hits[0].index !== undefined) {
        const p = points[hits[0].index]
        clickRef.current(p.id, p.label)
      }
    }

    renderer.domElement.addEventListener('mousemove', handleMouseMove)
    renderer.domElement.addEventListener('click', handleClick)
    renderer.domElement.style.cursor = 'default'

    // ── Animate ─────────────────────────────────────────────────────────────
    let rafId: number
    const tmpVec = new THREE.Vector3()

    const drawLabels = () => {
      if (!ctx2d) return
      ctx2d.clearRect(0, 0, width, height)

      const ids = highlightRef.current
      const hasHighlight = ids && ids.size > 0

      // Raycast for hover
      raycaster.setFromCamera(mouse, camera)
      const hits = raycaster.intersectObject(cloud)
      const newHovered = hits.length > 0 && hits[0].index !== undefined ? hits[0].index : -1
      if (newHovered !== hoveredIndex) {
        hoveredIndex = newHovered
        renderer.domElement.style.cursor = newHovered >= 0 ? 'pointer' : 'default'
        hoverRef.current?.(newHovered >= 0 ? points[newHovered] : null)
      }

      // Draw label for hovered point
      if (hoveredIndex >= 0) {
        const p = points[hoveredIndex]
        const [sx, sy, inFront] = project(p.x, p.y, p.z)
        if (inFront) {
          drawLabel(ctx2d, p.label, sx, sy, p.color, 14, true)
        }
      }

      // Draw labels for highlighted points (top 30 by size)
      if (hasHighlight) {
        const toLabel = points
          .filter(p => ids!.has(p.id) && p.id !== (hoveredIndex >= 0 ? points[hoveredIndex].id : ''))
          .sort((a, b) => (b.size ?? 0) - (a.size ?? 0))
          .slice(0, 30)

        for (const p of toLabel) {
          const [sx, sy, inFront] = project(p.x, p.y, p.z)
          if (inFront && sx > 0 && sx < width && sy > 0 && sy < height) {
            drawLabel(ctx2d, p.label, sx, sy, p.color, 11, false)
          }
        }
      } else {
        // No filter: label the N largest points near camera
        const camPos = camera.position
        const byDist = points
          .map((p, i) => {
            tmpVec.set(p.x, p.y, p.z)
            return { p, i, dist: tmpVec.distanceTo(camPos) }
          })
          .filter(({ dist }) => dist < 600)
          .sort((a, b) => (b.p.size ?? 0) - (a.p.size ?? 0))
          .slice(0, 20)

        for (const { p, i } of byDist) {
          if (i === hoveredIndex) continue
          const [sx, sy, inFront] = project(p.x, p.y, p.z)
          if (inFront && sx > 0 && sx < width && sy > 0 && sy < height) {
            drawLabel(ctx2d, p.label, sx, sy, p.color, 10, false)
          }
        }
      }
    }

    const animate = () => {
      rafId = requestAnimationFrame(animate)
      controls.update()
      renderer.render(scene, camera)
      drawLabels()
    }
    animate()

    // ── Resize ─────────────────────────────────────────────────────────────
    const handleResize = () => {
      const w = mount.clientWidth
      const h = mount.clientHeight
      camera.aspect = w / h
      camera.updateProjectionMatrix()
      renderer.setSize(w, h)
      labelCanvas.width = w
      labelCanvas.height = h
      labelCanvas.style.width = w + 'px'
      labelCanvas.style.height = h + 'px'
    }
    window.addEventListener('resize', handleResize)

    return () => {
      cancelAnimationFrame(rafId)
      window.removeEventListener('resize', handleResize)
      renderer.domElement.removeEventListener('mousemove', handleMouseMove)
      renderer.domElement.removeEventListener('click', handleClick)
      controls.dispose()
      renderer.dispose()
      geo.dispose()
      mat.dispose()
      sceneRef.current = null
      if (mount.contains(renderer.domElement)) mount.removeChild(renderer.domElement)
    }
  }, [points])

  // Mount Three.js scene once when points are available
  useEffect(() => {
    const cleanup = setupScene()
    return cleanup ?? undefined
  }, [setupScene])

  // Update highlights without remounting
  useEffect(() => {
    sceneRef.current?.updateHighlights(highlightIds)
  }, [highlightIds])

  return (
    <div ref={mountRef} className="w-full h-full relative">
      <canvas
        ref={labelCanvasRef}
        className="absolute inset-0 pointer-events-none"
        style={{ zIndex: 1 }}
      />
    </div>
  )
}

function drawLabel(
  ctx: CanvasRenderingContext2D,
  text: string,
  x: number,
  y: number,
  color: string,
  fontSize: number,
  bold: boolean,
) {
  const label = text.length > 28 ? text.slice(0, 26) + '…' : text
  ctx.save()
  ctx.font = `${bold ? '600' : '400'} ${fontSize}px Inter, system-ui, sans-serif`
  const tw = ctx.measureText(label).width
  const pad = 4
  const bx = x + 8
  const by = y - fontSize / 2 - pad

  // Pill background
  ctx.fillStyle = 'rgba(2,11,20,0.82)'
  ctx.beginPath()
  ctx.roundRect(bx - pad, by, tw + pad * 2, fontSize + pad * 2, 4)
  ctx.fill()

  // Colour accent border
  ctx.strokeStyle = color + '80'
  ctx.lineWidth = 1
  ctx.stroke()

  // Text
  ctx.fillStyle = bold ? '#f0f9ff' : '#94a3b8'
  ctx.fillText(label, bx, by + fontSize + pad - 2)
  ctx.restore()
}
