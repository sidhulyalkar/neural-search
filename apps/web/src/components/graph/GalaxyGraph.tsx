import { useEffect, useRef } from 'react'
import * as THREE from 'three'
import type { GalaxyPoint } from '../../types/graph'

// OrbitControls is in three/examples — import path depends on @types/three version
// eslint-disable-next-line @typescript-eslint/ban-ts-comment
// @ts-ignore
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'

interface GalaxyGraphProps {
  points: GalaxyPoint[]
  onPointClick: (id: string, label: string) => void
}

export function GalaxyGraph({ points, onPointClick }: GalaxyGraphProps) {
  const mountRef = useRef<HTMLDivElement>(null)
  const clickRef = useRef(onPointClick)
  clickRef.current = onPointClick

  useEffect(() => {
    const mount = mountRef.current
    if (!mount || points.length === 0) return

    const width = mount.clientWidth
    const height = mount.clientHeight

    // Scene
    const scene = new THREE.Scene()
    const camera = new THREE.PerspectiveCamera(60, width / height, 0.1, 20000)
    camera.position.set(0, 0, 800)

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false })
    renderer.setPixelRatio(window.devicePixelRatio)
    renderer.setSize(width, height)
    renderer.setClearColor(0x020b14)
    mount.appendChild(renderer.domElement)

    // OrbitControls
    const controls = new OrbitControls(camera, renderer.domElement)
    controls.autoRotate = true
    controls.autoRotateSpeed = 0.4
    controls.enableDamping = true
    controls.dampingFactor = 0.05

    // Point cloud
    const geometry = new THREE.BufferGeometry()
    const positions = new Float32Array(points.length * 3)
    const colors = new Float32Array(points.length * 3)
    const sizes = new Float32Array(points.length)

    points.forEach((p, i) => {
      positions[i * 3] = p.x
      positions[i * 3 + 1] = p.y
      positions[i * 3 + 2] = p.z
      const c = new THREE.Color(p.color)
      colors[i * 3] = c.r
      colors[i * 3 + 1] = c.g
      colors[i * 3 + 2] = c.b
      sizes[i] = Math.max(2, p.size * 0.6)
    })

    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3))
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3))

    const material = new THREE.PointsMaterial({
      size: 4,
      vertexColors: true,
      transparent: true,
      opacity: 0.85,
      sizeAttenuation: true,
    })

    const cloud = new THREE.Points(geometry, material)
    scene.add(cloud)

    // Raycaster for click
    const raycaster = new THREE.Raycaster()
    raycaster.params.Points = { threshold: 8 }
    const mouse = new THREE.Vector2()

    const handleClick = (event: MouseEvent) => {
      const rect = renderer.domElement.getBoundingClientRect()
      mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1
      mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1
      raycaster.setFromCamera(mouse, camera)
      const hits = raycaster.intersectObject(cloud)
      if (hits.length > 0 && hits[0].index !== undefined) {
        const p = points[hits[0].index]
        clickRef.current(p.id, p.label)
      }
    }
    renderer.domElement.addEventListener('click', handleClick)

    // Resize
    const handleResize = () => {
      const w = mount.clientWidth
      const h = mount.clientHeight
      camera.aspect = w / h
      camera.updateProjectionMatrix()
      renderer.setSize(w, h)
    }
    window.addEventListener('resize', handleResize)

    // Animate
    let rafId: number
    const animate = () => {
      rafId = requestAnimationFrame(animate)
      controls.update()
      renderer.render(scene, camera)
    }
    animate()

    return () => {
      cancelAnimationFrame(rafId)
      window.removeEventListener('resize', handleResize)
      renderer.domElement.removeEventListener('click', handleClick)
      controls.dispose()
      renderer.dispose()
      geometry.dispose()
      material.dispose()
      if (mount.contains(renderer.domElement)) mount.removeChild(renderer.domElement)
    }
  }, [points])

  return <div ref={mountRef} className="w-full h-full" />
}
