import { useCallback, useRef } from 'react'
import ForceGraph2D from 'react-force-graph-2d'
import type { GraphData, GraphEdge, GraphNode } from '../../types/graph'

interface ExplorerGraphProps {
  graphData: GraphData
  mode: '3d' | '2d'
  onNodeClick: (node: GraphNode) => void
  highlightIds?: Set<string>
}

const NODE_REL_SIZE = 4

export function ExplorerGraph({ graphData, onNodeClick, highlightIds }: ExplorerGraphProps) {
  const fgRef = useRef<unknown>(null)

  const handleNodeClick = useCallback(
    (node: object) => {
      onNodeClick(node as GraphNode)
    },
    [onNodeClick],
  )

  const nodeColor = useCallback(
    (node: object) => {
      const n = node as GraphNode
      if (highlightIds && highlightIds.size > 0) {
        return highlightIds.has(n.id) ? n.color : `${n.color}44`
      }
      return n.color
    },
    [highlightIds],
  )

  const nodeVal = useCallback((node: object) => (node as GraphNode).size, [])
  const linkColor = useCallback((link: object) => (link as GraphEdge).color, [])
  const linkWidth = useCallback((link: object) => (link as GraphEdge).weight * 0.8, [])
  const nodeLabel = useCallback((node: object) => (node as GraphNode).label, [])

  const commonProps = {
    ref: fgRef,
    graphData,
    nodeId: 'id' as const,
    nodeLabel,
    nodeColor,
    nodeVal,
    linkColor,
    linkWidth,
    linkDirectionalParticles: 1,
    linkDirectionalParticleSpeed: 0.004,
    onNodeClick: handleNodeClick,
    backgroundColor: '#020b14',
    nodeRelSize: NODE_REL_SIZE,
  }

  // 3D mode falls back to 2D (ForceGraph3D requires A-Frame which crashes in Vite/ESM)
  return (
    // @ts-expect-error react-force-graph-2d types
    <ForceGraph2D
      {...commonProps}
      nodeCanvasObject={(node, ctx, globalScale) => {
        const n = node as GraphNode & { x?: number; y?: number }
        if (n.x === undefined || n.y === undefined) return
        const r = Math.sqrt(n.size || 0) * NODE_REL_SIZE
        ctx.arc(n.x, n.y, r, 0, 2 * Math.PI)
        ctx.fillStyle = nodeColor(node)
        ctx.fill()
        if (globalScale > 2 && n.label) {
          ctx.font = `${10 / globalScale}px Inter, sans-serif`
          ctx.fillStyle = '#e2e8f0'
          ctx.textAlign = 'center'
          ctx.fillText(n.label.slice(0, 20), n.x, n.y + r + 4 / globalScale)
        }
      }}
    />
  )
}
