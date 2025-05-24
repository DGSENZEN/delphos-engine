"use client"

import { useEffect, useRef } from "react"

interface OrbitalBackgroundProps {
  scrollY: number
}

export default function OrbitalBackground({ scrollY }: OrbitalBackgroundProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext("2d")
    if (!ctx) return

    // Create noise texture
    let noiseCanvas: HTMLCanvasElement | null = null
    let noiseCtx: CanvasRenderingContext2D | null = null

    const createNoiseTexture = () => {
      noiseCanvas = document.createElement("canvas")
      noiseCanvas.width = 256
      noiseCanvas.height = 256
      noiseCtx = noiseCanvas.getContext("2d")

      if (!noiseCtx) return

      // Create noise pattern
      const imageData = noiseCtx.createImageData(noiseCanvas.width, noiseCanvas.height)
      const data = imageData.data

      for (let i = 0; i < data.length; i += 4) {
        const value = Math.floor(Math.random() * 255)
        data[i] = value // r
        data[i + 1] = value // g
        data[i + 2] = value // b
        data[i + 3] = 10 // alpha (very subtle)
      }

      noiseCtx.putImageData(imageData, 0, 0)
    }

    createNoiseTexture()

    // Draw noise overlay
    const drawNoise = () => {
      if (!ctx || !noiseCanvas) return

      // Create pattern and fill canvas
      const pattern = ctx.createPattern(noiseCanvas, "repeat")
      if (pattern) {
        ctx.fillStyle = pattern
        ctx.globalCompositeOperation = "overlay"
        ctx.fillRect(0, 0, canvas.width, canvas.height)
        ctx.globalCompositeOperation = "source-over"
      }
    }

    // Set canvas dimensions
    const resizeCanvas = () => {
      canvas.width = window.innerWidth
      canvas.height = window.innerHeight
      drawNoise()
    }

    resizeCanvas()
    window.addEventListener("resize", resizeCanvas)

    // Orbital system parameters
    const orbits = [
      { radius: 150, width: 1, color: "rgba(249, 115, 22, 0.15)", dashArray: [] },
      { radius: 250, width: 1, color: "rgba(20, 184, 166, 0.15)", dashArray: [] },
      { radius: 350, width: 1, color: "rgba(96, 165, 250, 0.15)", dashArray: [] },
      { radius: 450, width: 1, color: "rgba(249, 115, 22, 0.15)", dashArray: [1, 3] },
      { radius: 550, width: 1, color: "rgba(20, 184, 166, 0.15)", dashArray: [1, 3] },
    ]

    const planets = [
      {
        orbitIndex: 0,
        size: 6,
        color: "#f97316",
        angle: 0,
        speed: 0.0004,
        trail: true,
        trailLength: 20,
        trailFade: true,
      },
      {
        orbitIndex: 1,
        size: 8,
        color: "#14b8a6",
        angle: Math.PI / 3,
        speed: 0.0003,
        trail: false,
      },
      {
        orbitIndex: 2,
        size: 5,
        color: "#f97316",
        angle: Math.PI,
        speed: 0.0002,
        trail: true,
        trailLength: 15,
        trailFade: true,
      },
      {
        orbitIndex: 3,
        size: 7,
        color: "#60a5fa",
        angle: Math.PI / 2,
        speed: 0.00015,
        trail: false,
      },
      {
        orbitIndex: 4,
        size: 9,
        color: "#f97316",
        angle: Math.PI * 1.5,
        speed: 0.0001,
        trail: true,
        trailLength: 25,
        trailFade: true,
      },
    ]

    // Create fixed set of particles
    const particles: {
      x: number
      y: number
      size: number
      speed: number
      color: string
      opacity: number
      direction: number
      initialY: number
    }[] = []

    // Create particles once
    const createParticles = (count: number) => {
      for (let i = 0; i < count; i++) {
        const x = Math.random() * canvas.width
        const y = Math.random() * canvas.height
        particles.push({
          x,
          y,
          initialY: y, // Store initial Y position for parallax
          size: Math.random() * 1.5,
          speed: Math.random() * 0.2,
          color: Math.random() > 0.5 ? "#f97316" : Math.random() > 0.5 ? "#14b8a6" : "#60a5fa",
          opacity: Math.random() * 0.5 + 0.1,
          direction: Math.random() * Math.PI * 2,
        })
      }
    }

    createParticles(100)

    // Grid parameters
    const gridSize = 40
    const gridOpacity = 0.05

    // Planet trail history
    const planetTrails: { x: number; y: number; alpha: number }[][] = planets.map(() => [])

    // Animation loop
    let animationFrameId: number
    let lastTime = 0

    const render = (time: number) => {
      const deltaTime = time - lastTime
      lastTime = time

      ctx.clearRect(0, 0, canvas.width, canvas.height)

      // Calculate viewport center for animation focus
      const viewportHeight = window.innerHeight
      const scrollProgress = Math.min(1, scrollY / (viewportHeight * 0.5))

      // Center point for the orbital system - fixed at center of viewport
      const centerX = canvas.width / 2
      const centerY = viewportHeight / 2

      // Scale orbits based on viewport size
      const scale = Math.min(1, viewportHeight / 1000)

      // Draw grid
      ctx.strokeStyle = `rgba(226, 232, 240, ${gridOpacity})`
      ctx.lineWidth = 0.5

      // Vertical lines
      for (let x = 0; x < canvas.width; x += gridSize) {
        ctx.beginPath()
        ctx.moveTo(x, 0)
        ctx.lineTo(x, canvas.height)
        ctx.stroke()
      }

      // Horizontal lines
      for (let y = 0; y < canvas.height; y += gridSize) {
        ctx.beginPath()
        ctx.moveTo(0, y)
        ctx.lineTo(canvas.width, y)
        ctx.stroke()
      }

      // Draw particles with subtle parallax
      particles.forEach((particle) => {
        // Update particle position
        particle.x += Math.cos(particle.direction) * particle.speed
        particle.y += Math.sin(particle.direction) * particle.speed

        // Wrap particles around screen
        if (particle.x < 0) particle.x = canvas.width
        if (particle.x > canvas.width) particle.x = 0
        if (particle.y < 0) particle.y = canvas.height
        if (particle.y > canvas.height) particle.y = 0

        // Draw particle with glow
        ctx.beginPath()
        ctx.arc(particle.x, particle.y, particle.size, 0, Math.PI * 2)
        ctx.fillStyle = particle.color.replace(")", `, ${particle.opacity * 1.5})`).replace("rgb", "rgba")
        ctx.fill()

        // Add subtle glow to larger particles
        if (particle.size > 1) {
          const particleGradient = ctx.createRadialGradient(
            particle.x,
            particle.y,
            particle.size,
            particle.x,
            particle.y,
            particle.size * 3,
          )
          particleGradient.addColorStop(0, particle.color.replace(")", ", 0.3)").replace("rgb", "rgba"))
          particleGradient.addColorStop(1, "rgba(0, 0, 0, 0)")

          ctx.beginPath()
          ctx.arc(particle.x, particle.y, particle.size * 3, 0, Math.PI * 2)
          ctx.fillStyle = particleGradient
          ctx.fill()
        }
      })

      // Draw orbits
      orbits.forEach((orbit) => {
        ctx.beginPath()
        ctx.arc(centerX, centerY, orbit.radius * scale, 0, Math.PI * 2)
        ctx.strokeStyle = orbit.color
        ctx.lineWidth = orbit.width

        if (orbit.dashArray.length) {
          ctx.setLineDash(orbit.dashArray)
        } else {
          ctx.setLineDash([])
        }

        ctx.stroke()
        ctx.setLineDash([])
      })

      // Update and draw planets
      planets.forEach((planet, index) => {
        // Update planet position
        planet.angle += planet.speed * (deltaTime / 16)

        // Calculate position
        const orbit = orbits[planet.orbitIndex]
        const x = centerX + Math.cos(planet.angle) * orbit.radius * scale
        const y = centerY + Math.sin(planet.angle) * orbit.radius * scale

        // Update trail
        if (planet.trail) {
          const trailLength = planet.trailLength || 10

          // Add current position to trail
          planetTrails[index].unshift({ x, y, alpha: 1 })

          // Limit trail length
          if (planetTrails[index].length > trailLength) {
            planetTrails[index].pop()
          }

          // Draw trail
          planetTrails[index].forEach((point, i) => {
            const alpha = planet.trailFade ? 1 - i / trailLength : 0.5

            point.alpha = alpha

            ctx.beginPath()
            ctx.arc(point.x, point.y, planet.size * (1 - i / trailLength), 0, Math.PI * 2)
            ctx.fillStyle = planet.color.replace(")", `, ${alpha * 0.3})`).replace("rgb", "rgba")
            ctx.fill()
          })
        }

        // Draw planet
        ctx.beginPath()
        ctx.arc(x, y, planet.size, 0, Math.PI * 2)
        ctx.fillStyle = planet.color
        ctx.fill()

        // Draw subtle glow
        const gradient = ctx.createRadialGradient(x, y, planet.size, x, y, planet.size * 4)
        gradient.addColorStop(0, planet.color.replace(")", ", 0.4)").replace("rgb", "rgba"))
        gradient.addColorStop(1, "rgba(0, 0, 0, 0)")

        ctx.beginPath()
        ctx.arc(x, y, planet.size * 4, 0, Math.PI * 2)
        ctx.fillStyle = gradient
        ctx.fill()
      })

      // Draw noise overlay
      drawNoise()

      animationFrameId = requestAnimationFrame(render)
    }

    render(0)

    return () => {
      window.removeEventListener("resize", resizeCanvas)
      cancelAnimationFrame(animationFrameId)
    }
  }, [scrollY]) // Add scrollY as a dependency

  return (
    <>
      <canvas ref={canvasRef} className="fixed inset-0 z-0 bg-slate-900" />
      <div className="fixed inset-0 z-0 bg-noise opacity-5 pointer-events-none"></div>
    </>
  )
}
