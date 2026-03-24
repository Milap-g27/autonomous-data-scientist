'use client'

import { useState, useEffect, lazy, Suspense } from 'react'
import { Card } from "@/components/ui/card"
import { Spotlight } from "@/components/ui/spotlight"

const SplineScene = lazy(() => import('@splinetool/react-spline'))

export function SplineSceneBasic() {
  const [load3D, setLoad3D] = useState(false)

  useEffect(() => {
    // Defer Spline loading until after initial paint + idle time
    // This ensures FCP/LCP/TBT are measured before the heavy 3D runtime loads
    if ('requestIdleCallback' in window) {
      const id = requestIdleCallback(() => setLoad3D(true), { timeout: 3000 })
      return () => cancelIdleCallback(id)
    } else {
      const timer = setTimeout(() => setLoad3D(true), 1500)
      return () => clearTimeout(timer)
    }
  }, [])

  return (
    <Card className="w-full h-[500px] bg-black/[0.96] border-white/10 rounded-2xl relative overflow-hidden">
      <Spotlight
        className="-top-40 left-0 md:left-60 md:-top-20"
        fill="white"
      />

      <div className="flex h-full">
        {/* Left content */}
        <div className="flex-1 p-8 relative z-10 flex flex-col justify-center">
          <h1 className="text-4xl md:text-5xl font-bold bg-clip-text text-transparent bg-gradient-to-b from-neutral-50 to-neutral-400">
            AI Data Scientist
          </h1>
          <p className="mt-4 text-neutral-300 max-w-lg">
            Automate your entire data science workflow. Upload a dataset to clean data, engineer features, train models, and generate insights autonomously.
          </p>
        </div>

        {/* Right content */}
        <div className="flex-1 relative">
          {load3D ? (
            <Suspense
              fallback={
                <div className="w-full h-full flex items-center justify-center">
                  <div className="flex flex-col items-center gap-3">
                    <div className="w-8 h-8 border-2 border-neutral-600 border-t-neutral-200 rounded-full animate-spin" />
                    <span className="text-neutral-400 text-sm">Loading 3D Scene...</span>
                  </div>
                </div>
              }
            >
              <SplineScene
                scene="https://prod.spline.design/kZDDjO5HuC9GJUM2/scene.splinecode"
                className="w-full h-full"
              />
            </Suspense>
          ) : (
            <div className="w-full h-full flex items-center justify-center">
              <div className="relative w-48 h-48">
                <div className="absolute inset-0 rounded-full bg-gradient-to-br from-neutral-700/30 to-neutral-900/30 animate-pulse" />
                <div className="absolute inset-4 rounded-full bg-gradient-to-br from-neutral-600/20 to-neutral-800/20 animate-pulse [animation-delay:150ms]" />
                <div className="absolute inset-8 rounded-full bg-gradient-to-br from-neutral-500/10 to-neutral-700/10 animate-pulse [animation-delay:300ms]" />
                <div className="absolute inset-0 flex items-center justify-center">
                  <svg className="w-16 h-16 text-neutral-500" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1">
                    <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5" />
                  </svg>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </Card>
  )
}
