"use client"

import { useEffect, useState } from "react"
import { cn } from "@/lib/utils"

interface PredictionDisplayProps {
  value: number | null
}

export function PredictionDisplay({ value }: PredictionDisplayProps) {
  const [isAnimating, setIsAnimating] = useState(false)
  const [prevValue, setPrevValue] = useState<number | null>(null)

  useEffect(() => {
    if (value !== prevValue && value !== null) {
      setIsAnimating(true)
      const timer = setTimeout(() => setIsAnimating(false), 500)
      setPrevValue(value)
      return () => clearTimeout(timer)
    }
  }, [value, prevValue])

  return (
    <div className="flex flex-col items-center justify-center h-full min-h-[300px]">
      {value === null ? (
        <div className="text-muted-foreground text-xl">Waiting for prediction...</div>
      ) : (
        <>
          <div className="text-sm text-muted-foreground mb-4">Detected Brain Wave Pattern</div>
          <div
            className={cn(
              "text-8xl font-bold transition-all duration-500",
              isAnimating ? "scale-125 text-primary" : "scale-100",
            )}
          >
            {value}
          </div>
          <div className="mt-8 text-center text-muted-foreground">
            <p>This value represents the pattern detected by your ML model</p>
          </div>
        </>
      )}
    </div>
  )
}

