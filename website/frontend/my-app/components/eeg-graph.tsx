"use client"

import { useEffect, useState } from "react"
import { Line, LineChart, XAxis, YAxis } from "recharts"
import { ChartContainer, ChartTooltip, ChartTooltipContent } from "@/components/ui/chart"

interface EEGGraphProps {
  data: number[][]
  timeWindow?: number // Number of data points to display
}

export function EEGGraph({ data, timeWindow = 50 }: EEGGraphProps) {
  const [formattedData, setFormattedData] = useState<any[]>([])

  useEffect(() => {
    // Only take the most recent 'timeWindow' data points
    const recentData = data.slice(-timeWindow)

    // Transform the data into the format expected by Recharts
    const formatted = recentData.map((point, index) => ({
      index,
      channel1: point[0],
      channel2: point[1],
      channel3: point[2],
      channel4: point[3],
    }))

    setFormattedData(formatted)
  }, [data, timeWindow])

  if (data.length === 0) {
    return (
      <div className="flex items-center justify-center h-[400px] border rounded-lg">
        <p className="text-muted-foreground">Waiting for EEG data...</p>
      </div>
    )
  }

  // Fixed Y-axis domain to prevent dynamic scaling
  const yDomain = [-100, 100]

  return (
    <div className="w-full h-[400px]">
      <ChartContainer
        config={{
          channel1: {
            label: "Channel 1",
            color: "hsl(var(--chart-1))",
          },
          channel2: {
            label: "Channel 2",
            color: "hsl(var(--chart-2))",
          },
          channel3: {
            label: "Channel 3",
            color: "hsl(var(--chart-3))",
          },
          channel4: {
            label: "Channel 4",
            color: "hsl(var(--chart-4))",
          },
        }}
        className="h-full"
      >
        <LineChart data={formattedData} margin={{ top: 20, right: 20, left: 20, bottom: 20 }}>
          <XAxis
            dataKey="index"
            tick={false}
            label={{ value: `Last ${timeWindow} samples`, position: "insideBottom", offset: -10 }}
          />
          <YAxis domain={yDomain} label={{ value: "Amplitude (μV)", angle: -90, position: "insideLeft" }} />
          <ChartTooltip content={<ChartTooltipContent />} />
          <Line
            type="monotone"
            dataKey="channel1"
            stroke="var(--color-channel1)"
            dot={false}
            isAnimationActive={false}
          />
          <Line
            type="monotone"
            dataKey="channel2"
            stroke="var(--color-channel2)"
            dot={false}
            isAnimationActive={false}
          />
          <Line
            type="monotone"
            dataKey="channel3"
            stroke="var(--color-channel3)"
            dot={false}
            isAnimationActive={false}
          />
          <Line
            type="monotone"
            dataKey="channel4"
            stroke="var(--color-channel4)"
            dot={false}
            isAnimationActive={false}
          />
        </LineChart>
      </ChartContainer>
    </div>
  )
}

