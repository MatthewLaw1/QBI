"use client"

import { useEffect, useState } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { EEGGraph } from "@/components/eeg-graph"
import { PredictionDisplay } from "@/components/prediction-display"

export default function Home() {
  const [eegData, setEegData] = useState<number[][]>([])
  const [prediction, setPrediction] = useState<number | null>(null)

  // This is a placeholder for your actual data fetching logic
  // You'll need to replace this with your actual implementation to connect to the Muse 2 device
  useEffect(() => {
    // Simulate incoming EEG data for demonstration
    const interval = setInterval(() => {
      // Generate random data for 4 EEG channels
      const newDataPoint = [
        Math.sin(Date.now() / 1000) * 50 + Math.random() * 20,
        Math.sin(Date.now() / 1000 + 1) * 40 + Math.random() * 15,
        Math.sin(Date.now() / 1000 + 2) * 30 + Math.random() * 10,
        Math.sin(Date.now() / 1000 + 3) * 20 + Math.random() * 5,
      ]

      setEegData((prevData) => {
        // Keep only the last 100 data points
        const newData = [...prevData, newDataPoint]
        if (newData.length > 100) {
          return newData.slice(-100)
        }
        return newData
      })

      // Simulate ML prediction (replace with your actual backend call)
      if (Math.random() > 0.9) {
        // Occasionally update the prediction
        setPrediction(Math.floor(Math.random() * 10))
      }
    }, 100)

    return () => clearInterval(interval)
  }, [])

  return (
    <main className="flex min-h-screen flex-col p-4 md:p-8">
      <h1 className="text-2xl font-bold mb-6">EEG Brain Wave Analysis</h1>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 flex-1 w-full">
        <Card className="lg:col-span-2 overflow-hidden flex flex-col h-full">
          <CardHeader className="pb-2">
            <CardTitle>EEG Data Stream</CardTitle>
          </CardHeader>
          <CardContent className="flex-grow">
            <div className="w-full h-full min-h-[500px] p-4">
              <EEGGraph data={eegData} timeWindow={50} />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>ML Prediction</CardTitle>
          </CardHeader>
          <CardContent>
            <PredictionDisplay value={prediction} />
          </CardContent>
        </Card>
      </div>
    </main>
  )
}

