"use client"

import { useEffect, useState } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { EEGGraph } from "@/components/eeg-graph"
import { PredictionDisplay } from "@/components/prediction-display"

export default function Home() {
  const [eegData, setEegData] = useState<number[][]>([])
  const [prediction, setPrediction] = useState<number | null>(null)
  const [isConnected, setIsConnected] = useState(false)

  // Connect to the SSE server to receive real-time EEG data
  useEffect(() => {
    // Create EventSource connection to the backend SSE endpoint
    const eventSource = new EventSource('http://localhost:8765/eeg-stream')
    
    // Handle connection open
    eventSource.addEventListener('connected', (event) => {
      console.log('Connected to EEG stream:', event.data)
      setIsConnected(true)
    })
    
    // Handle incoming EEG data
    eventSource.addEventListener('eeg', (event) => {
      try {
        const parsedData = JSON.parse(event.data)
        const newDataPoint = parsedData.eeg
        
        // Update the EEG data state
        setEegData((prevData) => {
          // Add the new data point
          const newData = [...prevData, newDataPoint]
          
          // Keep only the last 500 data points
          if (newData.length > 500) {
            return newData.slice(-500)
          }
          
          // If we have less than 500 data points, pad with zeros at the beginning
          if (newData.length < 500) {
            const paddingNeeded = 500 - newData.length
            const padding = Array(paddingNeeded).fill([0, 0, 0, 0])
            return [...padding, ...newData]
          }
          
          return newData
        })
        
      } catch (error) {
        console.error('Error parsing EEG data:', error)
      }
    })
    
    // Handle errors
    eventSource.onerror = (error) => {
      console.error('EventSource error:', error)
      setIsConnected(false)
      
      // Try to reconnect after a delay
      setTimeout(() => {
        eventSource.close()
        // The browser will automatically try to reconnect
      }, 5000)
    }
    
    // Clean up on component unmount
    return () => {
      eventSource.close()
    }
  }, [])

  // Poll the backend API for predictions
  useEffect(() => {
    // Only start polling when we have EEG data
    if (eegData.length > 0) {
      console.log("Starting prediction polling")
      
      // Poll for predictions every second
      const interval = setInterval(() => {
        fetch('http://localhost:8000/prediction')
          .then(response => {
            if (!response.ok) {
              throw new Error(`HTTP error! Status: ${response.status}`)
            }
            return response.json()
          })
          .then(data => {
            if (data.predicted_number !== null) {
              console.log("Received prediction:", data.predicted_number)
              setPrediction(data.predicted_number)
            }
          })
          .catch(error => {
            console.error('Error fetching prediction:', error)
          })
      }, 5000)
      
      // Clean up interval on unmount
      return () => clearInterval(interval)
    }
  }, [eegData.length])

  return (
    <main className="flex min-h-screen flex-col p-4 md:p-8">
      <h1 className="text-2xl font-bold mb-6">EEG Brain Wave Analysis</h1>
      
      <div className="mb-4">
        <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
          isConnected ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'
        }`}>
          {isConnected ? 'Connected to EEG Stream' : 'Disconnected'}
        </span>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 flex-1 w-full">
        <Card className="lg:col-span-2 overflow-hidden flex flex-col h-full">
          <CardHeader className="pb-2">
            <CardTitle>EEG Data Stream</CardTitle>
          </CardHeader>
          <CardContent className="flex-grow">
            <div className="w-full h-full min-h-[500px] p-4">
              <EEGGraph data={eegData} timeWindow={500} />
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

