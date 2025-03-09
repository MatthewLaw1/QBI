"use client"

import { useEffect } from 'react'
import { Line } from 'react-chartjs-2'
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  ChartOptions
} from 'chart.js'

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
)

interface EEGGraphProps {
  data: number[][]
  timeWindow: number
}

export function EEGGraph({ data, timeWindow }: EEGGraphProps) {
  // Define colors for each channel
  const channelColors = [
    'rgba(255, 99, 132, 1)',   // Red - Channel 1 (TP9)
    'rgba(54, 162, 235, 1)',   // Blue - Channel 2 (FP1)
    'rgba(255, 206, 86, 1)',   // Yellow - Channel 3 (FP2)
    'rgba(75, 192, 192, 1)'    // Green - Channel 4 (TP10)
  ]

  // Define vertical offsets for each channel to create separate subgraphs
  // Each channel has a range of -500 to +500, with 1000 units of separation between channels
  const channelOffsets = [1500, 500, -500, -1500]  // Centered at these values
  
  // Channel names - updated to match the correct order
  const channelNames = ["TP9", "FP1", "FP2", "TP10"]

  // Generate labels (time points)
  const labels = Array.from({ length: data.length }, (_, i) => i.toString())

  // Scale factor to ensure each channel stays within its -500 to +500 range
  const scaleData = (value: number, channelIndex: number) => {
    // Clamp the value between -500 and 500 before adding offset
    const clampedValue = Math.max(-500, Math.min(500, value));
    return clampedValue + channelOffsets[channelIndex];
  }

  // Prepare datasets (one for each channel)
  const datasets = data.length > 0 && data[0].length > 0 
    ? Array.from({ length: data[0].length }, (_, channelIndex) => ({
        label: channelNames[channelIndex],
        data: data.map(point => scaleData(point[channelIndex] || 0, channelIndex)),
        borderColor: channelColors[channelIndex],
        backgroundColor: 'transparent',
        borderWidth: 1.5,
        pointRadius: 0,
        tension: 0.2,
        fill: false,
      }))
    : []

  const chartData = {
    labels,
    datasets,
  }

  const options: ChartOptions<'line'> = {
    responsive: true,
    maintainAspectRatio: false,
    animation: {
      duration: 0 // Disable animation for performance
    },
    scales: {
      x: {
        display: true,
        title: {
          display: true,
          text: 'Time'
        },
        ticks: {
          maxTicksLimit: 10,
        },
        grid: {
          display: false
        }
      },
      y: {
        display: true,
        title: {
          display: true,
          text: 'Channels'
        },
        min: -2000,  // Lowest channel offset (-1500) minus range (500)
        max: 2000,   // Highest channel offset (1500) plus range (500)
        grid: {
          color: (context) => {
            // Draw horizontal lines at each channel's center and boundaries
            if (context.tick && channelOffsets.includes(context.tick.value)) {
              return 'rgba(0, 0, 0, 0.2)'; // Center line (darker)
            }
            // Draw lighter lines at the +500/-500 boundaries of each channel
            if (context.tick && channelOffsets.map(offset => offset + 500).includes(context.tick.value) ||
                context.tick && channelOffsets.map(offset => offset - 500).includes(context.tick.value)) {
              return 'rgba(0, 0, 0, 0.1)'; // Boundary lines
            }
            return 'rgba(0, 0, 0, 0.05)';
          }
        },
        ticks: {
          // Custom ticks to show channel names and their ranges
          callback: function(value) {
            // Show channel name at the center of each channel's range
            const centerIndex = channelOffsets.indexOf(Number(value));
            if (centerIndex !== -1) {
              return channelNames[centerIndex];
            }
            
            // Show +500/-500 at the boundaries of each channel
            for (let i = 0; i < channelOffsets.length; i++) {
              if (value === channelOffsets[i] + 500) return '+500';
              if (value === channelOffsets[i] - 500) return '-500';
            }
            
            return '';
          },
          stepSize: 500
        }
      }
    },
    plugins: {
      legend: {
        position: 'top',
      },
      tooltip: {
        enabled: true,
        mode: 'index',
        intersect: false,
        callbacks: {
          label: function(context) {
            const datasetIndex = context.datasetIndex;
            // Subtract the offset to get the actual value
            const value = context.parsed.y - channelOffsets[datasetIndex];
            return `${channelNames[datasetIndex]}: ${value.toFixed(2)} μV`;
          }
        }
      }
    },
  }

  return (
    <div className="w-full h-full">
      <Line data={chartData} options={options} />
    </div>
  )
}

