import time
import asyncio
import json
from aiohttp import web
from pylsl import StreamInlet, resolve_streams
from pythonosc import udp_client
from pythonosc.osc_message_builder import OscMessageBuilder
import aiohttp

# OSC Setup - Set the IP and port of TouchDesigner
osc_ip = '127.0.0.1'  # Localhost
osc_port = 7000        # Port to receive OSC data in TouchDesigner
client = udp_client.SimpleUDPClient(osc_ip, osc_port)

# Store for SSE clients
sse_clients = set()

# EEG data queue for sharing data between coroutines
eeg_queue = asyncio.Queue()

# Blink Detection Settings
blink_threshold = -200.0  # Adjust based on actual EEG data
blink_channels = [1, 2]  # Indexes of channels where blinks occur
blink_detected = False

# SSE endpoint handler
async def sse_handler(request):
    try:
        # Create a StreamResponse object for SSE
        response = web.StreamResponse(
            status=200,
            reason='OK',
            headers={
                'Content-Type': 'text/event-stream',
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
                'Access-Control-Allow-Origin': '*',  # For CORS, adjust as needed
            }
        )
        
        # Prepare the response - this sends the headers
        await response.prepare(request)
        
        # Create a queue for this client
        client_queue = asyncio.Queue()
        sse_clients.add(client_queue)
        
        # Write initial connection message
        await response.write(b'event: connected\ndata: Connected to EEG stream\n\n')
        print(f"Client connected, total clients: {len(sse_clients)}")
        
        try:
            while True:
                # Get data from the client queue
                data = await client_queue.get()
                
                # Format as SSE message
                message = f"event: eeg\ndata: {data}\n\n"
                await response.write(message.encode('utf-8'))
                
                # Yield control to allow other coroutines to run
                await asyncio.sleep(0)
        except ConnectionResetError:
            print("Client disconnected (connection reset)")
        except Exception as e:
            print(f"Error in SSE handler loop: {str(e)}")
        finally:
            # Remove client when disconnected
            if client_queue in sse_clients:
                sse_clients.remove(client_queue)
                print(f"Client removed, remaining clients: {len(sse_clients)}")
        
        return response
    except Exception as e:
        print(f"Error in SSE handler: {str(e)}")
        import traceback
        traceback.print_exc()
        return web.Response(status=500, text=f"Internal Server Error: {str(e)}")

# Function to broadcast EEG data to all SSE clients
async def broadcast_eeg_data():
    while True:
        # Get data from the queue
        eeg_data, timestamp, has_blink = await eeg_queue.get()
        
        # Create a JSON message with EEG data, timestamp, and blink status
        message = json.dumps({
            "eeg": eeg_data,
            "timestamp": timestamp,
            "blink": 1 if has_blink else 0
        })
        
        # Send to all connected clients
        for client_queue in list(sse_clients):
            try:
                await client_queue.put(message)
            except Exception:
                # If there's an error, we'll remove this client
                if client_queue in sse_clients:
                    sse_clients.remove(client_queue)

# Main function to process EEG data with minimal buffering
async def process_eeg_data():
    try:
        # Resolve the stream from MuseLSL (looking for EEG stream)
        print("Looking for an EEG stream...")
        streams = resolve_streams(5.0)
        
        if not streams:
            print("No EEG stream found! Make sure your Muse headset is connected.")
            # Keep trying to find streams
            while not streams:
                await asyncio.sleep(5)
                print("Looking for an EEG stream again...")
                streams = resolve_streams(5.0)
        
        print(f"Found {len(streams)} stream(s). Using the first one: {streams[0].name()}")
        
        # Create an inlet with minimal buffering for real-time performance
        inlet = StreamInlet(streams[0], max_buflen=1, max_chunklen=1)
        
        sample_count = 0
        blink_detected = False
        last_2000_samples = []
        while True:
            try:
                # Get the sample with minimal timeout for real-time performance
                sample, timestamp = inlet.pull_sample(timeout=0.0)
                
                if sample:
                    sample_count += 1
                    if sample_count % 100 == 0:  # Log every 100th sample to avoid console spam
                        print(f"Sample #{sample_count}: {sample}")

                    if len(last_2000_samples) >= 2000 : 
                        print(f"Sending batch of {len(last_2000_samples)} samples to prediction server")
                        
                        # Send the batch to the prediction API
                        try:
                            # Reorganize data to match EEGData format
                            # We need to transpose the data from time-based to channel-based
                            
                            # Extract EEG samples from all recordings
                            all_eeg_samples = [s[0] for s in last_2000_samples]
                            
                            # Organize by channel (4 channels)
                            channels = [[], [], [], []]
                            for sample in all_eeg_samples:
                                for i in range(len(sample)):
                                    if i < len(channels):
                                        channels[i].append(sample[i])
                            
                            # Create data in the format expected by EEGData
                            batch_data = {
                                "channels": channels,
                                # Optional fields
                                "timestamp": last_2000_samples[-1][1] if last_2000_samples else None,
                                # We don't have accelerometer or gyroscope data in this context
                            }
                            
                            # Use aiohttp to make an async API call
                            async with aiohttp.ClientSession() as session:
                                async with session.post('http://localhost:8000/inference/predict', 
                                                       json=batch_data) as response:
                                    if response.status == 200:
                                        result = await response.json()
                                        print(f"Prediction result: {result}")
                                    else:
                                        print(f"API call failed with status {response.status}")
                        except Exception as e:
                            print(f"Error sending batch to prediction server: {str(e)}")
                        
                        # Reset the samples list after sending
                        last_2000_samples = []
                    
                    # Send the raw EEG data to TouchDesigner
                    client.send_message("/muse/eeg", sample)
                    
                    # Check for blink (spike downward in specified channels)
                    first_channel_value = sample[0]  # Using first channel for blink detection
                    has_blink = False
                    
                    if first_channel_value < blink_threshold:
                        if not blink_detected:  # Prevent multiple triggers for the same blink
                            print("Blink detected!")
                            client.send_message("/muse/blink", 1)  # Send blink trigger to TouchDesigner
                            blink_detected = True
                            has_blink = True
                    else:
                        if blink_detected:  # Only send zero when the blink was previously detected
                            print("No blink detected.")
                            client.send_message("/muse/blink", 0)  # Send zero signal to indicate no blink
                        blink_detected = False
                    
                    last_2000_samples.append((sample, timestamp, has_blink))
                    # Put EEG data in the queue for broadcasting
                    await eeg_queue.put((sample, timestamp, has_blink))
                
                # Small delay to prevent CPU overload but keep real-time performance
                await asyncio.sleep(0.001)
            except Exception as e:
                print(f"Error processing EEG sample: {str(e)}")
                await asyncio.sleep(1)  # Prevent tight loop in case of repeated errors
    except Exception as e:
        print(f"Error in process_eeg_data: {str(e)}")
        import traceback
        traceback.print_exc()

# Configure and start the web server
async def start_server():
    app = web.Application()
    
    # Simple health check endpoint
    async def health_check(request):
        return web.Response(text="SSE server is running")
    
    app.router.add_get('/', health_check)
    app.router.add_get('/eeg-stream', sse_handler)
    
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, 'localhost', 8765)
    await site.start()
    print(f"SSE server started on http://localhost:8765/eeg-stream")
    print(f"Health check available at http://localhost:8765/")

# Main function to start everything
async def main():
    # Start the web server
    await start_server()
    
    # Start the broadcaster
    broadcast_task = asyncio.create_task(broadcast_eeg_data())
    
    # Start EEG processing
    eeg_task = asyncio.create_task(process_eeg_data())
    
    # Wait for both tasks to complete (they won't unless there's an error)
    await asyncio.gather(broadcast_task, eeg_task)

# Run the main function
if __name__ == "__main__":
    asyncio.run(main())
