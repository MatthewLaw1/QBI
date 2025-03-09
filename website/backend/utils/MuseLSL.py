import time
import asyncio
import json
from aiohttp import web
from pylsl import StreamInlet, resolve_streams
from pythonosc import udp_client
from pythonosc.osc_message_builder import OscMessageBuilder

# OSC Setup - Set the IP and port of TouchDesigner
osc_ip = '127.0.0.1'  # Localhost
osc_port = 7000        # Port to receive OSC data in TouchDesigner
client = udp_client.SimpleUDPClient(osc_ip, osc_port)

# Store for SSE clients
sse_clients = set()

# EEG data queue for sharing data between coroutines
eeg_queue = asyncio.Queue()

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
        eeg_data, timestamp = await eeg_queue.get()
        
        # Create a JSON message with EEG data and timestamp
        message = json.dumps({
            "eeg": eeg_data,
            "timestamp": timestamp
        })
        
        # Send to all connected clients
        for client_queue in list(sse_clients):
            try:
                await client_queue.put(message)
            except Exception:
                # If there's an error, we'll remove this client
                if client_queue in sse_clients:
                    sse_clients.remove(client_queue)

# Main function to process EEG data
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
        
        # Create an inlet for the EEG stream
        inlet = StreamInlet(streams[0])
        
        sample_count = 0
        
        while True:
            try:
                # Get the sample (EEG data) and timestamp from the LSL stream
                sample, timestamp = inlet.pull_sample()
                
                sample_count += 1
                if sample_count % 100 == 0:  # Log every 100th sample to avoid console spam
                    print(f"Sample #{sample_count}: {sample}")
                
                # Build OSC message with EEG data (for TouchDesigner)
                message = OscMessageBuilder(address="/muse/eeg")
                for eeg_value in sample:
                    message.add_arg(eeg_value)
                
                # Send the message to TouchDesigner
                message = message.build()
                client.send(message)
                
                # Put EEG data in the queue for broadcasting
                await eeg_queue.put((sample, timestamp))
                
                # Small delay to prevent overload
                await asyncio.sleep(0.01)
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
    app.router.add_get('/eeg-stream', sse_handler)
    
    # Add CORS middleware if needed
    # app.add_middleware(...)
    
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, 'localhost', 8765)
    await site.start()
    print(f"SSE server started on http://localhost:8765/eeg-stream")

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
