import time
from pylsl import StreamInlet, resolve_streams
from pythonosc import udp_client
from pythonosc.osc_message_builder import OscMessageBuilder

# OSC Setup - Set the IP and port of TouchDesigner
osc_ip = '127.0.0.1'  # Localhost
osc_port = 7000        # Port to receive OSC data in TouchDesigner
client = udp_client.SimpleUDPClient(osc_ip, osc_port)

# Resolve the stream from MuseLSL (looking for EEG stream)
print("Looking for an EEG stream...")
streams = resolve_streams(5.0)  # Correct function call here

# Create an inlet for the EEG stream
inlet = StreamInlet(streams[0])

# Continuously read data from the Muse device and forward it to OSC
while True:
    # Get the sample (EEG data) and timestamp from the LSL stream
    sample, timestamp = inlet.pull_sample()

    # Build OSC message with EEG data
    message = OscMessageBuilder(address="/muse/eeg")
    for eeg_value in sample:  # Add each EEG value as a separate argument
        message.add_arg(eeg_value)
    
    # Send the message to TouchDesigner
    message = message.build()
    client.send(message)

    # Print the EEG data (for debugging)
    print(sample)

    # Sleep for a short time to prevent overload
    time.sleep(0.1)
