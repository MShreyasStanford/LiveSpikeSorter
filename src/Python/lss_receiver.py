"""REFERENCE RECEIVER FOR LSS SPIKES

1. In the launcher, tick "Stream spikes over UDP" and set the address to
   127.0.0.1:9100 (or this machine's IP if LSS runs on another computer).
2. Run this script, then launch LSS:
       python src/Python/lss_receiver.py
3. Every sorted spike is printed as it arrives. Replace the print in the loop
   with your own closed-loop logic.

The packet layout is defined in src/Networking/SpikeStream.h. For a fuller
receiver (dropped-packet stats, latency, checking against spikeOutput.txt),
see src/Python/lss_stream.py.
"""
import socket
import struct

PORT = 9100  # must match the port set in the launcher

HEADER = struct.Struct("<IHHIHHQQQIHH")
RECORD = struct.Struct("<Qif")
MAGIC = 0x3153534C  # bytes "LSS1"

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(("0.0.0.0", PORT))  # 0.0.0.0 = accept packets from any machine
print(f"Listening for LSS spikes on port {PORT} (Ctrl+C to stop)")

while True:
    packet, _ = sock.recvfrom(2048)  # one packet holds at most 89 spikes

    header = HEADER.unpack_from(packet, 0)
    magic, sorter, n_records, probe = header[0], header[2], header[10], header[11]
    if magic != MAGIC:
        continue  # not an LSS packet

    # A packet arrives for every sorter batch, even one with no spikes.
    for i in range(n_records):
        sample, cluster, amplitude = RECORD.unpack_from(packet, HEADER.size + i * RECORD.size)
        # ---------------------------------- RENAME ----------------------------------
        # cluster uses oss_input/templateMap.npy to map to the Kilosort cluster ID (column 2 of spikeOutput.txt)
        print(f"sample {sample}  cluster {cluster}  amp {amplitude:.1f}")
