from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def fetch_batch_from_file(filename, C, W, channel_mask, start_sample=0):
    with open(filename, 'rb') as fidInput:
        y = np.fromfile(fidInput, dtype=np.int16, offset=start_sample * C, count=C * W)

        if y.size == 0 or y.size != C * W:
            print(f"Error: read {self.y.size} entries from {filename}, expected {self.C * self.W}")

        
        y = np.reshape(y, (C, W), order='F')[channel_mask, :].flatten(order='F')

        return y
        
sglx_output_path = Path("C:/") / "SGL_DATA" / "01_27_p1_templategeneration_g0" / "01_27_p1_templategeneration_g0_imec0" / "rawBinaryData.txt"
    
with open(sglx_output_path) as file:
    end_sample = -1
    C = -1
    W = -1
    channel_mask = []
    
    for line in file:
        if line[:len("Sample")] == "Sample":
            end_sample = int(line.split(" ")[1])
            continue

        if line[:len("Channels")] == "Channels":
            C = int(line.split(" ")[1])
            continue

        if line[:len("Window")] == "Window":
            W = int(line.split(" ")[1])
            continue

        if line[:len("ChannelMap")] == "ChannelMap":
            channel_mask = [int(token) for token in line.split(" ")[1:] if token != "\n"]
            continue
        
        sglx_data = [int(token) for token in line.split(" ") if token != "\n" and token != ""]
        bin_data = fetch_batch_from_file(Path("C:\\SGL_DATA\\05_31\\imec_raw") / '240531_g0_t0.imec0.ap.bin', 385, W, channel_mask, start_sample=end_sample - W * C)
        
    for chan_index in range(C):
        sglx_channel = sglx_data[chan_index: len(sglx_data) : C]
        bin_channel = bin_data[chan_index: len(sglx_data) : C]
        print(sum(sglx_channel) / len(sglx_channel))
        print(sum(bin_channel) / len(bin_channel))
        plt.plot(sglx_channel, "b")
        plt.plot(bin_channel, "r")
        plt.show()
        
        

        
