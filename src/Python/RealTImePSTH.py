from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import TextBox
import numpy as np
import cupy as cp
import time

# === Global Parameters ===
# Spike time offset (in sample units)
offset = 26984448 -1 * 256 * 64  # Change as needed

# PSTH parameters (assumes sample rate of 30 samples/sec)
inc = 10 * 30           # step size (300 samples)
window_size = 100 * 30   # window length (3000 samples)
start_initial = -300 * 30  # start offset (-15000 samples)
max_end = 300 * 30        # end offset (15000 samples)

# Derived constants: number of windows, window start offsets, and standard x-axis (in time units)
num_windows = int((max_end - (start_initial + window_size)) // inc + 1)
window_starts = cp.arange(num_windows) * inc + start_initial
standard_x = cp.asnumpy(window_starts + window_size) / 30

# === Global Variables for Incremental Computation ===
# All spikes: list of tuples (time, template)
spike_data = []  
# Also maintain spikes sorted by template for faster lookup.
spikes_by_template = {}  # { template : [spike_time1, spike_time2, ...] }

# Static event data: list of event times (read once at startup)
event_data = []  

# For incremental processing, events whose entire window (event + max_end) is in the past are "closed"
closed_event_count = 0  # index in event_data up to which events are closed
# Store closed events' aggregated histograms by template:
closed_hist = {}       # { template : numpy.array of length num_windows }
# For incremental update of plotted PSTHs, store previous aggregated histogram per template.
prev_agg_hist = {}     # { template : numpy.array }

# === File Pointers ===
spike_fp = None  # Only spike file will be updated over time

# === Plotting Globals ===
lines_by_template = {}  # mapping: template -> line object in the individual PSTH plot
avg_line = None         # line object for the average PSTH plot

# === Utility Functions ===
def stonum(s):
    if '.' in s or 'e' in s or 'E' in s:
        return float(s)
    else:
        return int(s)

def deterministic_hash(s: str) -> int:
    hash_val = 5381
    for c in s:
        hash_val = ((hash_val << 5) + hash_val) + ord(c)
    return hash_val & 0xFFFFFFFF

# === Functions to Update Data from Files ===
def update_spike_data():
    global spike_data, spike_fp, spikes_by_template, offset
    if spike_fp is None:
        return
    # Read any new lines from the spike file (assumed to be appended over time)
    while True:
        line = spike_fp.readline()
        if not line:
            break
        line = line.strip()
        if not line:
            continue
        parts = line.split(',')
        if len(parts) < 2:
            continue
        try:
            # Apply the offset to the spike time.
            time_val = stonum(parts[0]) + offset
            template = stonum(parts[1])
            spike_data.append((time_val, template))
            # Update per-template spike list
            if template not in spikes_by_template:
                spikes_by_template[template] = []
            spikes_by_template[template].append(time_val)
        except Exception as e:
            print(f"Error parsing spike line: {line} - {e}")

# === Incremental PSTH Update Function ===
def update_plot(frame):
    global closed_event_count, closed_hist, prev_agg_hist, avg_line

    # First, update spike data.
    update_spike_data()
    
    # We require at least one event (static) and one spike to compute PSTHs.
    if not event_data or not spike_data:
        return

    # Use the latest spike time as the current time.
    current_time = spike_data[-1][0]
    
    # --- Process New Closed Events ---
    # Closed events are those for which event_time + max_end <= current_time.
    new_index = closed_event_count
    while new_index < len(event_data) and (event_data[new_index] + max_end) <= current_time:
        new_index += 1
    new_closed_events = event_data[closed_event_count:new_index]
    if new_closed_events:
        events_cp = cp.asarray(new_closed_events)
        # For each template, compute contributions for these newly closed events.
        for template, spike_list in spikes_by_template.items():
            S = cp.asarray(spike_list)  # already sorted (monotonically increasing)
            lower_bounds = events_cp[:, None] + window_starts[None, :]
            upper_bounds = lower_bounds + window_size
            left_indices = cp.searchsorted(S, lower_bounds, side='left')
            right_indices = cp.searchsorted(S, upper_bounds, side='right')
            counts = right_indices - left_indices  # shape: (num_new_events, num_windows)
            hist_new = cp.sum(counts, axis=0)
            hist_new = cp.asnumpy(hist_new)
            if template in closed_hist:
                closed_hist[template] += hist_new
            else:
                closed_hist[template] = hist_new
        closed_event_count = new_index

    # --- Process Open Events ---
    # These are events whose window is still open.
    open_events = event_data[closed_event_count:]
    if open_events:
        open_events_cp = cp.asarray(open_events)
    
    # For each template, combine closed events with the current contribution from open events.
    new_agg_hist = {}
    for template, spike_list in spikes_by_template.items():
        S = cp.asarray(spike_list)
        # For open events (if any), compute contribution.
        if open_events:
            lower_bounds = open_events_cp[:, None] + window_starts[None, :]
            upper_bounds = lower_bounds + window_size
            left_indices = cp.searchsorted(S, lower_bounds, side='left')
            right_indices = cp.searchsorted(S, upper_bounds, side='right')
            counts = right_indices - left_indices
            hist_open = cp.sum(counts, axis=0)
            hist_open = cp.asnumpy(hist_open)
        else:
            hist_open = np.zeros(num_windows, dtype=np.int64)
        # Combine with closed events (if any)
        if template in closed_hist:
            agg_hist = closed_hist[template] + hist_open
        else:
            agg_hist = hist_open
        new_agg_hist[template] = agg_hist

    # --- Update Only PSTH Lines That Changed ---
    for template, new_hist in new_agg_hist.items():
        # Compare with the previously stored histogram for this template.
        if template in prev_agg_hist:
            if np.array_equal(prev_agg_hist[template], new_hist):
                # No change for this template; skip updating its plot.
                continue
        # Update the plot for this template.
        if template in lines_by_template:
            line = lines_by_template[template]
            line.set_ydata(new_hist)
        else:
            # Create a new line if needed.
            line, = ax_individual.plot(standard_x, new_hist, label=f"Template {template}", picker=5, alpha=0.7)
            lines_by_template[template] = line
        # Save the new histogram for later comparison.
        prev_agg_hist[template] = new_hist.copy()
        
    # Remove plot lines for templates that are no longer present.
    for template in list(lines_by_template.keys()):
        if template not in new_agg_hist:
            line = lines_by_template.pop(template)
            line.remove()

    # --- Update the Average PSTH Plot ---
    if new_agg_hist:
        all_hist = np.array(list(new_agg_hist.values()))
        avg_hist = np.mean(all_hist, axis=0)
        # Normalize the average PSTH if nonzero.
        if np.max(avg_hist) != 0:
            avg_hist = avg_hist / np.max(avg_hist)
    else:
        avg_hist = np.zeros(num_windows)
    if avg_line is None:
        avg_line, = ax_average.plot(standard_x, avg_hist, label="Average PSTH", alpha=0.9)
    else:
        avg_line.set_ydata(avg_hist)

    # Adjust axes and legends.
    ax_individual.relim()
    ax_individual.autoscale_view()
    ax_average.relim()
    ax_average.autoscale_view()
    ax_individual.legend()
    ax_average.legend()
    fig.canvas.draw_idle()

# === Interactive Plotting (TextBox & Pick Event) ===
# Set up the figure with two subplots: one for individual PSTHs and one for the average.
fig, (ax_individual, ax_average) = plt.subplots(2, 1, figsize=(18, 10))
ax_individual.set_title("Individual Template PSTHs")
ax_average.set_title("Average PSTH")

# Create a TextBox for filtering by template.
plt.subplots_adjust(right=0.8)
text_box_ax = fig.add_axes([0.85, 0.9, 0.1, 0.05])
text_box = TextBox(text_box_ax, 'Enter Template:', initial='All')

# Create an annotation to display template info upon clicking.
annotation = ax_individual.annotate(
    "",
    xy=(0, 0),
    xytext=(20, 20),
    textcoords="offset points",
    bbox=dict(boxstyle="round", fc="w"),
    arrowprops=dict(arrowstyle="->"),
)
annotation.set_visible(False)

def isolate_template(template):
    """Highlight the selected template and dim others."""
    for tmpl, line in lines_by_template.items():
        if str(tmpl) == str(template):
            line.set_alpha(1.0)
        else:
            line.set_alpha(0.1)
    text_box.set_val(str(template))
    fig.canvas.draw_idle()

def reset_visibility():
    """Reset all lines to normal opacity."""
    for line in lines_by_template.values():
        line.set_alpha(0.7)
    annotation.set_visible(False)
    fig.canvas.draw_idle()

def submit(text):
    label = text.strip()
    if label.lower() == 'all' or label == '':
        reset_visibility()
    else:
        if label in [str(t) for t in lines_by_template.keys()]:
            isolate_template(label)
        else:
            print(f"Template '{label}' not found.")
            annotation.xy = (0, 0)
            annotation.set_text(f"Template '{label}' not found.")
            annotation.set_visible(True)
            fig.canvas.draw_idle()

text_box.on_submit(submit)

def on_pick(event):
    clicked_line = event.artist
    for tmpl, line in lines_by_template.items():
        if line == clicked_line:
            clicked_template = tmpl
            break
    else:
        return
    if hasattr(event, 'mouseevent') and event.mouseevent.xdata and event.mouseevent.ydata:
        x_val = event.mouseevent.xdata
        y_val = event.mouseevent.ydata
        annotation.xy = (x_val, y_val)
        annotation.set_text(f"Template: {clicked_template}")
        annotation.set_visible(True)
        fig.canvas.draw_idle()
    isolate_template(clicked_template)

fig.canvas.mpl_connect('pick_event', on_pick)

# === Main Function to Run in Real Time ===
def run_realtime(spike_file_path, event_file_path):
    global spike_fp, spike_data, event_data
    spike_file_path = Path(spike_file_path)
    event_file_path = Path(event_file_path)
    
    # --- Load the static event file once ---
    try:
        with open(event_file_path, 'r') as ef:
            for line in ef:
                line = line.strip()
                if not line:
                    continue
                try:
                    event_time = stonum(line.split()[0])
                    event_data.append(event_time)
                except Exception as e:
                    print(f"Error parsing event line: {line} - {e}")
    except Exception as e:
        print(f"Error opening event file: {e}")
        return
    
    # --- Open the spike file for real-time updates ---
    try:
        spike_fp = open(spike_file_path, 'r')
    except Exception as e:
        print(f"Error opening spike file: {e}")
        return

    # Initial spike data read.
    update_spike_data()
    
    # Start the animation timer (updates every 1000 ms).
    ani = FuncAnimation(fig, update_plot, interval=1000)
    plt.tight_layout(rect=[0, 0, 0.8, 1])
    plt.show()
    
    spike_fp.close()

# === Example Usage ===
# Replace the paths below with your actual file locations.
base_dir = Path("C:/", "SGL_DATA", "joplin_20240222")
spike_file = Path(base_dir, "cuda_output", "spikeOutput.txt")
event_file = Path(base_dir, "decoder_input", "eventfile_15.txt")
run_realtime(spike_file, event_file)
