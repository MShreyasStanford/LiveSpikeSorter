from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox
import random
import os
import pickle
import numpy as np
import cupy as cp

def stonum(s):
    if '.' in s or 'e' in s or 'E' in s:
        return float(s)
    else:
        return int(s)

def deterministic_hash(s: str) -> int:
    hash_val = 5381
    for c in s:
        # Equivalent to: hash_val = hash_val * 33 + ord(c)
        hash_val = ((hash_val << 5) + hash_val) + ord(c)
    # Optionally, restrict to 32 bits
    return hash_val & 0xFFFFFFFF

def psth(event_time, spikes, templates=None, inc=10*30, window_size=100*30):
    if len(spikes) == 0:
        return None, None

    if min([spike[0] for spike in spikes]) >= event_time + 500 * 300:
        return None, None

    if max([spike[0] for spike in spikes]) <= event_time - 500 * 30:
        return None, None

    # The list of counts to plot for PSTH
    psth_x = []
    psth_y = []

    start = -500 * 30
    end = start + window_size
    
    while end <= 500 * 30:
        window_spike_count = 0
        
        for (sample, template) in spikes:
            if templates and (template not in templates):
                continue
            
            if sample >= start + event_time and sample <= end + event_time:
                window_spike_count += 1
            
        psth_x.append(end / 30)
        psth_y.append(window_spike_count)
        start += inc
        end += inc

    return psth_x, psth_y

def get_events(eventfile):
    print(f"Loading events from {eventfile}.")
    with open(eventfile, 'r') as file:
        return  [stonum(line) if len(line.split(' ')) == 1 else stonum(line.split(' ')[0]) for line in file ]
    return None

def cropped_events(events, min_spike, max_spike, k):
    cropped_events = [ event for event in events if min_spike <= event <= max_spike ]
    cropped_events = sorted(random.sample(cropped_events, min(k, len(cropped_events))))
    return cropped_events

def get_spikes(file):
    print(f"Loading spikes from {file}")
    with open(file, 'r') as f:
        return [
            (stonum(line.split(',')[0]), stonum(line.split(',')[1]))
            for line in f
        ]
    return None

def compute_psth_data(spikes1, spikes2, events):
    psth_data = {}
    first_event = events[0]
    last_event = events[-1]
    T = max(max([ spike[1] for spike in spikes1 ]), max([ spike[1] for spike in spikes2]))
    print(f"Processing events from {first_event} to {last_event}")

    min_time = first_event - 1000 * 30
    max_time = last_event + 1000 * 30
    psth_len = None

    # Initialize standard_x for consistent x-axis across all plots
    standard_x = None

    # Iterate over each template to plot PSTHs
    for template in range(1, T + 1):
        print(f"Processing template {template}")

        # Process Kilosort spikes for the current template
        spikes1_cropped = [(s, t) for s, t in spikes1 if t == template]
        oss1_x, oss1_y = None, None
        for event in events:
            psth_x, psth_y = psth(event, spikes1_cropped)
            if psth_y is None:
                continue
            if oss1_y:
                oss1_y = [oss1_y[i] + psth_y[i] for i in range(len(psth_y))]
            else:
                oss1_x, oss1_y = psth_x, psth_y

        # Process OSS spikes for the current template
        oss2_x, oss2_y = None, None
        spikes2_cropped = [(s, t) for s, t in spikes2 if t == template]
        for event in events:
            psth_x, psth_y = psth(event, spikes2_cropped)
            if psth_y is None:
                continue
            if oss2_y:
                oss2_y = [oss2_y[i] + psth_y[i] for i in range(len(psth_y))]
            else:
                oss2_x, oss2_y = psth_x, psth_y

        # Determine the standard x-axis from the first successful PSTH
        if standard_x is None:
            if oss1_x is not None:
                standard_x = oss1_x.copy()
            elif oss2_x is not None:
                standard_x = oss2_x.copy()

        # Handle cases where PSTH returns None by plotting y=0
        if oss1_x is None or oss1_y is None:
            if standard_x is not None:
                oss1_x = standard_x.copy()
                oss1_y = [0] * len(standard_x)
                print(f"LSS1 PSTH for template {template} is None. Plotting zeros.")
            else:
                print(f"LSS1 PSTH for template {template} is None and standard_x is not set. Skipping.")
                continue

        if oss2_x is None or oss2_y is None:
            if standard_x is not None:
                oss2_x = standard_x.copy()
                oss2_y = [0] * len(standard_x)
                print(f"LSS2 PSTH for template {template} is None. Plotting zeros.")
            else:
                print(f"LSS2 PSTH for template {template} is None and standard_x is not set. Skipping.")
                continue

        # Ensure the x-axes match the standard_x
        if oss1_x != standard_x:
            print(f"LSS1 PSTH x-axis for template {template} does not match standard_x. Skipping.")
            continue

        if oss2_x != standard_x:
            print(f"LSS2 PSTH x-axis for template {template} does not match standard_x. Skipping.")
            continue

        psth_data[template] = {'standard_x': standard_x, 'oss1_y': oss1_y, 'oss2_y': oss2_y}

    return psth_data

def compute_psth_data_gpu(spikes1, spikes2, events):
    psth_data = {}
    # Convert events list to a CuPy array (in sample units)
    events_cp = cp.asarray(events)  # shape: (num_events,)

    # Define parameters (in samples)
    inc = 10 * 30         # 300
    window_size = 100 * 30  # 3000
    start_initial = -500 * 30  # -6000
    max_end = 500 * 30         # 9000

    # Determine number of windows. We require: start_initial + j*inc + window_size <= max_end.
    num_windows = int((max_end - (start_initial + window_size)) // inc + 1)  # should be 41
    # Window offsets (in samples)
    window_starts = cp.arange(num_windows) * inc + start_initial  # shape: (num_windows,)
    # Standard x-axis computed as window_end/30
    standard_x = cp.asnumpy(window_starts + window_size) / 30  # convert back to time units

    # Determine maximum template number from spikes1 and spikes2
    T1 = max(s[1] for s in spikes1) if spikes1 else 0
    T2 = max(s[1] for s in spikes2) if spikes2 else 0
    T = max(T1, T2)

    for template in range(1, T + 1):
        print(f"Processing template {template}.")
        # Filter spikes for the current template
        template_spikes1 = [s for s in spikes1 if s[1] == template]
        template_spikes2 = [s for s in spikes2 if s[1] == template]
        
        # If there are no spikes in both lists, skip this template
        if not template_spikes1 and not template_spikes2:
            continue

        # Convert spike times to CuPy arrays (they are assumed to be in sample units)
        if template_spikes1:
            S1 = cp.asarray([s[0] for s in template_spikes1])
            S1 = cp.sort(S1)
        else:
            S1 = cp.array([], dtype=cp.float32)

        if template_spikes2:
            S2 = cp.asarray([s[0] for s in template_spikes2])
            S2 = cp.sort(S2)
        else:
            S2 = cp.array([], dtype=cp.float32)

        # For each event, compute the lower and upper bounds for each window.
        # lower_bounds shape: (num_events, num_windows)
        lower_bounds = events_cp[:, None] + window_starts[None, :]
        upper_bounds = lower_bounds + window_size

        # Use searchsorted to count spikes falling in each window for S1:
        if S1.size > 0:
            left_indices1 = cp.searchsorted(S1, lower_bounds, side='left')
            right_indices1 = cp.searchsorted(S1, upper_bounds, side='right')
            counts1 = right_indices1 - left_indices1  # shape: (num_events, num_windows)
            agg_counts1 = cp.sum(counts1, axis=0)       # shape: (num_windows,)
            aggregated_counts1 = cp.asnumpy(agg_counts1)
        else:
            aggregated_counts1 = np.zeros(num_windows, dtype=int)

        # And similarly for S2:
        if S2.size > 0:
            left_indices2 = cp.searchsorted(S2, lower_bounds, side='left')
            right_indices2 = cp.searchsorted(S2, upper_bounds, side='right')
            counts2 = right_indices2 - left_indices2
            agg_counts2 = cp.sum(counts2, axis=0)
            aggregated_counts2 = cp.asnumpy(agg_counts2)
        else:
            aggregated_counts2 = np.zeros(num_windows, dtype=int)

        psth_data[template] = {
            'standard_x': standard_x.tolist(), 
            'oss1_y': aggregated_counts1.tolist(), 
            'oss2_y': aggregated_counts2.tolist()
        }

    return psth_data

def hashed_pickle_file_name(file1, file2, eventfile, spikes1, spikes2, events):
    return deterministic_hash(f'{file1}_{file2}_{eventfile}_{len(spikes1)}_{len(spikes2)}_{len(events)}')

def run(directory, file1, file2, eventfile, k):
    # Load spike data from file
    spikes1 = get_spikes(file1)
    spikes2 = get_spikes(file2)

    if not spikes1 or not spikes2:
        print("One of the spike files was not loaded correctly.")
        return

    # Parameters
    T = max(max([ spike[1] for spike in spikes1 ]), max([ spike[1] for spike in spikes2]))
    print(f"Creating PSTHs for {T} templates.")

    # Load eventfile
    events = get_events(eventfile)

    if not events:
        print("Could not load events.")
        return

    # Crop events to cut on processing time
    min_spike = max(min([ spike[0] for spike in spikes1 ]), min([ spike[0] for spike in spikes2 ]))
    max_spike = min(max([ spike[0] for spike in spikes2 ]), max([ spike[0] for spike in spikes2 ]))
    events = cropped_events(events, min_spike, max_spike, k)

    # Check if pickled file exists
    psth_pickle_file = directory / f"{hashed_pickle_file_name(file1, file2, eventfile, spikes1, spikes2, events)}.pkl"

    if os.path.exists(psth_pickle_file):
        print(f"Loading PSTH data from pickle: {psth_pickle_file}")
        with open(psth_pickle_file, 'rb') as pf:
            psth_data = pickle.load(pf)
    else:
        psth_data = compute_psth_data_gpu(spikes1, spikes2, events)

        # After processing all templates, pickle the PSTH data.
        with open(psth_pickle_file, 'wb') as pf:
            pickle.dump(psth_data, pf)

    # Initialize plotting objects
    lines_to_templates = {}

    # Initialize the figure and subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 10), sharex=True)
    ax1.set_title(f"{file1}")
    ax2.set_title(f"{file2}")

    standard_x = None
    # Plot the lines for the single-neuron PSTHs
    for template in range(1, T + 1):
        oss1_y = psth_data[template]['oss1_y']
        oss2_y = psth_data[template]['oss2_y']
        curr_x = psth_data[template]['standard_x']

        if standard_x is None:
            standard_x = curr_x.copy()

        # Plot OSS data
        if not oss1_y:
            oss1_y = [0] * len(standard_x)
        if not oss2_y:
            oss2_y = [0] * len(standard_x)

        line1, = ax1.plot(standard_x, oss1_y, label=f"Template {template}", picker=5, alpha=0.7)
        lines_to_templates[line1] = str(template)  # Map line to template

        # Plot Kilosort data
        line2, = ax2.plot(standard_x, oss2_y, label=f"Template {template}", picker=5, alpha=0.7)
        lines_to_templates[line2] = str(template)  # Map line to template

    # Adjust the main plot area to make room for TextBox on the right
    plt.subplots_adjust(right=0.8)

    # Create an axes for the TextBox on the right
    text_box_ax = fig.add_axes([0.85, 0.9, 0.1, 0.05])  # [left, bottom, width, height]
    text_box = TextBox(text_box_ax, 'Enter Template:', initial='All')

    # Create an annotation for displaying the template name upon clicking a line
    annotation = ax1.annotate(
        "",
        xy=(0, 0),
        xytext=(20, 20),
        textcoords="offset points",
        bbox=dict(boxstyle="round", fc="w"),
        arrowprops=dict(arrowstyle="->"),
    )
    annotation.set_visible(False)

    def on_pick(event):
        # Get the line that was clicked
        clicked_line = event.artist

        # Get the template associated with the clicked line
        clicked_template = lines_to_templates.get(clicked_line, None)

        if clicked_template is None:
            return  # Click was not on a recognized line

        # Display the template name using annotation at the click location
        if hasattr(event, 'mouseevent') and event.mouseevent.xdata and event.mouseevent.ydata:
            x_val = event.mouseevent.xdata
            y_val = event.mouseevent.ydata
            annotation.xy = (x_val, y_val)
            annotation.set_text(f"Template: {clicked_template}")
            annotation.set_visible(True)
            fig.canvas.draw_idle()

        # Highlight the clicked template's lines and dim others
        isolate_template(clicked_template)

    def isolate_template(template):
        # Highlight the selected template and dim others
        for line, tmpl in lines_to_templates.items():
            if tmpl == template:
                line.set_alpha(1.0)
            else:
                line.set_alpha(0.1)

        # Update the TextBox to reflect the selected template
        text_box.set_val(str(template))
        fig.canvas.draw_idle()

    def reset_visibility():
        # Reset all lines to full opacity
        for line in lines_to_templates.keys():
            line.set_alpha(0.7)

        # Hide the annotation
        annotation.set_visible(False)
        fig.canvas.draw_idle()

    def submit(text):
        label = text.strip()
        if label.lower() == 'all' or label == '':
            reset_visibility()
        elif label in lines_to_templates.values():
            isolate_template(label)
        else:
            self.logger.warning(f"Template '{label}' not found.")
            # Provide visual feedback to the user
            annotation.xy = (0, 0)
            annotation.set_text(f"Template '{label}' not found.")
            annotation.set_visible(True)
            fig.canvas.draw_idle()

    # Connect the pick event handler
    fig.canvas.mpl_connect('pick_event', on_pick)

    # Connect the TextBox submission event handler
    text_box.on_submit(submit)

    print("Showing plot!")
    plt.tight_layout(rect=[0, 0, 0.8, 1])  # Adjust layout to make room for TextBox
    plt.show()

    all_oss1 = []
    all_oss2 = []
    for template in range(1, T + 1):
        oss1_y = psth_data[template]['oss1_y']
        oss2_y = psth_data[template]['oss2_y']
        if not oss1_y:
            oss1_y = [0] * len(standard_x)
        if not oss2_y:
            oss2_y = [0] * len(standard_x)

        all_oss1.append(oss1_y)
        all_oss2.append(oss2_y)

    avg_oss1 = np.mean(all_oss1, axis=0)
    avg_oss2 = np.mean(all_oss2, axis=0)

    if np.max(avg_oss1) != 0:
        avg_oss1 = avg_oss1 / np.max(avg_oss1)
    if np.max(avg_oss2) != 0:
        avg_oss2 = avg_oss2 / np.max(avg_oss2)

    plt.plot(standard_x, avg_oss1, label="Average OSS", alpha=0.9)
    plt.plot(standard_x, avg_oss2, label="Average KS", alpha=0.9)
    plt.legend()
    plt.show()

base_dir = Path("C:/", "SGL_DATA", "joplin_20240222")
run(
    base_dir,
    Path(base_dir, "cuda_output") / "spikeOutput_train_train.txt",
    Path(base_dir, "decoder_input") / "ksSpikeOutput.txt",
    Path(base_dir, "decoder_input") / "eventfile_15.txt",
    1000
)