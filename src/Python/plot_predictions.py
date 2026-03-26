from pathlib import Path
import matplotlib.pyplot as plt
import random
import numpy as np
import cupy as cp  # Use CuPy for GPU-accelerated array operations

def randomly(seq):
    shuffled = list(seq)
    random.shuffle(shuffled)
    return iter(shuffled)

def load_predictions(file_path):
    """
    Loads predictions from a file and returns a CuPy array of shape (n, 2).
    Each line in the file should contain two values separated by a space:
    a float (time) and an int (label).
    """
    # Use numpy to load the text file and then transfer to CuPy
    data = np.loadtxt(file_path)
    predictions = cp.asarray(data)
    return predictions

def evaluate_event(predictions, event, window_start_offset=150*30, window_end_offset=250*30):
    """
    Given an event (time, true_label) and a CuPy array of predictions,
    filters predictions within the window [event_time + window_start_offset, event_time + window_end_offset].
    Returns the ratio of predictions that match the event’s true label.
    If no predictions are found in the window, returns None.
    """
    true_label = event[1]
    event_time = event[0]
    start_time = event_time + window_start_offset
    end_time = event_time + window_end_offset
    # Create a mask for the window using CuPy vectorized operations
    mask = (predictions[:, 0] >= start_time) & (predictions[:, 0] <= end_time)
    predictions_cropped = predictions[mask]
    if predictions_cropped.shape[0] == 0:
        return None
    good = cp.sum(predictions_cropped[:, 1] == true_label)
    ratio = good / predictions_cropped.shape[0]
    return float(ratio)

def percent_mismatch(oss_predictions, ks_predictions, bin_width=100*30):
    """
    Computes the fraction of bins (of width `bin_width`) where the majority vote between the two
    prediction sets differs. The predictions are assumed to be CuPy arrays.
    """
    # Determine overlapping time range (transfer scalar values to CPU)
    start_time = float(cp.maximum(cp.min(ks_predictions[:, 0]), cp.min(oss_predictions[:, 0])).item())
    end_time   = float(cp.minimum(cp.max(ks_predictions[:, 0]), cp.max(oss_predictions[:, 0])).item())

    # Create bins from start_time to end_time in steps of bin_width/2
    bins = list(range(int(start_time), int(end_time), bin_width // 2))

    mismatch_count = 0
    valid_bins = 0

    for bin_start in bins:
        bin_end = bin_start + bin_width
        # Get the predictions in the current bin for each sorter using CuPy
        ks_mask = (ks_predictions[:, 0] >= bin_start) & (ks_predictions[:, 0] < bin_end)
        oss_mask = (oss_predictions[:, 0] >= bin_start) & (oss_predictions[:, 0] < bin_end)
        ks_bin_labels = ks_predictions[ks_mask][:, 1]
        oss_bin_labels = oss_predictions[oss_mask][:, 1]

        # Only evaluate this bin if both sorters have at least one prediction
        if ks_bin_labels.size == 0 or oss_bin_labels.size == 0:
            continue

        # Compute majority vote (1 if more than half are 1, else 0)
        ks_majority = 1 if int(cp.sum(ks_bin_labels).item()) > ks_bin_labels.size / 2 else 0
        oss_majority = 1 if int(cp.sum(oss_bin_labels).item()) > oss_bin_labels.size / 2 else 0
        valid_bins += 1

        if ks_majority != oss_majority:
            mismatch_count += 1

    if valid_bins > 0:
        return mismatch_count / valid_bins
    else:
        return None

def shifted(predictions, shift):
    """
    Shifts the prediction times by the given shift value (in time units).
    Operates on CuPy arrays and returns a new CuPy array.
    """
    return cp.concatenate([predictions[:, 0:1] + shift, predictions[:, 1:2]], axis=1)

def decoder_accuracy(predictions, events, p=0.5):
    """
    Computes the fraction of events for which the ratio of matching predictions in the evaluation
    window exceeds the threshold p.
    """
    count = 0
    total = 0

    for event in events:
        result = evaluate_event(predictions, event)

        if result is None:
            continue

        total += 1
        if result > p:
            count += 1

    if total == 0:
        return None

    return count / total

def plot_shifted_accuracy(predictions, events, start=-300, end=300, inc=10, p=0.5):
    """
    Evaluates and plots decoder accuracy as a function of time shift.
    """
    shifts = []
    accuracies = []

    print(f"Shift (ms)    |           Accuracy")
    print("-----------------------------------")
    for shift in range(start, end, inc):
        shifts.append(shift)
        # Shift predictions (note: shift * 30 conversion remains as in your original code)
        predictions_shifted = shifted(predictions, shift * 30)
        accuracy = decoder_accuracy(predictions_shifted, events, p)
        accuracies.append(accuracy)
        print(f"{str(shift).ljust(4)}          |  {accuracy}")

    plt.title(f"Shift (ms) vs Accuracy (p = {p})")
    plt.plot(shifts, accuracies)
    plt.xlabel("Shift (ms)")
    plt.ylabel("Accuracy")
    plt.show()

def shifted_percent_mismatches(oss_predictions, ks_predictions, start=-300, end=300, inc=10, p=0.5):
    """
    Evaluates and plots the percentage of mismatches between two prediction sets as a function of time shift.
    """
    shifts = []
    mismatches = []
    print(f"Shift (ms)    |         Mismatches")
    print("-----------------------------------")
    for shift in range(start, end, inc):
        shifts.append(shift)
        oss_predictions_shifted = shifted(oss_predictions, shift * 30)
        ks_predictions_shifted  = shifted(ks_predictions, shift * 30)
        mismatch = percent_mismatch(oss_predictions_shifted, ks_predictions_shifted)
        mismatches.append(mismatch)
        print(f"{str(shift).ljust(4)}          |  {mismatch}")

    plt.title(f"Shift (ms) vs Mismatch (p = {p})")
    plt.plot(shifts, mismatches)
    plt.xlabel("Shift (ms)")
    plt.ylabel("Mismatch Fraction")
    plt.show()

def events_in_range(events, start, end):
    return len([event for event in events if start <= event[0] <= end])

def analyze(oss_predictions, ks_predictions, events, 
            report=False, report_path=None, report_title=None,
            train_start=0, train_end=26984448, 
            test_start=26984448, test_end=83984384):
    # For reference, plot all predictions (ks as 'x', oss as 'o') with event time markers
    ks_times = cp.asnumpy(ks_predictions[:, 0])
    ks_labels = cp.asnumpy(ks_predictions[:, 1])
    oss_times = cp.asnumpy(oss_predictions[:, 0])
    oss_labels = cp.asnumpy(oss_predictions[:, 1])

    num_train = events_in_range(events, train_start, train_end)
    num_test  = events_in_range(events, test_start, test_end)
    oss_acc   = decoder_accuracy(oss_predictions, events, p=0.5)
    ks_acc    = decoder_accuracy(ks_predictions, events, p=0.5)
    mismatch  = percent_mismatch(oss_predictions, ks_predictions)
    print(f"Number of events in training set: {num_train}")
    print(f"Number of events in testing set: {num_test}")
    print(f"LSS decoder accuracy: {oss_acc}")
    print(f"KS decoder accuracy: {ks_acc}")
    print(f"Percentage mismatch = {mismatch}")

    if report and report_path and report_title:
        with open(report_path, 'a') as file:
            file.write(report_title + '\n')
            file.write(f"Number of events in training set: {num_train}\n")
            file.write(f"Number of events in testing set: {num_test}\n")
            file.write(f"OSS decoder accuracy: {oss_acc}\n")
            file.write(f"KS decoder accuracy: {ks_acc}\n")
            file.write(f"Percentage mismatch = {mismatch}\n")

    for event in randomly(events):
        event_time = event[0]

        if event_time < cp.asnumpy(cp.min(ks_predictions[:, 0])) or event_time < cp.asnumpy(cp.min(oss_predictions[:, 0])):
            continue

        ks_mask = (ks_predictions[:, 0] >= event_time - 6000) & (ks_predictions[:, 0] <= event_time + 9000)
        oss_mask = (oss_predictions[:, 0] >= event_time - 6000) & (oss_predictions[:, 0] <= event_time + 9000)
        ks_cropped = cp.asnumpy(ks_predictions[ks_mask])
        oss_cropped = cp.asnumpy(oss_predictions[oss_mask])

        if ks_cropped.size == 0 and oss_cropped.size == 0:
            continue

        plt.figure(figsize=(10, 4))
        if ks_cropped.size > 0:
            plt.scatter((ks_cropped[:, 0] - event_time) / 30 - 100, ks_cropped[:, 1] - 0.05, marker='x', color='blue', label='ks_predictions')
        if oss_cropped.size > 0:
            plt.scatter((oss_cropped[:, 0] - event_time) / 30 - 100, oss_cropped[:, 1], marker='o', color='red', label='oss_predictions')
        plt.vlines(0, 0, 1, colors='gray', linestyles='dashed')
        plt.vlines(100, 0, 1, colors='gray', linestyles='dashed')
        plt.hlines(event[1], 0, 100, colors='green', linestyles='solid')
        plt.title(f"True label = {event[1]}, event_time = {event_time}")
        plt.legend()
        plt.show()

# Define the directory containing the data
directory = Path("C:/", "SGL_DATA", "joplin_20240222", "decoder_input")

# Define the files you want to analyze
event_filenames = ["eventfile_15.txt"]
ks_filenames    = ["ks_predictions.txt"]
oss_filenames   = ["oss_predictions.txt"]
assert(len(event_filenames) == len(ks_filenames) and len(ks_filenames) == len(oss_filenames))

# Load events and predictions
print(f"Loading events from {directory / 'eventfile.txt'}")
print(f"Analyzing {len(event_filenames)} eventfiles.")

for event_filename, ks_filename, oss_filename in zip(event_filenames, ks_filenames, oss_filenames):
    with open(directory / event_filename, 'r') as f:
        events = [(int(line.split(' ')[0]), int(line.split(' ')[1])) for line in f]

    # Load predictions using the new load_predictions function (which returns CuPy arrays)
    print(f"Loading ks_predictions from {directory / ks_filename}")
    ks_predictions = load_predictions(directory / ks_filename)

    print(f"Loading oss_predictions from {directory / oss_filename}")
    oss_predictions = load_predictions(directory / oss_filename)

    analyze(oss_predictions, ks_predictions, events,
            report=True,
            report_path=directory / "report.txt",
            report_title=event_filename)
