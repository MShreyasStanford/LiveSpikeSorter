from pathlib import Path
import numpy as np
import cupy as cp  # Using CuPy for GPU acceleration
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

# Fixed bin parameters (samples at 30kHz)
BIN_SIZE = 100 * 30  # 100ms window (in samples)
BIN_INC  = 10 * 30   # 10ms step (in samples)

def train_decoder(spike_times, spike_templates, spike_amplitudes, 
                  event_times, event_labels,
                  training_start, training_end,
                  train_lookahead=0):
    """
    Trains a logistic regression decoder using binned spike counts computed 
    from spikes in a defined training period.
    
    Parameters:
        spike_times, spike_templates, spike_amplitudes (np.ndarray): 
            Spike data for training.
        event_times, event_labels (np.ndarray): 
            Combined event data (aggregated from eventfiles).
        training_start (int): First sample (inclusive) for training.
        training_end (int): Last sample (inclusive) for training.
        train_lookahead (int): Lookahead offset in ms for feature extraction.
        
    Returns:
        dict: Contains the trained classifier ('clf'), feature scaler ('scaler')
              and the unique spike templates ('templates') used for column ordering.
    """
    # Convert data to CuPy arrays.
    spike_times_cp    = cp.asarray(spike_times)
    spike_templates_cp = cp.asarray(spike_templates)
    event_times_cp    = cp.asarray(event_times)
    event_labels_cp   = cp.asarray(event_labels)
    
    # Crop spikes to training period.
    train_idx_left = int(cp.asnumpy(cp.where(spike_times_cp > training_start)[0][0]))
    train_idx_right = int(cp.asnumpy(cp.where(spike_times_cp < training_end)[0][-1]))
    train_spike_times = spike_times_cp[train_idx_left: train_idx_right]
    train_spike_templates = spike_templates_cp[train_idx_left: train_idx_right]
    
    # Define training spike range.
    train_spike_start = train_spike_times[0]
    train_spike_end   = train_spike_times[-1]
    
    # Crop events so that their times fall within the training spike period.
    evt_idx_left = int(cp.asnumpy(cp.where(event_times_cp > train_spike_start)[0][0]))
    evt_idx_right = int(cp.asnumpy(cp.where(event_times_cp < train_spike_end)[0][-1]))
    train_event_times = event_times_cp[evt_idx_left: evt_idx_right]
    train_event_labels = event_labels_cp[evt_idx_left: evt_idx_right]
    
    # Determine overall training time range.
    base_time_train = cp.minimum(train_event_times.min(), train_spike_times.min())
    max_time_train  = cp.maximum(train_event_times.max(), train_spike_times.max())
    num_bins_train = int(cp.floor((max_time_train - base_time_train - BIN_SIZE) / BIN_INC).get()) + 1
    
    # Precompute bin boundaries.
    bin_left_edges_train = base_time_train + cp.arange(num_bins_train) * BIN_INC
    bin_right_edges_train = bin_left_edges_train + BIN_SIZE
    
    # Get unique templates (defines feature column order).
    templates_train = cp.unique(train_spike_templates)
    num_templates = int(templates_train.size)
    
    # Build a 2D array of binned spike counts.
    binned_counts_train = cp.zeros((num_bins_train, num_templates), dtype=cp.int32)
    for i, tmpl in enumerate(cp.asnumpy(templates_train)):
        tmpl_spikes = train_spike_times[train_spike_templates == tmpl]
        start_idx = cp.searchsorted(tmpl_spikes, bin_left_edges_train, side='left')
        end_idx   = cp.searchsorted(tmpl_spikes, bin_right_edges_train, side='left')
        binned_counts_train[:, i] = end_idx - start_idx

    # Compute indices for events using the training lookahead offset.
    train_event_bin_idx = ((train_event_times + train_lookahead * 30 - base_time_train) // BIN_INC).astype(cp.int32)
    valid_train = train_event_bin_idx < num_bins_train
    if not bool(cp.all(valid_train)):
        print("Warning: Some training events fall outside the binned range and will be ignored.")
        train_event_bin_idx = train_event_bin_idx[valid_train]
        train_event_labels = train_event_labels[valid_train]
    
    X_train = binned_counts_train[train_event_bin_idx]
    y_train = train_event_labels
    X_train = cp.asnumpy(X_train)
    y_train = cp.asnumpy(y_train)
    
    # Standardize features and train the logistic regression classifier.
    scaler = StandardScaler()
    X_train_norm = scaler.fit_transform(X_train)
    clf = LogisticRegression()
    clf.fit(X_train_norm, y_train)
    
    return {'clf': clf, 'scaler': scaler, 'templates': cp.asnumpy(templates_train)}

def test_decoder(trained_params, spike_times, spike_templates, spike_amplitudes,
                 event_times, event_labels,
                 test_start, test_end, test_lookahead):
    """
    Evaluates a pretrained decoder on test data by computing binned spike features,
    transforming them with the trained scaler and using the pretrained classifier to 
    predict event labels.
    
    Parameters:
        trained_params (dict): Contains 'clf', 'scaler', and 'templates'.
        spike_times, spike_templates, spike_amplitudes (np.ndarray):
            Test spike data.
        event_times, event_labels (np.ndarray):
            Test event data.
        test_start (int): Start sample of the testing period.
        test_end (int): End sample of the testing period.
        test_lookahead (int): Lookahead offset (ms) for test feature extraction.
    
    Returns:
        float: The test accuracy (percentage correct).
    """
    # Convert to CuPy arrays.
    spike_times_cp    = cp.asarray(spike_times)
    spike_templates_cp = cp.asarray(spike_templates)
    event_times_cp    = cp.asarray(event_times)
    event_labels_cp   = cp.asarray(event_labels)
    
    # Crop spikes to test period.
    test_idx_left  = int(cp.asnumpy(cp.where(spike_times_cp > test_start)[0][0]))
    test_idx_right = int(cp.asnumpy(cp.where(spike_times_cp < test_end)[0][-1]))
    test_spike_times = spike_times_cp[test_idx_left:test_idx_right]
    test_spike_templates = spike_templates_cp[test_idx_left:test_idx_right]
    
    # Define the test spike range.
    test_spike_start = test_spike_times[0]
    test_spike_end   = test_spike_times[-1]
    
    # Crop test events to fall within the test spike period.
    evt_idx_left  = int(cp.asnumpy(cp.where(event_times_cp > test_spike_start)[0][0]))
    evt_idx_right = int(cp.asnumpy(cp.where(event_times_cp < test_spike_end)[0][-1]))
    test_event_times  = event_times_cp[evt_idx_left:evt_idx_right]
    test_event_labels = event_labels_cp[evt_idx_left:evt_idx_right]
    
    # Overall test time range.
    base_time_test = cp.minimum(test_event_times.min(), test_spike_times.min())
    max_time_test  = cp.maximum(test_event_times.max(), test_spike_times.max())
    num_bins_test = int(cp.floor((max_time_test - base_time_test - BIN_SIZE) / BIN_INC).get()) + 1
    
    # Compute test bin boundaries.
    bin_left_edges_test = base_time_test + cp.arange(num_bins_test) * BIN_INC
    bin_right_edges_test = bin_left_edges_test + BIN_SIZE
    
    # Use the trained templates ordering.
    templates = trained_params['templates']
    num_templates = len(templates)
    
    # Build the binned spike count matrix.
    binned_counts_test = cp.zeros((num_bins_test, num_templates), dtype=cp.int32)
    for i, tmpl in enumerate(templates):
        tmpl_spikes_test = test_spike_times[test_spike_templates == tmpl]
        start_idx_test = cp.searchsorted(tmpl_spikes_test, bin_left_edges_test, side='left')
        end_idx_test   = cp.searchsorted(tmpl_spikes_test, bin_right_edges_test, side='left')
        binned_counts_test[:, i] = end_idx_test - start_idx_test

    # Compute test event bin indices.
    test_event_bin_idx = ((test_event_times + test_lookahead * 30 - base_time_test) // BIN_INC).astype(cp.int32)
    valid_test = test_event_bin_idx < num_bins_test
    if not bool(cp.all(valid_test)):
        print("Warning: Some test events fall outside the binned range and will be ignored.")
        test_event_bin_idx = test_event_bin_idx[valid_test]
        test_event_labels = test_event_labels[valid_test]
    
    X_test = binned_counts_test[test_event_bin_idx]
    y_test = test_event_labels
    X_test = cp.asnumpy(X_test)
    y_test = cp.asnumpy(y_test)
    
    # Transform features and compute accuracy.
    X_test_norm = trained_params['scaler'].transform(X_test)
    y_pred = trained_params['clf'].predict(X_test_norm)
    accuracy = np.mean(y_pred == y_test) * 100
    return accuracy

# --------------------------------------------------
# Main Code
# --------------------------------------------------
# Define directories.
base_dir       = Path('C:/', 'SGL_DATA', '01_27_p1_templategeneration_g0')
decoder_dir    = base_dir / "decoder_input"
ks_train_dir   = base_dir / "kilosort4_train"   # For OSS decoder training.
ks_dir         = base_dir / "kilosort4_test"          # For KS decoder training & testing.
oss_test_dir   = base_dir / "cuda_output"          # For OSS decoder testing.

# List eventfiles (used for both training event aggregation and testing).
eventfile_names = ["eventfile.txt"]
eventfile_paths = [decoder_dir / name for name in eventfile_names]

# Training and testing parameters.
training_start = 0
training_end   = 27136946
test_start     = 0
test_end       = 17806296

if Path(f"{base_dir / 'ks_heatmap'}.npy").exists():
    ks_results = np.load(f"{base_dir / 'ks_heatmap'}.npy")
    oss_results = np.load(f"{base_dir / 'oss_heatmap'}.npy")
    train_la_values = np.arange(0, 201, 10)  # 0, 10, ..., 200 ms
    test_la_values  = np.arange(-600, 1401, 10)  # -600, -590, ..., 1400 ms
else:
    # Helper function to load event data.
    def load_event_data(event_file_path):
        event_times = []
        event_labels = []
        with open(event_file_path, 'r') as f:
            for line in f:
                tokens = line.strip().split()
                event_times.append(float(tokens[0]))
                event_labels.append(int(tokens[1]))
        return np.array(event_times), np.array(event_labels)

    # Aggregate training event data from all eventfiles.
    all_train_event_times = []
    all_train_event_labels = []
    for event_file in eventfile_paths:
        ev_times, ev_labels = load_event_data(event_file)
        all_train_event_times.append(ev_times)
        all_train_event_labels.append(ev_labels)
    all_train_event_times = np.concatenate(all_train_event_times)
    all_train_event_labels = np.concatenate(all_train_event_labels)

    # Load spikes for training.
    # For KS decoder: train on spikes from kilosort4.
    print(f"Loading KS training spikes from {ks_dir}")
    ks_spike_times       = np.load(ks_dir / "spike_times.npy")
    ks_spike_templates   = np.load(ks_dir / "spike_templates.npy")
    ks_spike_amplitudes  = np.load(ks_dir / "amplitudes.npy")

    # For OSS decoder: train on spikes from kilosort4_train.
    print(f"Loading LSS training spikes from {ks_train_dir}")
    ks_train_spike_times      = np.load(ks_train_dir / "spike_times.npy")
    ks_train_spike_templates  = np.load(ks_train_dir / "spike_templates.npy")
    ks_train_spike_amplitudes = np.load(ks_train_dir / "amplitudes.npy")

    # Define the range for training lookahead values (outer loop) and test lookahead values (inner loop).
    train_la_values = np.arange(0, 201, 10)  # 0, 10, ..., 200 ms
    test_la_values  = np.arange(-600, 1401, 10)  # -600, -590, ..., 1400 ms

    n_train = len(train_la_values)
    n_test  = len(test_la_values)

    # Initialize 2D result matrices.
    ks_results  = np.zeros((n_train, n_test))
    oss_results = np.zeros((n_train, n_test))

    # Loop over training lookahead values.
    for i, train_la in enumerate(train_la_values):
        print(f"Training lookahead: {train_la} ms")
        # Train decoders using current training lookahead.
        print("Training KS decoder (using kilosort4 spikes)...")
        trained_KS_decoder = train_decoder(ks_spike_times, ks_spike_templates, ks_spike_amplitudes,
                                           all_train_event_times, all_train_event_labels,
                                           training_start, training_end,
                                           train_lookahead=train_la)

        print("Training LSS decoder (using kilosort4_train spikes)...")
        trained_OSS_decoder = train_decoder(ks_train_spike_times, ks_train_spike_templates, ks_train_spike_amplitudes,
                                            all_train_event_times, all_train_event_labels,
                                            training_start, training_end,
                                            train_lookahead=train_la)
        
        # Load test spikes (these are loaded in each outer loop iteration, but if desired, you can move them outside
        # if the data files remain constant).
        print(f"Loading KS test spikes from {ks_dir}")
        ks_test_spike_times       = np.load(ks_dir / "spike_times.npy")
        ks_test_spike_templates   = np.load(ks_dir / "spike_templates.npy")
        ks_test_spike_amplitudes  = np.load(ks_dir / "amplitudes.npy")
        
        print(f"Loading LSS test spikes from {oss_test_dir}")
        oss_spike_times = []
        oss_spike_templates = []
        oss_spike_amplitudes = []
        # Note: Adjust the file name below if needed.
        with open(oss_test_dir / 'spikeOutput.txt', 'r') as f:
            for line in f:
                tokens = line.split(',')
                oss_spike_times.append(int(tokens[0]))
                oss_spike_templates.append(int(tokens[1]))
                oss_spike_amplitudes.append(float(tokens[2]))
        oss_spike_times = np.array(oss_spike_times)
        oss_spike_templates = np.array(oss_spike_templates)
        oss_spike_amplitudes = np.array(oss_spike_amplitudes)

        # Loop over test lookahead values.
        for j, test_la in enumerate(test_la_values):
            ks_acc_list  = []
            oss_acc_list = []
            print(f"  Testing at test lookahead: {test_la} ms over {len(eventfile_paths)} eventfiles")
            
            for event_file in eventfile_paths:
                event_times, event_labels = load_event_data(event_file)
                
                # Evaluate KS decoder on kilosort4 test spikes.
                acc_ks = test_decoder(trained_KS_decoder,
                                      ks_test_spike_times, ks_test_spike_templates, ks_test_spike_amplitudes,
                                      event_times, event_labels,
                                      test_start, test_end, test_la)
                # Evaluate OSS decoder on OSS test spikes.
                acc_oss = test_decoder(trained_OSS_decoder,
                                       oss_spike_times, oss_spike_templates, oss_spike_amplitudes,
                                       event_times, event_labels,
                                       test_start, test_end, test_la)
                ks_acc_list.append(acc_ks)
                oss_acc_list.append(acc_oss)
            
            # Average accuracy over eventfiles and store in the matrices.
            ks_results[i, j]  = np.mean(ks_acc_list)
            oss_results[i, j] = np.mean(oss_acc_list)
            print(f"    (Train_la={train_la} ms, Test_la={test_la} ms) -> KS Acc: {ks_results[i, j]:.2f}%, LSS Acc: {oss_results[i, j]:.2f}%")

    # --------------------------------------------------
    # Create Heatmaps
    # --------------------------------------------------
    np.save(f"{base_dir / 'ks_heatmap'}", ks_results)
    np.save(f"{base_dir / 'oss_heatmap'}", oss_results)

common_vmin = min(ks_results.min(), oss_results.min())
common_vmax = max(ks_results.max(), oss_results.max())

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# For the heatmaps, we set the x-axis to be test lookahead and the y-axis training lookahead.
extent = [test_la_values[0], test_la_values[-1], train_la_values[0], train_la_values[-1]]

im1 = axes[0].imshow(ks_results, origin='lower', aspect='auto', extent=extent, vmin=common_vmin, vmax=common_vmax)
axes[0].set_xlabel("Test Lookahead (ms)")
axes[0].set_ylabel("Training Lookahead (ms)")
axes[0].set_title("KS Decoder Accuracy (%)")
fig.colorbar(im1, ax=axes[0])

im2 = axes[1].imshow(oss_results, origin='lower', aspect='auto', extent=extent, vmin=common_vmin, vmax=common_vmax)
axes[1].set_xlabel("Test Lookahead (ms)")
axes[1].set_ylabel("Training Lookahead (ms)")
axes[1].set_title("LSS Decoder Accuracy (%)")
fig.colorbar(im2, ax=axes[1])

plt.suptitle("Heatmaps of Test Accuracy vs. Training/Test Lookahead Offsets")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
