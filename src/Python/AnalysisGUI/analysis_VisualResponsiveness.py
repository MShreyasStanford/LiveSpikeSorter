import os
import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import f_oneway

__analysis_metadata__ = {
    "name": "Visual Responsiveness Significance",
    "description": "TODO",
    "parameters": [
        ("spike_file", "FilePath"),
        ("eventfile", "FilePath", r"^(\d+)\s+(\d+)$"),
        ("Lookbehind Start (samples)", "int"),
        ("Lookbehind End (samples)", "int"),
        ("Lookahead Start (samples)", "int"),
        ("Lookahead End (samples)", "int")
    ]
}


def run(base_dir, spike_output, eventfile, 
        lookbehind_start, lookbehind_end, 
        lookahead_start, lookahead_end):
    """
    Perform two-group ANOVA per template (neuron) comparing pre- vs post-event firing.

    For each spike template j:
      - Build two groups:
          1) "before": spike counts in [event_time+lookbehind_start, event_time+lookbehind_end)
          2) "after":  spike counts in [event_time+lookahead_start, event_time+lookahead_end)
      - Perform one-way ANOVA across these two groups (equivalent to a t-test).

    Outputs written into base_dir:
      - 'anova_before_after_results.csv'
          columns: template, F_statistic, p_value
      - 'anova_before_after_pvalue_histogram.png'
      - 'template_<templ>_low_p_before_after.png': mean±SEM plot for up to 5 templates p<0.05
      - 'template_<templ>_high_p_before_after.png': mean±SEM plot for up to 5 templates p>0.9

    Parameters
    ----------
    base_dir : str or Path
        Directory to write output files.
    spike_output : str or Path
        CSV file with columns: spike_time (int), spike_template (int).
    eventfile : str or Path
        Whitespace-delimited file with a single column: event_time (int).
    lookbehind_start : int
        Start offset (in samples) before event_time.
    lookbehind_end : int
        End offset (in samples) before event_time.
    lookahead_start : int
        Start offset (in samples) after event_time.
    lookahead_end : int
        End offset (in samples) after event_time.
    """
    # Prepare output directory
    base_path = Path(base_dir)
    base_path.mkdir(parents=True, exist_ok=True)

    # Load spike data
    spike_path = Path(spike_output)
    if not spike_path.is_absolute():
        spike_path = base_path / spike_path
    spike_times, spike_templates = [], []
    with open(spike_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            try:
                t = int(row[0]); templ = int(row[1])
            except (ValueError, IndexError):
                continue
            spike_times.append(t); spike_templates.append(templ)
    spike_times = np.array(spike_times)
    spike_templates = np.array(spike_templates)

    # Load event times
    evt_path = Path(eventfile)
    if not evt_path.is_absolute():
        evt_path = base_path / eventfile
    event_times = []
    with open(evt_path, 'r') as f:
        for line in f:
            try:
                et = int(line.strip().split()[0])
            except (ValueError, IndexError):
                continue
            event_times.append(et)
    event_times = np.array(event_times)

    # Unique templates
    unique_templates = np.unique(spike_templates)

    # Initialize counts for before and after
    counts = {templ: {'before': [], 'after': []} for templ in unique_templates}

    # Count spikes per event
    for et in event_times:
        # before window
        start_b = et + lookbehind_start
        end_b = et + lookbehind_end
        mask_b = (spike_times >= start_b) & (spike_times < end_b)
        temps_b, cnts_b = np.unique(spike_templates[mask_b], return_counts=True)
        dict_b = dict(zip(temps_b, cnts_b))
        # after window
        start_a = et + lookahead_start
        end_a = et + lookahead_end
        mask_a = (spike_times >= start_a) & (spike_times < end_a)
        temps_a, cnts_a = np.unique(spike_templates[mask_a], return_counts=True)
        dict_a = dict(zip(temps_a, cnts_a))
        # append counts
        for templ in unique_templates:
            counts[templ]['before'].append(dict_b.get(templ, 0))
            counts[templ]['after'].append(dict_a.get(templ, 0))

    # Perform ANOVA per template
    results = []
    for templ in unique_templates:
        grp1 = counts[templ]['before']
        grp2 = counts[templ]['after']
        # need at least 2 samples in each
        if len(grp1) < 2 or len(grp2) < 2:
            continue
        # skip if constant
        if len(set(grp1))==1 and len(set(grp2))==1:
            continue
        try:
            F, p = f_oneway(grp1, grp2)
        except Exception:
            continue
        results.append((templ, F, p))

    # Save results CSV
    out_csv = base_path / 'anova_before_after_results.csv'
    with open(out_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['template', 'F_statistic', 'p_value'])
        for templ, F, p in sorted(results, key=lambda x: x[2]):
            writer.writerow([templ, F, p])

    # Histogram of p-values
    pvals = [p for _,_,p in results]
    plt.figure()
    plt.hist(pvals, bins=50)
    plt.xlabel('p-value')
    plt.ylabel('Count')
    plt.title('ANOVA p-values (before vs after)')
    plt.tight_layout()
    plt.savefig(base_path / 'anova_before_after_pvalue_histogram.png')
    plt.close()

    # Select low/high p templates
    low = [t for t,_,p in sorted(results,key=lambda x:x[2]) if p<0.05][:5]
    high = [t for t,_,p in sorted(results,key=lambda x:x[2],reverse=True) if p>0.9][:5]

    # Plot mean±SEM for two groups
    def plot_two_group(templ, p, category):
        before = np.array(counts[templ]['before'])
        after = np.array(counts[templ]['after'])
        means = [before.mean(), after.mean()]
        sems = [before.std(ddof=1)/np.sqrt(before.size),
                after.std(ddof=1)/np.sqrt(after.size)]
        plt.figure()
        plt.errorbar([0,1], means, yerr=sems, fmt='-o', capsize=5)
        plt.xticks([0,1], ['before','after'])
        plt.ylabel('Mean spike count ± SEM')
        plt.title(f'Template {templ} ({category}), p={p:.3f}')
        plt.tight_layout()
        plt.savefig(base_path / f'template_{templ}_{category}_before_after.png')
        plt.close()

    for templ, _, p in [r for r in results if r[0] in low]:
        plot_two_group(templ, p, 'low_p')
    for templ, _, p in [r for r in results if r[0] in high]:
        plot_two_group(templ, p, 'high_p')

    print(f"ANOVA before/after complete on {len(results)} templates. See {out_csv} and plots.")
