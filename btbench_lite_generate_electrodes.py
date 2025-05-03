import random
import json
import os
import re
from braintreebank_subject import BrainTreebankSubject

def get_probe_name(electrode_label):
    """Remove trailing digits to get the probe name."""
    return re.sub(r'\d+$', '', electrode_label)

def main():
    # Set random seed for reproducibility
    random.seed(42)

    # List of subject IDs to process
    subject_ids = [1, 2, 3, 4, 7, 10]

    # Load all subjects and collect their electrode labels
    subject_electrode_map = {}
    for sid in subject_ids:
        subj = BrainTreebankSubject(sid)
        subject_electrode_map[sid] = list(subj.electrode_labels)

    # Prepare kept electrodes per subject, using probe-wise proportional removal
    kept_electrodes_per_subject = {}
    for sid, electrodes in subject_electrode_map.items():
        n_total = len(electrodes)
        if n_total <= 120:
            kept = list(electrodes)
        else:
            # Group electrodes by probe
            probe_map = {}
            for e in electrodes:
                probe = get_probe_name(e)
                probe_map.setdefault(probe, []).append(e)
            proportion_to_keep = 120 / n_total
            kept = []
            # For each probe, keep the proportion
            for probe, probe_electrodes in probe_map.items():
                n_probe = len(probe_electrodes)
                n_keep_probe = int(round(n_probe * proportion_to_keep))
                # Ensure at least 1 is kept if probe is non-empty and n_keep_probe > 0
                if n_probe > 0 and n_keep_probe == 0 and proportion_to_keep > 0:
                    n_keep_probe = 1
                if n_keep_probe > n_probe:
                    n_keep_probe = n_probe
                kept_probe = random.sample(probe_electrodes, n_keep_probe)
                kept.extend(kept_probe)
            # Adjust if rounding error means we have too many/few
            if len(kept) > 120:
                kept = random.sample(kept, 120)
            elif len(kept) < 120:
                # Add back some from the remaining electrodes at random
                removed = [e for e in electrodes if e not in kept]
                to_add = 120 - len(kept)
                add_back = random.sample(removed, to_add)
                kept.extend(add_back)
        kept_electrodes_per_subject[f"btbank{sid}"] = kept

    # Save the results in the 'btbench-lite' folder
    output_dir = "btbench-lite"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "btbench_lite_electrodes.json")
    with open(output_path, "w") as f:
        json.dump(kept_electrodes_per_subject, f, indent=2)

    print(f"Saved kept electrodes per subject in '{output_path}'.")

if __name__ == "__main__":
    main() 