import torch, json, os, argparse
from torch.utils.data import Dataset, DataLoader
import numpy as np
from btbench_config import *
from braintreebank_subject import Subject
from btbench_datasets import BrainTreebankSubjectTrialBenchmarkDataset
import scipy.signal as signal
from scipy import stats

# all possible pairs of (subject_id, trial_id)
all_subject_trials = [(1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (3, 0), (3, 1), (3, 2), (4, 0), (4, 1), (4, 2), (5, 0), (6, 0), (6, 1), (6, 4), (7, 0), (7, 1), (8, 0), (9, 0), (10, 0), (10, 1)]
def process_subject(subject_id, save_dir=SAVE_SPEECH_SELECTIVITY_DATA_DIR, verbose=True):
    """Process speech selectivity for a given subject.

    This function analyzes neural data to identify electrodes that show selective responses 
    to speech vs non-speech stimuli. It:
    1. Loads neural data for all trials of the subject
    2. Computes electrode activities during speech and non-speech conditions. Quantified as total power (in dB) in the high-gamma band (70-300 Hz).
    3. Performs statistical testing (t-test) to identify speech-selective electrodes
    4. Applies Benjamini-Hochberg FDR correction for multiple comparisons

    Args:
        subject_id (int): ID of the subject to process
        save_dir (str, optional): Directory to save results. Defaults to "btbench_speech_selectivity".
        verbose (bool, optional): Whether to print progress messages. Defaults to True.

    Returns:
        None: Results are saved to the specified save_dir
    """
    high_gamma_min_freq, high_gamma_max_freq = 70, 300

    # Step 1. Load all datasets for this subject and trial
    subject = Subject(subject_id, cache=True)
    all_electrode_activities = []
    all_speech_nonspeech_condition = []
    for trial_id in [trial_id for (_subject_id, trial_id) in all_subject_trials if _subject_id == subject_id]:
        if verbose: print(f"Loading data for subject {subject_id}, trial {trial_id}...")
        dataset = BrainTreebankSubjectTrialBenchmarkDataset(subject, trial_id, dtype=torch.float32, eval_name="speech") # eval_name can be "pitch", "rms", "onset", or "speech"

        # Compute electrode activities and speech/non-speech conditions
        n_electrodes = dataset[0][0].shape[0]
        electrode_activities = np.zeros((len(dataset), n_electrodes)) 
        speech_nonspeech_condition = np.zeros(len(dataset)) # 2 condiions: speech and non-speech
        for i in range(len(dataset)):
            features, label = dataset[i]
            f, t, Sxx = signal.spectrogram(features, fs=2048, nperseg=256, noverlap=0, window='boxcar')
            # Bin #4 corresponds to 125 ms right after the word onset
            electrode_activities[i, :] = Sxx[:, (f>high_gamma_min_freq) & (f<high_gamma_max_freq), 4].sum(axis=1) 
            speech_nonspeech_condition[i] = label

        all_electrode_activities.append(electrode_activities)
        all_speech_nonspeech_condition.append(speech_nonspeech_condition)
        subject.unload_neural_data(trial_id) # unload neural data from RAM after loading it into the dataset
    electrode_activities = np.concatenate(all_electrode_activities, axis=0)
    speech_nonspeech_condition = np.concatenate(all_speech_nonspeech_condition)
    
    if verbose: print(f"Computing p-values for each electrode...")
    # Step 3. Compute p-values for each electrode
    p_values_ttest = np.zeros(n_electrodes)
    for electrode in range(n_electrodes):
        speech_values = 10 * np.log10(electrode_activities[speech_nonspeech_condition == 1, electrode] + 1e-10)
        nonspeech_values = 10 * np.log10(electrode_activities[speech_nonspeech_condition == 0, electrode] + 1e-10)
        _, p_value = stats.ttest_ind(speech_values, nonspeech_values)
        p_values_ttest[electrode] = p_value

    if verbose: print(f"Applying Benjamini-Hochberg FDR correction...")
    # Apply Benjamini-Hochberg FDR correction
    sorted_p_idx = np.argsort(p_values_ttest)
    sorted_p_values = p_values_ttest[sorted_p_idx]
    fdr_level = 0.05
    fdr_thresholds = np.arange(1, n_electrodes + 1) * fdr_level / n_electrodes
    significant_idx = np.where(sorted_p_values <= fdr_thresholds)[0]
    p_values_ttest_corrected = np.zeros(n_electrodes)
    if len(significant_idx) > 0:
        critical_idx = significant_idx[-1]
        critical_threshold = sorted_p_values[critical_idx]
        p_values_ttest_corrected = np.minimum(p_values_ttest * n_electrodes / (sorted_p_idx + 1), 1.0)
    else:
        critical_threshold = 0.0
        p_values_ttest_corrected = np.ones(n_electrodes)

    if verbose: print(f"Saving metadata...")
    os.makedirs(save_dir, exist_ok=True)
    metadata = {
        "subject_id": subject_id,
        "electrode_p_values": p_values_ttest.tolist(),
        "electrode_p_values_corrected": p_values_ttest_corrected.tolist(),
        "fdr_threshold": float(critical_threshold),
        "fdr_level": fdr_level
    }
    output_file = f"{save_dir}/subject{subject_id}_stats.json"
    with open(output_file, "w") as f: json.dump(metadata, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--subject', type=int, help='Subject ID to process. If not specified, processes all subjects.')
    args = parser.parse_args()
    
    if args.subject is not None: all_subject_ids = [args.subject]
    else: all_subject_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    for subject_id in all_subject_ids:
        print(f"Processing subject {subject_id}...")
        process_subject(subject_id)