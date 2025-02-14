# NOTE: Settings in this file have global effect on the code. All parts of the pipeline have to run with the same settings.
# If you want to change a setting, you have to rerun all parts of the pipeline with the new setting. Otherwise, things will break.

ROOT_DIR = "braintreebank" # Root directory for the braintreebank data
SAVE_SUBJECT_TRIAL_DF_DIR = "subject_metadata"
SAVE_SPEECH_SELECTIVITY_DATA_DIR = "speech_selectivity_data"
SAMPLING_RATE = 2048 # Sampling rate

N_PER_SEG = 256
SPECTROGRAM_DIMENSIONALITY = 128 # number of frequency bins in the spectrogram

START_NEURAL_DATA_BEFORE_WORD_ONSET = 0.5 # in seconds
END_NEURAL_DATA_AFTER_WORD_ONSET = 2 # in seconds
NEURAL_DATA_NONVERBAL_WINDOW_PADDING_TIME = 2 # how many seconds to wait between the last word off-set and the start of a "non-verbal" chunk
NEURAL_DATA_NONVERBAL_WINDOW_OVERLAP = 0.8 # proportion of overlap between consecutive nonverbal chunks (0 means no overlap)

# some sanity check code as well as disabling file locking for HDF5 files
assert NEURAL_DATA_NONVERBAL_WINDOW_OVERLAP >= 0 and NEURAL_DATA_NONVERBAL_WINDOW_OVERLAP < 1, "NONVERBAL_CONSECUTIVE_CHUNKS_OVERLAP must be between 0 and 1, strictly below 1"
import os; os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE" # Disable file locking for HDF5 files. This is helpful for parallel processing.