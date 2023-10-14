import numpy as np

FFT_HOP = 256

NOTES_BINS_PER_SEMITONE = 1
CONTOURS_BINS_PER_SEMITONE = 3
# base frequency of the CENTRAL bin of the first semitone (i.e., the
# second bin if annotations_bins_per_semitone is 3)
ANNOTATIONS_BASE_FREQUENCY = 27.5  # lowest key on a piano
ANNOTATIONS_N_SEMITONES = 88  # number of piano keys
AUDIO_SAMPLE_RATE = 22050
AUDIO_N_CHANNELS = 1
N_FREQ_BINS_NOTES = ANNOTATIONS_N_SEMITONES * NOTES_BINS_PER_SEMITONE
N_FREQ_BINS_CONTOURS = ANNOTATIONS_N_SEMITONES * CONTOURS_BINS_PER_SEMITONE

AUDIO_WINDOW_LENGTH = 2  # duration in seconds of training examples - original 1

ANNOTATIONS_FPS = AUDIO_SAMPLE_RATE // FFT_HOP
ANNOTATION_HOP = 1.0 / ANNOTATIONS_FPS

# ANNOT_N_TIME_FRAMES is the number of frames in the time-frequency representations we compute
ANNOT_N_FRAMES = ANNOTATIONS_FPS * AUDIO_WINDOW_LENGTH
MAX_N_SEMITONES = int(np.floor(12.0 * np.log2(0.5 * AUDIO_SAMPLE_RATE / ANNOTATIONS_BASE_FREQUENCY)))
