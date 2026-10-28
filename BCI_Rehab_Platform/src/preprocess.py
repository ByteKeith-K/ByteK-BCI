import mne
import numpy as np
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
DATA_DIR = os.path.join(project_root, "data")


def create_epochs_for_subject(filepath):
    raw = mne.io.read_raw_gdf(filepath, preload=True, verbose=False)
    raw.drop_channels(['EOG-left', 'EOG-central', 'EOG-right'], on_missing='ignore')

    event_id = {'769': 769, '770': 770}
    events, _ = mne.events_from_annotations(raw, event_id=event_id, verbose=False)

    tmin, tmax = 1.0, 4.0
    raw.filter(8.0, 30.0, fir_design='firwin', verbose=False)

    epochs = mne.Epochs(raw, events, event_id, tmin, tmax,
                        baseline=None, preload=True, verbose=False)

    X = epochs.get_data()
    y = epochs.events[:, -1]
    y = np.where(y == 769, 0, 1)

    return X, y


def load_single_subject_for_calibration(subject_num):
    filename = f"A0{subject_num}T.gdf"
    filepath = os.path.join(DATA_DIR, filename)
    print(f"Processing {filename}...")
    return create_epochs_for_subject(filepath)


def load_evaluation_subject(subject_num):
    filename = f"A0{subject_num}E.gdf"
    filepath = os.path.join(DATA_DIR, filename)
    print(f"Processing {filename}...")

    raw = mne.io.read_raw_gdf(filepath, preload=True, verbose=False)
    raw.drop_channels(['EOG-left', 'EOG-central', 'EOG-right'], on_missing='ignore')
    events_all, event_dict = mne.events_from_annotations(raw, verbose=False)

    if not event_dict:
        print("No text annotations found. Falling back to stim channel 'STI 014'...")
        # In the evaluation files, the triggers are often in the stim channel
        events = mne.find_events(raw, stim_channel='STI 014', verbose=False)
        # Filter to keep only Left (769) and Right (770)
        events = events[np.isin(events[:, 2], [769, 770])]
        event_id = {'left': 769, 'right': 770}
    else:
        print(f"Found annotations: {event_dict}")
        # Find the exact string keys for 769 and 770
        left_key = [k for k in event_dict.keys() if '769' in k]
        right_key = [k for k in event_dict.keys() if '770' in k]

        if not left_key or not right_key:
            print(f"Unique annotations found: {set(raw.annotations.description)}")
            raise ValueError("Could not find 769 or 770 in annotations.")

        event_id = {left_key[0]: 769, right_key[0]: 770}
        events, _ = mne.events_from_annotations(raw, event_id=event_id, verbose=False)

    tmin, tmax = 1.0, 4.0
    raw.filter(8.0, 30.0, fir_design='firwin', verbose=False)
    epochs = mne.Epochs(raw, events, event_id=event_id, tmin=tmin, tmax=tmax,
                        baseline=None, preload=True, verbose=False)

    X = epochs.get_data()
    y = epochs.events[:, -1]
    y = np.where(y == 769, 0, 1)

    return X, y