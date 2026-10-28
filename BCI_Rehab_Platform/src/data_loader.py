import mne
import matplotlib.pyplot as plt
import os

current_dir = os.path.dirname(os.path.abspath(__file__))

project_root = os.path.dirname(current_dir)

DATA_PATH = os.path.join(project_root, "data", "A01T.gdf")


def load_and_visualize():
    if not os.path.exists(DATA_PATH):
        print(f"ERROR: Could not find {DATA_PATH}. Please check your data folder.")
        return

    print("1. Loading raw EEG data...")
    raw = mne.io.read_raw_gdf(DATA_PATH, preload=True)

    print("Original Data Info:")
    print(raw.info)

    print("\n2. Applying Bandpass Filter (8.0 - 30.0 Hz)...")
    raw_filtered = raw.copy().filter(8.0, 30.0, fir_design='firwin')

    print("\n3. Plotting the brainwaves...")
    print("A window should pop up. You can scroll and zoom.")

    raw_filtered.plot(duration=10, n_channels=15, scalings='auto', title="Filtered EEG (8-30Hz) - Subject 1")

    plt.show()


if __name__ == "__main__":
    load_and_visualize()