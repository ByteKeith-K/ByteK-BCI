import mne
import os
import glob

STROKE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "stroke")

def find_stroke_events():
    edf_search_path = os.path.join(STROKE_DIR, "**", "*.edf")
    edf_files = glob.glob(edf_search_path, recursive=True)

    if not edf_files:
        print("No .edf files found.")
        return

    first_file = edf_files[0]
    print(f"Loading {first_file}...")
    raw = mne.io.read_raw_edf(first_file, preload=True, verbose=False)

    print("\n--- SEARCHING FOR HIDDEN TRIGGERS ---")

    # 1. Check raw annotations
    print(f"1. Raw Annotations: {raw.annotations}")

    print("\n2. Channel Types:")
    for i, ch in enumerate(raw.info['chs']):
        if ch['kind'] == mne.io.constants.FIFF.FIFFV_STIM_CH:
            print(f"   Found STIM channel: {raw.ch_names[i]}")
        elif ch['kind'] == mne.io.constants.FIFF.FIFFV_MISC_CH:
            print(f"   Found MISC channel: {raw.ch_names[i]}")

    print("\n3. Checking data in the last channel:")
    last_ch_data = raw.get_data()[-1, :1000]  # first 1000 samples
    unique_vals = set(last_ch_data)
    print(f"   Unique values in the last channel: {unique_vals}")

    print("\n4. Checking if a .tsv file exists in the folder...")
    folder_path = os.path.dirname(first_file)
    tsv_files = [f for f in os.listdir(folder_path) if f.endswith('.tsv')]
    if tsv_files:
        print(f"   Found event files: {tsv_files}")
    else:
        print("   No .tsv files found.")


if __name__ == "__main__":
    find_stroke_events()


