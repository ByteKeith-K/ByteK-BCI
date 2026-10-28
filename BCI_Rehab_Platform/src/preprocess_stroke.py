import scipy.io
import numpy as np
import scipy.signal as signal
import os


def load_stroke_patient(subject_num):
    filepath = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "data", "stroke", f"sub-{subject_num:02d}",
                            f"sub-{subject_num:02d}_task-motor-imagery_eeg.mat")

    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return None, None

    mat = scipy.io.loadmat(filepath)
    eeg_data = mat['eeg'][0][0][0]
    labels = mat['eeg'][0][0][1].flatten()

    X = eeg_data[:, :-1, :]  # Drop trigger channel
    y = np.where(labels == 1, 0, 1)  # 1=Left(0), 2=Right(1)

    # Filter 8-30 Hz
    b, a = signal.butter(4, [8, 30], btype='bandpass', fs=500)
    for i in range(X.shape[0]):
        X[i] = signal.filtfilt(b, a, X[i], axis=1)

    # Downsample to 250 Hz and crop to 3s
    X = X[:, :, ::2]
    X = X[:, :, 250:1001]

    return X, y


def load_combined_stroke_data(max_patients=5):
    all_X = []
    all_y = []

    for i in range(1, max_patients + 1):
        X, y = load_stroke_patient(i)
        if X is not None:
            all_X.append(X)
            all_y.append(y)

    if not all_X:
        return None, None

    X_combined = np.concatenate(all_X, axis=0)
    y_combined = np.concatenate(all_y, axis=0)

    print(f"Loaded {max_patients} patients. Total trials: {X_combined.shape[0]}")

    print("Applying Data Augmentation (Adding Gaussian Noise)...")
    noise = np.random.normal(0, 0.5, X_combined.shape)
    X_aug = X_combined + noise

    X_final = np.concatenate((X_combined, X_aug), axis=0)
    y_final = np.concatenate((y_combined, y_combined), axis=0)

    print(f"Augmented Dataset Size: {X_final.shape[0]} trials")
    return X_final, y_final