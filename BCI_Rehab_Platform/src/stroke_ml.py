import numpy as np
import os
import scipy.io
import scipy.signal as signal
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from mne.decoding import CSP
from sklearn.metrics import accuracy_score


def load_single_stroke_patient(subject_num):
    filepath = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "data", "stroke", f"sub-{subject_num:02d}",
                            f"sub-{subject_num:02d}_task-motor-imagery_eeg.mat")
    if not os.path.exists(filepath): return None, None
    mat = scipy.io.loadmat(filepath)
    eeg_data = mat['eeg'][0][0][0]
    labels = mat['eeg'][0][0][1].flatten()

    X = eeg_data[:, :-1, :]
    y = np.where(labels == 1, 0, 1)

    # Filter 8-30 Hz
    b, a = signal.butter(4, [8, 30], btype='bandpass', fs=500)
    for i in range(X.shape[0]):
        X[i] = signal.filtfilt(b, a, X[i], axis=1)

    # Downsample to 250 Hz and crop to 3s (750 points)
    X = X[:, :, ::2]
    X = X[:, :, 250:1000]
    return X, y


def screen_patients():
    print("--- SCREENING STROKE PATIENTS (CSP + SVM) ---")

    results = {}

    for subject in range(1, 11):
        X, y = load_single_stroke_patient(subject)
        if X is None: continue

        clf = Pipeline([
            ('CSP', CSP(n_components=4, reg='ledoit_wolf', log=True, norm_trace=False)),
            ('SVM', SVC(kernel='linear', C=1.0))
        ])

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        accs = []

        for train_idx, test_idx in skf.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            clf.fit(X_train, y_train)
            preds = clf.predict(X_test)
            accs.append(accuracy_score(y_test, preds) * 100)

        avg_acc = np.mean(accs)
        results[subject] = avg_acc
        print(f"Patient {subject:02d} Accuracy: {avg_acc:.2f}%")

    # Find the best patient
    best_patient = max(results, key=results.get)
    print(f"\n=========================================")
    print(f"BEST PATIENT: Subject {best_patient} with {results[best_patient]:.2f}% accuracy")
    print(f"=========================================")


if __name__ == "__main__":
    screen_patients()