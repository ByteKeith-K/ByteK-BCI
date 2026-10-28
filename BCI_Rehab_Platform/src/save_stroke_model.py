import numpy as np
import os
import scipy.io
import scipy.signal as signal
import joblib
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from mne.decoding import CSP
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score


def load_stroke_patient(subject_num):
    filepath = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "data", "stroke", f"sub-{subject_num:02d}",
                            f"sub-{subject_num:02d}_task-motor-imagery_eeg.mat")

    if not os.path.exists(filepath):
        return None, None

    mat = scipy.io.loadmat(filepath)
    eeg_data = mat['eeg'][0][0][0]
    labels = mat['eeg'][0][0][1].flatten()

    X = eeg_data[:, :-1, :]
    y = np.where(labels == 1, 0, 1)

    b, a = signal.butter(4, [8, 30], btype='bandpass', fs=500)
    for i in range(X.shape[0]):
        X[i] = signal.filtfilt(b, a, X[i], axis=1)

    X = X[:, :, ::2]
    X = X[:, :, 250:1000]
    return X, y


def train_and_save_screened_patient():
    print("--- TRAINING FINAL MODEL FOR SCREENED STROKE PATIENT (Subject 3) ---")

    X, y = load_stroke_patient(3)
    if X is None:
        print("Data not found.")
        return

    print(f"Loaded Subject 3. Data shape: {X.shape}")

    # 1. Verify accuracy using 5-Fold CV (The scientific standard)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_accs = []

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Using CalibratedClassifierCV to get probabilities without warnings
        base_svm = SVC(kernel='linear', C=1.0)
        calibrated_svm = CalibratedClassifierCV(base_svm, method='sigmoid', cv=3)

        clf = Pipeline([
            ('CSP', CSP(n_components=4, reg='ledoit_wolf', log=True, norm_trace=False)),
            ('SVM', calibrated_svm)
        ])

        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        fold_accs.append(accuracy_score(y_test, preds) * 100)

    print(f"5-Fold CV Verified Accuracy: {np.mean(fold_accs):.2f}%")

    # 2. Train the Final Deployment Model on 100% of the data
    print("\nTraining Final Deployment Model on all 40 trials...")

    base_svm = SVC(kernel='linear', C=1.0)
    calibrated_svm = CalibratedClassifierCV(base_svm, method='sigmoid', cv=3)

    final_clf = Pipeline([
        ('CSP', CSP(n_components=4, reg='ledoit_wolf', log=True, norm_trace=False)),
        ('SVM', calibrated_svm)
    ])

    final_clf.fit(X, y)

    # 3. Save the model
    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "patients",
                              "PAT-STROKE-3")
    os.makedirs(models_dir, exist_ok=True)

    model_path = os.path.join(models_dir, "stroke_pipeline.joblib")
    joblib.dump(final_clf, model_path)

    print(f"SUCCESS: Final Stroke Pipeline saved to {model_path}")
    print("You can now use 'Stroke Brain' in the ByteK BCI dashboard!")


if __name__ == "__main__":
    train_and_save_screened_patient()