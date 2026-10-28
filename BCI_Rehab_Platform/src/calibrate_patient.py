import sys
import os
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import joblib
from mne.decoding import CSP

from preprocess import load_single_subject_for_calibration
from model import EEGNet


def calibrate_new_patient(patient_id, dataset_type):
    print(f"--- CALIBRATING AI FOR PATIENT {patient_id} ({dataset_type.upper()}) ---")

    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
    patient_dir = os.path.join(models_dir, "patients", f"PAT-{patient_id}")
    os.makedirs(patient_dir, exist_ok=True)

    if dataset_type == 'stroke':
        print("Stroke Profile detected. Loading Screened Patient (Subject 3) model...")
        source_model = os.path.join(models_dir, "patients", "PAT-STROKE-3", "stroke_pipeline.joblib")
        if os.path.exists(source_model):
            shutil.copy(source_model, os.path.join(patient_dir, "stroke_pipeline.joblib"))
            print("SUCCESS: Stroke Pipeline copied.")
        else:
            print("Error: Stroke Pipeline not found. Run save_stroke_model.py first.")
        return

    # --- HEALTHY PATIENT LOGIC ---
    print("Healthy Profile detected. Collecting 5 minutes of EEG data...")
    X, y = load_single_subject_for_calibration(3)
    X, y = X[:60], y[:60]

    print("Computing Spatial Filters (CSP)...")
    csp = CSP(n_components=4, reg=None, log=None, transform_into='csp_space')
    csp.fit(X, y)
    X_csp = csp.transform(X)

    for i in range(X_csp.shape[0]):
        X_csp[i] = (X_csp[i] - np.mean(X_csp[i])) / (np.std(X_csp[i]) + 1e-5)

    print("Training EEGNet (50 epochs)...")
    X_t = torch.tensor(np.expand_dims(X_csp, axis=1), dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.long)
    train_loader = DataLoader(TensorDataset(X_t, y_t), batch_size=16, shuffle=True)

    device = torch.device("cpu")
    model = EEGNet(nb_classes=2, Chans=4, Samples=X_csp.shape[2], F1=8, D=2, F2=16, dropoutRate=0.5).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(50):
        model.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    joblib.dump(csp, os.path.join(patient_dir, "csp_transform.joblib"))
    torch.save(model.state_dict(), os.path.join(patient_dir, "eegnet_model.pth"))
    print(f"SUCCESS: EEGNet saved to {patient_dir}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        calibrate_new_patient(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else 'healthy')
    else:
        print("Error: No Patient ID provided.")