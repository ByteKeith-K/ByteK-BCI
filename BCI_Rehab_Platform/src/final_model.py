import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import joblib  # New library to save CSP
from mne.decoding import CSP

from preprocess import load_single_subject_for_calibration
from model import EEGNet


def train_production_model():
    print("--- TRAINING FINAL PRODUCTION MODEL ---")

    X, y = load_single_subject_for_calibration(1)

    # 1. Apply CSP on the entire dataset
    print("Fitting CSP...")
    csp = CSP(n_components=4, reg=None, log=None, transform_into='csp_space')
    csp.fit(X, y)
    X_csp = csp.transform(X)

    # 2. Normalize
    for i in range(X_csp.shape[0]):
        X_csp[i] = (X_csp[i] - np.mean(X_csp[i])) / (np.std(X_csp[i]) + 1e-5)

    # 3. Reshape for PyTorch
    X_t = torch.tensor(np.expand_dims(X_csp, axis=1), dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.long)

    train_dataset = TensorDataset(X_t, y_t)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    device = torch.device("cpu")
    model = EEGNet(nb_classes=2, Chans=4, Samples=X_csp.shape[2],
                   F1=8, D=2, F2=16, dropoutRate=0.5).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

    epochs = 100
    print("Training production model...")
    for epoch in range(epochs):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        scheduler.step()

    # 4. Save BOTH the CSP model and the EEGNet model
    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
    os.makedirs(models_dir, exist_ok=True)

    # Save CSP using joblib
    csp_path = os.path.join(models_dir, "csp_transform.joblib")
    joblib.dump(csp, csp_path)

    # Save EEGNet
    eegnet_path = os.path.join(models_dir, "eegnet_production.pth")
    torch.save(model.state_dict(), eegnet_path)

    print(f"SUCCESS! Saved CSP to {csp_path}")
    print(f"SUCCESS! Saved EEGNet to {eegnet_path}")


if __name__ == "__main__":
    train_production_model()