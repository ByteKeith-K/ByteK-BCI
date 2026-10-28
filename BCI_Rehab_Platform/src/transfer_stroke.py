import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
import numpy as np
import os
import joblib
from mne.decoding import CSP

from preprocess_stroke import load_combined_stroke_data
from model import EEGNet


def run_transfer_learning():
    print("--- RUNNING TRANSFER LEARNING ON STROKE DATA ---")

    # 1. Load the Stroke Data (Patients 1-5, with Augmentation)
    X, y = load_combined_stroke_data(max_patients=5)
    if X is None: return

    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
    healthy_base_model_path = os.path.join(models_dir, "eegnet_production.pth")

    if not os.path.exists(healthy_base_model_path):
        print("Error: Healthy base model not found. Run final_model.py first.")
        return

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_accs = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        print(f"\n--- Fine-Tuning Fold {fold + 1}/5 ---")
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # 2. Fit a new CSP specifically for the stroke data
        csp = CSP(n_components=4, reg=None, log=None, transform_into='csp_space')
        csp.fit(X_train, y_train)

        X_train_csp = csp.transform(X_train)
        X_test_csp = csp.transform(X_test)

        # Normalize
        def normalize(X):
            for i in range(X.shape[0]):
                X[i] = (X[i] - np.mean(X[i])) / (np.std(X[i]) + 1e-5)
            return X

        X_train_csp = normalize(X_train_csp)
        X_test_csp = normalize(X_test_csp)

        X_train_t = torch.tensor(np.expand_dims(X_train_csp, axis=1), dtype=torch.float32)
        y_train_t = torch.tensor(y_train, dtype=torch.long)
        X_test_t = torch.tensor(np.expand_dims(X_test_csp, axis=1), dtype=torch.float32)
        y_test_t = torch.tensor(y_test, dtype=torch.long)

        train_dataset = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

        device = torch.device("cpu")

        # 3. Initialize the model with the same architecture
        model = EEGNet(nb_classes=2, Chans=4, Samples=X_train_csp.shape[2],
                       F1=8, D=2, F2=16, dropoutRate=0.5).to(device)

        print("Loading Healthy Base Model weights...")
        model.load_state_dict(torch.load(healthy_base_model_path, weights_only=True))

        criterion = nn.CrossEntropyLoss()
        # 5. Fine-tuning: We use a MUCH SMALLER learning rate (0.0001) so it doesn't forget what it already knows
        optimizer = optim.Adam(model.parameters(), lr=0.0001)

        epochs = 50
        for epoch in range(epochs):
            model.train()
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        # Evaluate
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test_t)
            _, predicted = torch.max(test_outputs.data, 1)
            acc = 100 * (predicted == y_test_t).sum().item() / y_test_t.size(0)
            fold_accs.append(acc)
            print(f"Fold {fold + 1} Accuracy: {acc:.2f}%")

    avg_acc = np.mean(fold_accs)
    print(f"\n=========================================")
    print(f"TRANSFER LEARNING STROKE ACCURACY: {avg_acc:.2f}%")
    print(f"=========================================")


if __name__ == "__main__":
    run_transfer_learning()