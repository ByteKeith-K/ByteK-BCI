import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
import numpy as np
import os
import joblib
from mne.decoding import CSP

from preprocess import load_single_subject_for_calibration
from model import EEGNet


def train_calibrated_subject2():
    print("--- SUBJECT-SPECIFIC CALIBRATION (Subject 3) ---")

    # 1. Load Subject 3's data
    X, y = load_single_subject_for_calibration(3)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_accs = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        print(f"\n--- Training Fold {fold + 1}/5 ---")
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # 2. Apply CSP specifically for Subject 2
        csp = CSP(n_components=4, reg=None, log=None, transform_into='csp_space')
        csp.fit(X_train, y_train)

        X_train_csp = csp.transform(X_train)
        X_test_csp = csp.transform(X_test)

        # 3. Normalize
        def normalize(X):
            for i in range(X.shape[0]):
                X[i] = (X[i] - np.mean(X[i])) / (np.std(X[i]) + 1e-5)
            return X

        X_train_csp = normalize(X_train_csp)
        X_test_csp = normalize(X_test_csp)

        # 4. Tensors
        X_train_t = torch.tensor(np.expand_dims(X_train_csp, axis=1), dtype=torch.float32)
        y_train_t = torch.tensor(y_train, dtype=torch.long)
        X_test_t = torch.tensor(np.expand_dims(X_test_csp, axis=1), dtype=torch.float32)
        y_test_t = torch.tensor(y_test, dtype=torch.long)

        train_dataset = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

        device = torch.device("cpu")
        model = EEGNet(nb_classes=2, Chans=4, Samples=X_train_csp.shape[2],
                       F1=8, D=2, F2=16, dropoutRate=0.5).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

        epochs = 100
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

        # 5. Evaluate
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test_t)
            _, predicted = torch.max(test_outputs.data, 1)
            acc = 100 * (predicted == y_test_t).sum().item() / y_test_t.size(0)
            fold_accs.append(acc)
            print(f"Fold {fold + 1} Accuracy: {acc:.2f}%")

    avg_acc = np.mean(fold_accs)
    print(f"\n=========================================")
    print(f"CALIBRATED AVERAGE ACCURACY FOR SUBJECT 2: {avg_acc:.2f}%")
    print(f"=========================================")


if __name__ == "__main__":
    train_calibrated_subject2()