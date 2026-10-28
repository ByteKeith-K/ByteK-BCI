import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
import numpy as np
import os
from mne.decoding import CSP

from preprocess import load_single_subject_for_calibration
from model import EEGNet


def train_csp_eegnet():
    print("--- TRAINING WITH CSP + EEGNET (5-FOLD CV) ---")

    X, y = load_single_subject_for_calibration(1)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_accs = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        print(f"\n--- Training Fold {fold + 1}/5 ---")
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        csp = CSP(n_components=4, reg=None, log=None, transform_into='csp_space')
        csp.fit(X_train, y_train)

        X_train_csp = csp.transform(X_train)
        X_test_csp = csp.transform(X_test)

        def normalize(X):
            for i in range(X.shape[0]):
                X[i] = (X[i] - np.mean(X[i])) / (np.std(X[i]) + 1e-5)
            return X

        X_train_csp = normalize(X_train_csp)
        X_test_csp = normalize(X_test_csp)

        # 4. Reshape for PyTorch (Batch, 1, Channels, Time)
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

        # Learning Rate Scheduler: Reduces LR if accuracy stalls
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

        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test_t)
            _, predicted = torch.max(test_outputs.data, 1)
            acc = 100 * (predicted == y_test_t).sum().item() / y_test_t.size(0)
            fold_accs.append(acc)
            print(f"Fold {fold + 1} Accuracy: {acc:.2f}%")

    # 7. Final Results
    avg_acc = np.mean(fold_accs)
    print(f"\n=========================================")
    print(f"FINAL AVERAGE ACCURACY (CSP + EEGNet): {avg_acc:.2f}%")
    print(f"=========================================")


if __name__ == "__main__":
    train_csp_eegnet()

