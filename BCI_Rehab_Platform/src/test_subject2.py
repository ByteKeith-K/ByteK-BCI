import torch
import numpy as np
import os
import joblib
from mne.decoding import CSP
from sklearn.metrics import classification_report, confusion_matrix

# We load Subject 2's data
from preprocess import load_single_subject_for_calibration
from model import EEGNet


def test_unseen_subject():
    print("--- TESTING SUBJECT 1 MODEL ON SUBJECT 2 DATA ---")

    # 1. Load Subject 2's data
    X_test, y_test = load_single_subject_for_calibration(4)

    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")

    # 2. Load Subject 1's saved CSP
    csp = joblib.load(os.path.join(models_dir, "csp_transform.joblib"))

    # 3. Apply CSP and Normalize
    X_csp = csp.transform(X_test)
    for i in range(X_csp.shape[0]):
        X_csp[i] = (X_csp[i] - np.mean(X_csp[i])) / (np.std(X_csp[i]) + 1e-5)

    # 4. Load Subject 1's saved EEGNet
    device = torch.device("cpu")
    model = EEGNet(nb_classes=2, Chans=4, Samples=X_csp.shape[2],
                   F1=8, D=2, F2=16, dropoutRate=0.5).to(device)
    model.load_state_dict(torch.load(os.path.join(models_dir, "eegnet_production.pth"), weights_only=True))
    model.eval()

    # 5. Evaluate
    X_t = torch.tensor(np.expand_dims(X_csp, axis=1), dtype=torch.float32)
    y_t = torch.tensor(y_test, dtype=torch.long)

    with torch.no_grad():
        outputs = model(X_t)
        _, predicted = torch.max(outputs.data, 1)

    accuracy = 100 * (predicted == y_t).sum().item() / y_t.size(0)

    print(f"\n=========================================")
    print(f"CROSS-SUBJECT ACCURACY (Subject 1 -> Subject 2): {accuracy:.2f}%")
    print(f"=========================================\n")

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predicted.numpy()))
    print("\nClassification Report:")
    print(classification_report(y_test, predicted.numpy(), target_names=["Left Hand", "Right Hand"]))


if __name__ == "__main__":
    test_unseen_subject()