import torch
import numpy as np
import os
import joblib
from mne.decoding import CSP
from sklearn.metrics import classification_report, confusion_matrix

from preprocess import load_single_subject_for_calibration
from model import EEGNet


def evaluate_saved_model():
    print("--- LOADING SAVED PRODUCTION MODEL FOR VERIFICATION ---")

    # 1. Load Data
    X, y = load_single_subject_for_calibration(1)

    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")

    # 2. Load the saved CSP object
    csp_path = os.path.join(models_dir, "csp_transform.joblib")
    if not os.path.exists(csp_path):
        print("ERROR: CSP model not found! Did you run final_model.py first?")
        return
    csp = joblib.load(csp_path)

    # 3. Apply saved CSP and Normalize
    X_csp = csp.transform(X)
    for i in range(X_csp.shape[0]):
        X_csp[i] = (X_csp[i] - np.mean(X_csp[i])) / (np.std(X_csp[i]) + 1e-5)

    # 4. Load the saved EEGNet model
    device = torch.device("cpu")
    model = EEGNet(nb_classes=2, Chans=4, Samples=X_csp.shape[2],
                   F1=8, D=2, F2=16, dropoutRate=0.5).to(device)

    eegnet_path = os.path.join(models_dir, "eegnet_production.pth")
    if not os.path.exists(eegnet_path):
        print("ERROR: EEGNet model not found! Did you run final_model.py first?")
        return

    model.load_state_dict(torch.load(eegnet_path, weights_only=True))
    model.eval()

    # 5. Run Evaluation
    X_t = torch.tensor(np.expand_dims(X_csp, axis=1), dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.long)

    with torch.no_grad():
        outputs = model(X_t)
        _, predicted = torch.max(outputs.data, 1)

    # 6. Print Report
    accuracy = 100 * (predicted == y_t).sum().item() / y_t.size(0)
    print(f"\n=========================================")
    print(f"VERIFIED ACCURACY ON ALL SAVED DATA: {accuracy:.2f}%")
    print(f"=========================================\n")

    print("Confusion Matrix:")
    print(confusion_matrix(y, predicted.numpy()))
    print("\nClassification Report:")
    print(classification_report(y, predicted.numpy(), target_names=["Left Hand", "Right Hand"]))


if __name__ == "__main__":
    evaluate_saved_model()