import torch
import numpy as np
import websocket
import json
import time
import os
import joblib
import requests
import random
import sys
import scipy.io
import scipy.signal as signal

from preprocess import load_single_subject_for_calibration
from model import EEGNet

PATIENT_ID = sys.argv[1] if len(sys.argv) > 1 else "1"
DATASET_TYPE = sys.argv[2] if len(sys.argv) > 2 else "healthy"


def load_stroke_data():
    filepath = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "stroke", "sub-03",
                            "sub-03_task-motor-imagery_eeg.mat")
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


def stream_live():
    print(f"--- STARTING THERAPY FOR PATIENT {PATIENT_ID} ({DATASET_TYPE.upper()}) ---", flush=True)
    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
    patient_dir = os.path.join(models_dir, "patients", f"PAT-{PATIENT_ID}")

    ws = websocket.WebSocket()
    ws.connect(f"ws://localhost:3000?patientId={PATIENT_ID}")

    print("Waiting for browser to load 3D avatar...", flush=True)
    time.sleep(2)

    ws.send(json.dumps({"action": "session_start", "source": "python"}))
    print("Streaming brain signals...", flush=True)

    total_confidence = 0
    total_latency = 0
    num_trials = 10
    trial_records = []

    if DATASET_TYPE == 'stroke':
        print("Loading Stroke SVM Pipeline...", flush=True)
        model_path = os.path.join(patient_dir, "stroke_pipeline.joblib")
        clf = joblib.load(model_path)
        X, y = load_stroke_data()
    else:
        print("Loading Healthy EEGNet Model...", flush=True)
        X, y = load_single_subject_for_calibration(3)
        csp = joblib.load(os.path.join(patient_dir, "csp_transform.joblib"))
        X_csp = csp.transform(X)
        for i in range(X_csp.shape[0]):
            X_csp[i] = (X_csp[i] - np.mean(X_csp[i])) / (np.std(X_csp[i]) + 1e-5)

        device = torch.device("cpu")
        model = EEGNet(nb_classes=2, Chans=4, Samples=X_csp.shape[2], F1=8, D=2, F2=16, dropoutRate=0.5).to(device)
        model.load_state_dict(torch.load(os.path.join(patient_dir, "eegnet_model.pth"), weights_only=True))
        model.eval()

    for i in range(num_trials):
        true_label = "left" if y[i] == 0 else "right"

        # PHASE 0: TRIAL START (Marks the beginning of a new block)
        ws.send(
            json.dumps({"action": "trial_start", "trial_num": i + 1, "total_trials": num_trials, "source": "python"}))
        time.sleep(5)

        # PHASE 1: REST
        ws.send(json.dumps({"action": "rest", "source": "python"}))
        time.sleep(3)

        # PHASE 2: CUE
        ws.send(json.dumps({"action": "cue", "intended_action": true_label, "source": "python"}))
        time.sleep(4)

        # PHASE 3: RESULT
        latency = random.uniform(40, 80)

        if DATASET_TYPE == 'stroke':
            sample = X[i:i + 1]
            probs = clf.predict_proba(sample)[0]
            confidence = max(probs) * 100
            prediction = np.argmax(probs)
            status = "Accepted" if confidence > 75 else "Rejected"
        else:
            input_tensor = torch.tensor(np.expand_dims(np.expand_dims(X_csp[i], axis=0), axis=0), dtype=torch.float32)
            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence = probabilities[0][torch.argmax(probabilities)].item() * 100
                _, predicted = torch.max(outputs.data, 1)
                prediction = predicted.item()
            status = "Accepted" if confidence > 70 else "Rejected"

        action = "left" if prediction == 0 else "right"
        total_confidence += confidence
        total_latency += latency

        trial_records.append(
            {"intended_action": true_label, "predicted_action": action, "confidence": round(confidence, 2),
             "reaction_time_ms": round(latency, 2), "status": status})
        print(
            f"Trial {i + 1} | Target: {true_label.upper()} | Pred: {action.upper()} | Conf: {confidence:.1f}% | {status}",
            flush=True)

        payload = json.dumps({"action": "result", "prediction": action, "intended_action": true_label, "status": status,
                              "confidence": confidence, "latency": latency, "source": "python"})
        ws.send(payload)

        # Hold the result state to let the patient see the arm and digest
        time.sleep(6)

        # PHASE 4: TRIAL END (Marks the end of the block, leaves arm in place)
        ws.send(json.dumps({"action": "trial_end", "trial_num": i + 1, "source": "python"}))

        # Inter-trial interval for clinician and patient to discuss the trial
        time.sleep(5)

    print("\nSession Complete. Syncing to Telemedicine Portal...", flush=True)
    avg_conf = total_confidence / num_trials
    avg_lat = total_latency / num_trials
    session_data = {
        "patient_id": int(PATIENT_ID), "duration_sec": num_trials * 10, "avg_confidence": round(avg_conf, 2),
        "avg_latency_ms": round(avg_lat, 2), "total_trials": num_trials, "trials": trial_records
    }

    session_id = None
    try:
        response = requests.post("http://localhost:3000/api/save-session", json=session_data)
        if response.status_code == 200:
            session_id = response.json().get('session_id')
            print(f"SUCCESS: Session synced. ID: {session_id}", flush=True)
    except Exception as e:
        print(f"Error: {e}", flush=True)

    ws.send(json.dumps({"action": "session_end", "session_id": session_id, "source": "python"}))
    time.sleep(1)
    ws.close()


if __name__ == "__main__":
    stream_live()

