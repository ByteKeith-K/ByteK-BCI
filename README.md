# 🧠 ByteK BCI: A Connected Web-Based Brain-Computer Interface Platform for Upper Limb Post-Stroke Rehabilitation in Resource-Constrained Settings

A low-cost, AI-powered Brain-Computer Interface (BCI) platform designed to bridge the neuro-rehabilitation gap in Zimbabwe and other resource-constrained settings.

## System Architecture
This repository contains the complete clinical platform (Monorepo):
*   `/BCI_Rehab_Platform`: The AI Engine and Backend (Python, EEGNet, CSP+SVM, PostgreSQL).
*   `/BCI_3D_Interface`: The Clinician & Doctor Portals, plus the WebGL 3D Therapy Game (Node.js, Three.js).

## How it Works
1. Clinician registers a patient and records mandatory pre-session vitals.
2. The system calibrates to the patient's brainwaves (Simulation Mode for now, Hardware Mode ready via LSL).
3. The patient plays a 3D game. The AI decodes brain intent (Left/Right) in real-time.
4. A strict 75% confidence threshold prevents the avatar from moving if the patient's signal is unclear.
5. Telemetry syncs to a remote Neurologist Dashboard for long-term recovery tracking.

## Tech Stack
*   AI/ML: PyTorch, MNE-Python, Scikit-Learn
*   Backend: Python, Node.js, Express, WebSockets
*   Database: PostgreSQL
*   Frontend: HTML/JS, Three.js (WebGL), Chart.js



## 📝 Development Methodology: R&D Local Sandbox

You may notice that the Git commit history for this project begins with a single, comprehensive "Initial Commit" rather than a long history of incremental commits. This was a deliberate and strategic engineering decision.

**1. The R&D Sandbox Phase:** 
The initial phase of this project involved rapid research and development (R&D) of complex AI models (EEGNet, CSP+SVM) and 3D WebGL rendering. To ensure rapid iteration without compromising security or cluttering the repository, development was conducted in a **secure, local sandbox environment**. 


**2. Production Baseline Release:** 
Once the system architecture was fully stabilized, the clinical safety gates were tested, and the database schema was finalized, the entire codebase was audited and committed as a clean **Production Baseline**. All subsequent feature additions, bug fixes, and optimizations (such as UI improvements and threshold tuning) will be tracked with standard, granular Git commits moving forward.