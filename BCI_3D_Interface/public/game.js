function playSound(type) {
    try {
        const AudioContext = window.AudioContext || window.webkitAudioContext;
        if (!AudioContext) return;
        const ctx = new AudioContext();
        const osc = ctx.createOscillator(); const gain = ctx.createGain();
        osc.connect(gain); gain.connect(ctx.destination);

        if (type === 'success') {
            osc.type = 'sine'; osc.frequency.setValueAtTime(600, ctx.currentTime);
            osc.frequency.exponentialRampToValueAtTime(1200, ctx.currentTime + 0.1);
            gain.gain.setValueAtTime(0.1, ctx.currentTime); gain.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + 0.1);
            osc.start(); osc.stop(ctx.currentTime + 0.1);
        } else if (type === 'reject') {
            osc.type = 'triangle'; osc.frequency.setValueAtTime(300, ctx.currentTime);
            osc.frequency.exponentialRampToValueAtTime(150, ctx.currentTime + 0.15);
            gain.gain.setValueAtTime(0.1, ctx.currentTime); gain.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + 0.15);
            osc.start(); osc.stop(ctx.currentTime + 0.15);
        } else if (type === 'cue') {
            osc.type = 'sine'; osc.frequency.setValueAtTime(800, ctx.currentTime);
            gain.gain.setValueAtTime(0.05, ctx.currentTime); gain.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + 0.2);
            osc.start(); osc.stop(ctx.currentTime + 0.2);
        }
    } catch (e) {}
}

// --- VOICE SYNTHESIS ENGINE ---
let currentAvatarGender = 'male';
let availableVoices = [];

if (window.speechSynthesis) {
    window.speechSynthesis.onvoiceschanged = () => {
        availableVoices = window.speechSynthesis.getVoices();
    };
}

function speak(text) {
    if (!window.speechSynthesis) return;
    window.speechSynthesis.cancel();

    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = 1.0;
    utterance.volume = 1.0;

    const enVoices = availableVoices.filter(v => v.lang.startsWith('en'));
    let selectedVoice = null;

    if (enVoices.length > 0) {
        const genderVoices = enVoices.filter(v => v.name.toLowerCase().includes(currentAvatarGender));
        if (genderVoices.length > 0) {
            selectedVoice = genderVoices[0];
        } else {
            selectedVoice = enVoices[0];
            utterance.pitch = currentAvatarGender === 'female' ? 1.3 : 0.7;
        }
    }

    if (selectedVoice) utterance.voice = selectedVoice;
    window.speechSynthesis.speak(utterance);
}

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0b1120);
const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 1000);
camera.position.set(0, 1.3, 2.2); camera.lookAt(0, 1.3, 0);
const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
renderer.setPixelRatio(window.devicePixelRatio); renderer.setSize(window.innerWidth, window.innerHeight);
renderer.shadowMap.enabled = true; document.body.appendChild(renderer.domElement);

scene.add(new THREE.AmbientLight(0xffffff, 0.7));
const dirLight = new THREE.DirectionalLight(0xffffff, 1.2);
dirLight.position.set(2, 5, 5); dirLight.castShadow = true; scene.add(dirLight);
scene.add(new THREE.GridHelper(20, 20, 0x38bdf8, 0x1e293b));

const statusText = document.getElementById('status');
const endModal = document.getElementById('end-modal');
const loader = document.getElementById('loader');
const uiConf = document.getElementById('ui-conf');
const uiLat = document.getElementById('ui-lat');
const cueText = document.getElementById('cue-text');

let isSessionActive = false;
let avatar = null;
let currentExercise = 'grip';

// Bone references
let leftArmBone = null, rightArmBone = null;
let leftForeArmBone = null, rightForeArmBone = null;
let leftHandBone = null, rightHandBone = null;

// Animation State Machine: 'idle', 'forward', 'holding', 'returning'
let animState = 'idle';
let animProgress = 0;
let currentTargetArm = 'left';
const animSpeed = 0.015;
const holdDurationFrames = 72;
let holdFramesCounter = 0;

// Exact Resting Pose
const restVals = {
    lArm: {x: 1.55, y: 0, z: -0.20},
    rArm: {x: 1.55, y: 0, z: 0.20},
    lFore: {x: 0.1, y: 0, z: 0},
    rFore: {x: 0.1, y: 0, z: 0},
    lHand: {x: 0, y: 0, z: -1.8},
    rHand: {x: 0, y: 0, z: 1.8}
};

const targetVals = {
    grip: {
        lArm: {x: 0.9, y: 0, z: -0.1}, lFore: {x: 0, y: 0, z: 1.8},
        rArm: {x: 0.9, y: 0, z: 0.1},  rFore: {x: 0, y: 0, z: -1.8}
    },
    lift: {
        lArm: {x: -1.0, y: 0, z: 0.5}, lFore: {x: 0, y: 0, z: 0},
        rArm: {x: -1.0, y: 0, z: -0.5}, rFore: {x: 0, y: 0, z: 0}
    },
    reach: {
        lArm: {x: 0.1, y: -0.2, z: 1.45}, lFore: {x: 0.0, y: 0.0, z: 0},
        rArm: {x: 0.1, y: 0.2, z: -1.45}, rFore: {x: 0.0, y: 0.0, z: 0}
    }
};

const canvas = document.getElementById('eeg-canvas');
const ctx = canvas.getContext('2d');
let eegOffset = 0; let eegAmplitude = 5; let eegColor = '#10b981';
let currentRealEEGData = []; // NEW: Holds actual data from Python

function loadAvatar(gender) {
    currentAvatarGender = gender;
    loader.style.display = 'block';
    const gltfLoader = new THREE.GLTFLoader();
    let modelURL = gender === 'female' ? '/female.glb' : '/male.glb';

    gltfLoader.load(modelURL, (gltf) => {
        avatar = gltf.scene; avatar.position.y = 0;

        avatar.traverse((node) => {
            if (node.isMesh) { node.castShadow = true; node.receiveShadow = true; }
            if (node.isBone) {
                let name = node.name.toLowerCase();
                if (name.includes('leftarm') && !name.includes('forearm')) leftArmBone = node;
                if (name.includes('rightarm') && !name.includes('forearm')) rightArmBone = node;
                if (name.includes('leftforearm')) leftForeArmBone = node;
                if (name.includes('rightforearm')) rightForeArmBone = node;
                if (name.includes('lefthand')) leftHandBone = node;
                if (name.includes('righthand')) rightHandBone = node;
            }
        });

        scene.add(avatar); loader.style.display = 'none';

        document.getElementById('header-bar').style.display = 'flex';
        document.getElementById('exercise-panel').style.display = 'block';
        document.getElementById('status-container').style.display = 'block';
        document.getElementById('eeg-container').style.display = 'flex';
        document.getElementById('cue-banner').style.display = 'block';

        resizeCanvas();
        resetUI();

    }, undefined, (error) => {
        loader.style.display = 'none';
        alert(`Failed to load ${gender}.glb.`);
    });
}

function setExercise(ex, btnElement) {
    playSound('success');
    currentExercise = ex;

    document.querySelectorAll('.ex-btn').forEach(btn => {
        btn.classList.remove('active');
        btn.disabled = false;
    });
    btnElement.classList.add('active');
}

function resetUI() {
    if (animProgress > 0.01) {
        animState = 'returning';
    } else {
        animState = 'idle';
        animProgress = 0;
        try {
            if (leftArmBone) leftArmBone.rotation.set(restVals.lArm.x, restVals.lArm.y, restVals.lArm.z);
            if (rightArmBone) rightArmBone.rotation.set(restVals.rArm.x, restVals.rArm.y, restVals.rArm.z);
            if (leftForeArmBone) leftForeArmBone.rotation.set(restVals.lFore.x, restVals.lFore.y, restVals.lFore.z);
            if (rightForeArmBone) rightForeArmBone.rotation.set(restVals.rFore.x, restVals.rFore.y, restVals.rFore.z);
            if (leftHandBone) leftHandBone.rotation.set(restVals.lHand.x, restVals.lHand.y, restVals.lHand.z);
            if (rightHandBone) rightHandBone.rotation.set(restVals.rHand.x, restVals.rHand.y, restVals.rHand.z);
        } catch(e) {}
    }

    statusText.innerText = "Waiting...";
    statusText.style.color = "#38bdf8";
    statusText.style.textShadow = "0 0 15px rgba(56, 189, 248, 0.4)";

    cueText.innerText = "GET READY...";
    cueText.style.color = "#fff";

    eegColor = '#10b981'; eegAmplitude = 5;
}

const ws = new WebSocket('ws://localhost:3000');

ws.onmessage = (event) => {
    try {
        const data = JSON.parse(event.data);
        if (data.source === 'browser') return;

        if (data.action === 'session_start') {
            isSessionActive = true;
            speak("Session started. Let us begin.");
            return;
        }

        if (data.action === 'session_end') {
            isSessionActive = false;
            resetUI();
            currentRealEEGData = []; // Clear real data
            cueText.innerText = "SESSION ENDED";
            cueText.style.color = "#10b981";
            speak("Session complete. Well done.");

            if (data.session_id) {
                window.currentSessionId = data.session_id;
            }

            endModal.style.display = 'flex';
            return;
        }

        if (!avatar) return;

        if (data.action === 'trial_start') {
            resetUI();
            currentRealEEGData = []; // Clear real data

            cueText.innerText = `TRIAL ${data.trial_num} OF ${data.total_trials}`;
            cueText.style.color = "#f59e0b";
            statusText.innerText = "Preparing...";
            statusText.style.color = "#f59e0b";

            uiConf.innerText = "--%";
            uiLat.innerText = "-- ms";

            speak(`Trial ${data.trial_num}. Get ready.`);
            return;
        }

        if (data.action === 'trial_end') {
            cueText.innerText = `TRIAL ${data.trial_num} COMPLETE`;
            cueText.style.color = "#64748b";
            statusText.innerText = "Trial Complete";
            statusText.style.color = "#64748b";
            return;
        }

        if (data.action === 'rest') {
            resetUI();
            currentRealEEGData = []; // Clear real data
            cueText.innerText = "REST";
            cueText.style.color = "#64748b";
            statusText.innerText = "Resting...";
            statusText.style.color = "#64748b";
            uiConf.innerText = "--%";
            uiLat.innerText = "-- ms";
            speak("Rest.");
            return;
        }

        if (data.action === 'cue') {
            playSound('cue');
            currentRealEEGData = []; // Clear real data
            let exerciseName = "grip";
            if (currentExercise === 'lift') exerciseName = "shoulder lift";
            if (currentExercise === 'reach') exerciseName = "functional reach";

            let instruction = `Imagine ${data.intended_action} ${exerciseName}`;

            cueText.innerText = instruction.toUpperCase();
            cueText.style.color = "#38bdf8";
            statusText.innerText = "Reading Brainwaves...";
            statusText.style.color = "#38bdf8";
            eegColor = '#38bdf8'; eegAmplitude = 20;

            speak(instruction);
            return;
        }

        if (data.action === 'result') {
            if (data.confidence !== undefined) uiConf.innerText = `${data.confidence.toFixed(1)}%`;
            if (data.latency !== undefined) uiLat.innerText = `${data.latency.toFixed(0)} ms`;

            // NEW: Capture the real EEG data payload
            if (data.eeg_data) {
                currentRealEEGData = data.eeg_data;
            }

            if (data.status === "Rejected") {
                playSound('reject');
                statusText.innerText = `REJECTED (Low Conf)`;
                statusText.style.color = "#f59e0b";
                statusText.style.textShadow = "0 0 15px rgba(245, 158, 11, 0.4)";
                showToast("Focus and try again", "warning");
                eegColor = '#f59e0b'; eegAmplitude = 15;
                speak("Signal unclear. Please focus.");
                return;
            }

            if (data.prediction === data.intended_action) {
                playSound('success');
                statusText.innerText = `SUCCESS!`;
                statusText.style.color = "#10b981";
                statusText.style.textShadow = "0 0 15px rgba(16, 185, 129, 0.4)";
                showToast("✓ Correct Intent", "success");
                eegColor = '#10b981'; eegAmplitude = 40;

                speak("Excellent. Correct.");

                animState = 'forward';
                animProgress = 0;
                holdFramesCounter = 0;
                currentTargetArm = data.prediction;

            } else {
                playSound('reject');
                resetUI();
                statusText.innerText = `MISMATCH`;
                statusText.style.color = "#ef4444";
                statusText.style.textShadow = "0 0 15px rgba(239, 68, 68, 0.4)";
                showToast(`✗ AI read ${data.prediction.toUpperCase()}`, "error");
                eegColor = '#ef4444'; eegAmplitude = 20;
                speak("Incorrect. Try to focus.");
            }
        }

    } catch (e) { console.error("WS Parsing Error:", e); }
};

// --- SMOOTH ANIMATION ENGINE ---
function animateArm() {
    if (animState === 'idle') return;

    if (animState === 'forward') {
        animProgress += animSpeed;
        if (animProgress >= 1) {
            animProgress = 1;
            animState = 'holding';
            holdFramesCounter = 0;
        }
    } else if (animState === 'holding') {
        holdFramesCounter++;
        if (holdFramesCounter >= holdDurationFrames) {
            animState = 'returning';
        }
    } else if (animState === 'returning') {
        animProgress -= animSpeed;
        if (animProgress <= 0) { animProgress = 0; animState = 'idle'; }
    }

    let targetL = restVals.lArm, targetR = restVals.rArm;
    let targetFL = restVals.lFore, targetFR = restVals.rFore;

    if (animState === 'forward' || animState === 'holding' || (animState === 'returning' && animProgress > 0)) {
        const activeTarget = targetVals[currentExercise];
        if (currentTargetArm === 'left') {
            targetL = activeTarget.lArm;
            targetFL = activeTarget.lFore;
        } else {
            targetR = activeTarget.rArm;
            targetFR = activeTarget.rFore;
        }
    }

    try {
        if (leftArmBone) {
            leftArmBone.rotation.x = THREE.MathUtils.lerp(restVals.lArm.x, targetL.x, animProgress);
            leftArmBone.rotation.y = THREE.MathUtils.lerp(restVals.lArm.y, targetL.y, animProgress);
            leftArmBone.rotation.z = THREE.MathUtils.lerp(restVals.lArm.z, targetL.z, animProgress);
        }
        if (rightArmBone) {
            rightArmBone.rotation.x = THREE.MathUtils.lerp(restVals.rArm.x, targetR.x, animProgress);
            rightArmBone.rotation.y = THREE.MathUtils.lerp(restVals.rArm.y, targetR.y, animProgress);
            rightArmBone.rotation.z = THREE.MathUtils.lerp(restVals.rArm.z, targetR.z, animProgress);
        }
        if (leftForeArmBone) {
            leftForeArmBone.rotation.x = THREE.MathUtils.lerp(restVals.lFore.x, targetFL.x, animProgress);
            leftForeArmBone.rotation.y = THREE.MathUtils.lerp(restVals.lFore.y, targetFL.y, animProgress);
            leftForeArmBone.rotation.z = THREE.MathUtils.lerp(restVals.lFore.z, targetFL.z, animProgress);
        }
        if (rightForeArmBone) {
            rightForeArmBone.rotation.x = THREE.MathUtils.lerp(restVals.rFore.x, targetFR.x, animProgress);
            rightForeArmBone.rotation.y = THREE.MathUtils.lerp(restVals.rFore.y, targetFR.y, animProgress);
            rightForeArmBone.rotation.z = THREE.MathUtils.lerp(restVals.rFore.z, targetFR.z, animProgress);
        }

        if (leftHandBone) leftHandBone.rotation.set(restVals.lHand.x, restVals.lHand.y, restVals.lHand.z);
        if (rightHandBone) rightHandBone.rotation.set(restVals.rHand.x, restVals.rHand.y, restVals.rHand.z);

    } catch(e) {}
}

function resizeCanvas() { canvas.width = canvas.parentElement.clientWidth; canvas.height = canvas.parentElement.clientHeight; }
window.addEventListener('resize', resizeCanvas);

function drawEEG() {
    requestAnimationFrame(drawEEG);

    // Dim the background to create the trailing effect
    ctx.fillStyle = 'rgba(11, 17, 32, 0.2)';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    ctx.beginPath();
    ctx.lineWidth = 2;
    ctx.strokeStyle = eegColor;
    ctx.shadowBlur = 10;
    ctx.shadowColor = eegColor;

    if (currentRealEEGData.length > 0) {
        // --- DRAW THE REAL AI DATA ---
        const dataLength = currentRealEEGData.length;
        let minVal = Math.min(...currentRealEEGData);
        let maxVal = Math.max(...currentRealEEGData);
        let range = maxVal - minVal;
        if (range === 0) range = 1;

        for (let x = 0; x < canvas.width; x++) {
            // Map canvas X to the real data array index
            let dataIndex = Math.floor((x / canvas.width) * dataLength);
            let val = currentRealEEGData[dataIndex];

            // Scale the value to fit the canvas height with padding
            let y = canvas.height / 2 - ((val - minVal) / range - 0.5) * (canvas.height * 0.8);

            if (x === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        }
    } else {
        // --- DRAW BASELINE NOISE (When waiting, rest, or cue) ---
        for (let x = 0; x < canvas.width; x++) {
            let y = canvas.height / 2;
            let t = (x + eegOffset) * 0.05;
            let wave = Math.sin(t) * Math.cos(t * 0.5) * eegAmplitude + (Math.random() - 0.5) * eegAmplitude * 0.5;
            ctx.lineTo(x, y + wave);
        }
        eegOffset -= 5;
        if (eegAmplitude > 5) eegAmplitude -= 0.5;
        if (eegAmplitude < 5) { eegAmplitude += 0.5; eegColor = '#10b981'; }
    }

    ctx.stroke();
    ctx.shadowBlur = 0;
}

function animate() {
    requestAnimationFrame(animate);
    animateArm();
    renderer.render(scene, camera);
}
animate();
drawEEG();