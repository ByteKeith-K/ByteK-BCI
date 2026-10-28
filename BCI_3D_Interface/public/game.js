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
const toast = document.getElementById('toast');

let isSessionActive = false;
let avatar = null;
let currentExercise = 'grip';

let leftArmBone = null, rightArmBone = null;
let leftForeArmBone = null, rightForeArmBone = null;
let leftHandBone = null, rightHandBone = null;

const canvas = document.getElementById('eeg-canvas');
const ctx = canvas.getContext('2d');
let eegOffset = 0; let eegAmplitude = 5; let eegColor = '#10b981';

function loadAvatar(gender) {
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
        if (!isSessionActive) resetUI();

    }, undefined, (error) => {
        loader.style.display = 'none';
        alert(`Failed to load ${gender}.glb.`);
    });
}

function setExercise(ex, btnElement) {
    playSound('success');
    currentExercise = ex;
    document.querySelectorAll('.ex-btn').forEach(btn => { btn.classList.remove('active'); btn.disabled = true; });
    btnElement.classList.add('active'); btnElement.disabled = false;
}

function resetUI() {
    try {
        if (leftArmBone) leftArmBone.rotation.set(1.55, 0, -0.20);
        if (rightArmBone) rightArmBone.rotation.set(1.55, 0, 0.20);
        if (leftForeArmBone) leftForeArmBone.rotation.set(0.1, 0, 0);
        if (rightForeArmBone) rightForeArmBone.rotation.set(0.1, 0, 0);
        if (leftHandBone) leftHandBone.rotation.set(0, 0, -1.8);
        if (rightHandBone) rightHandBone.rotation.set(0, 0, 1.8);
    } catch(e) {}

    statusText.innerText = "Waiting...";
    statusText.style.color = "#38bdf8";
    statusText.style.textShadow = "0 0 15px rgba(56, 189, 248, 0.4)";

    cueText.innerText = "GET READY...";
    cueText.style.color = "#fff";

    eegColor = '#10b981'; eegAmplitude = 5;
}

function showToast(message, type) {
    toast.innerText = message;
    toast.className = `toast-${type} toast-show`;
    setTimeout(() => { toast.className = ''; }, 2500);
}

const ws = new WebSocket('ws://localhost:3000');

ws.onmessage = (event) => {
    try {
        const data = JSON.parse(event.data);
        if (data.source === 'browser') return;

        if (data.action === 'session_start') {
            isSessionActive = true;
            return;
        }

        if (data.action === 'session_end') {
            isSessionActive = false;
            resetUI();
            cueText.innerText = "SESSION ENDED";
            cueText.style.color = "#10b981";

            // Use the exact session ID provided by Python
            if (data.session_id) {
                window.currentSessionId = data.session_id;
            }

            endModal.style.display = 'flex';
            return;
        }

        if (!avatar) return;

        if (data.action === 'rest') {
            resetUI();
            cueText.innerText = "REST";
            cueText.style.color = "#64748b";
            statusText.innerText = "Resting...";
            statusText.style.color = "#64748b";
            uiConf.innerText = "--%";
            uiLat.innerText = "-- ms";
            return;
        }

        if (data.action === 'cue') {
            playSound('cue');
            let exerciseName = "GRIP";
            if (currentExercise === 'lift') exerciseName = "SHOULDER LIFT";
            if (currentExercise === 'reach') exerciseName = "FUNCTIONAL REACH";

            cueText.innerText = `IMAGINE ${data.intended_action.toUpperCase()} ${exerciseName}`;
            cueText.style.color = "#38bdf8";
            statusText.innerText = "Reading Brainwaves...";
            statusText.style.color = "#38bdf8";
            eegColor = '#38bdf8'; eegAmplitude = 20;
            return;
        }

        if (data.action === 'result') {
            if (data.confidence !== undefined) uiConf.innerText = `${data.confidence.toFixed(1)}%`;
            if (data.latency !== undefined) uiLat.innerText = `${data.latency.toFixed(0)} ms`;

            if (data.status === "Rejected") {
                playSound('reject');
                statusText.innerText = `REJECTED (Low Conf)`;
                statusText.style.color = "#f59e0b";
                statusText.style.textShadow = "0 0 15px rgba(245, 158, 11, 0.4)";
                showToast("Focus and try again", "warning");
                eegColor = '#f59e0b'; eegAmplitude = 15;
                return;
            }

            if (data.prediction === data.intended_action) {
                playSound('success');
                resetUI();
                statusText.innerText = `SUCCESS!`;
                statusText.style.color = "#10b981";
                statusText.style.textShadow = "0 0 15px rgba(16, 185, 129, 0.4)";
                showToast("✓ Correct Intent", "success");
                eegColor = '#10b981'; eegAmplitude = 40;

                if (data.prediction === 'left') animateLeftArm();
                if (data.prediction === 'right') animateRightArm();

            } else {
                playSound('reject');
                resetUI();
                statusText.innerText = `MISMATCH`;
                statusText.style.color = "#ef4444";
                statusText.style.textShadow = "0 0 15px rgba(239, 68, 68, 0.4)";
                showToast(`✗ AI read ${data.prediction.toUpperCase()}`, "error");
                eegColor = '#ef4444'; eegAmplitude = 20;
            }
        }

    } catch (e) { console.error("WS Parsing Error:", e); }
};

function animateLeftArm() {
    if (!leftArmBone) return;
    if (currentExercise === 'grip') { leftArmBone.rotation.set(0.8, 0, 0.2); if (leftForeArmBone) leftForeArmBone.rotation.set(0, -1.5, 0); }
    else if (currentExercise === 'lift') { leftArmBone.rotation.set(-1.0, 0, 0.5); if (leftForeArmBone) leftForeArmBone.rotation.set(0, 0, 0); }
    else if (currentExercise === 'reach') { leftArmBone.rotation.set(0, 0, 1.2); if (leftForeArmBone) leftForeArmBone.rotation.set(0, 0, 0); }
}

function animateRightArm() {
    if (!rightArmBone) return;
    if (currentExercise === 'grip') { rightArmBone.rotation.set(0.8, 0, -0.2); if (rightForeArmBone) rightForeArmBone.rotation.set(0, 1.5, 0); }
    else if (currentExercise === 'lift') { rightArmBone.rotation.set(-1.0, 0, -0.5); if (rightForeArmBone) rightForeArmBone.rotation.set(0, 0, 0); }
    else if (currentExercise === 'reach') { rightArmBone.rotation.set(0, 0, -1.2); if (rightForeArmBone) rightForeArmBone.rotation.set(0, 0, 0); }
}

function resizeCanvas() { canvas.width = canvas.parentElement.clientWidth; canvas.height = canvas.parentElement.clientHeight; }
window.addEventListener('resize', resizeCanvas);

function drawEEG() {
    requestAnimationFrame(drawEEG);
    ctx.fillStyle = 'rgba(11, 17, 32, 0.2)';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.beginPath(); ctx.lineWidth = 2; ctx.strokeStyle = eegColor; ctx.shadowBlur = 10; ctx.shadowColor = eegColor;
    for (let x = 0; x < canvas.width; x++) {
        let y = canvas.height / 2;
        let t = (x + eegOffset) * 0.05;
        let wave = Math.sin(t) * Math.cos(t * 0.5) * eegAmplitude + (Math.random() - 0.5) * eegAmplitude * 0.5;
        ctx.lineTo(x, y + wave);
    }
    ctx.stroke(); ctx.shadowBlur = 0;
    eegOffset -= 5;
    if (eegAmplitude > 5) eegAmplitude -= 0.5;
    if (eegAmplitude < 5) { eegAmplitude += 0.5; eegColor = '#10b981'; }
}

function animate() { requestAnimationFrame(animate); renderer.render(scene, camera); }
animate(); drawEEG();
