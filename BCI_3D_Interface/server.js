require('dotenv').config(); // Load environment variables from .env

const express = require('express');
const http = require('http');
const WebSocket = require('ws');
const path = require('path');
const { Pool } = require('pg');
const { spawn } = require('child_process');

const app = express();
app.use(express.json());

// --- SECURE DATABASE CONFIGURATION ---
const pool = new Pool({
    user: process.env.DB_USER,
    host: process.env.DB_HOST,
    database: process.env.DB_NAME,
    password: process.env.DB_PASSWORD,
    port: parseInt(process.env.DB_PORT, 10),
});

// --- SECURE PYTHON PATHS ---
const pythonPath = process.env.PYTHON_EXECUTABLE;
const scriptDir = process.env.PYTHON_SCRIPT_DIR;

const server = http.createServer(app);
const wss = new WebSocket.Server({ server });

app.use(express.static(path.join(__dirname, 'public')));
app.get('/', (req, res) => res.sendFile(path.join(__dirname, 'public', 'login.html')));

// --- AUTH ROUTE ---
app.post('/api/login', async (req, res) => {
    const { username, password } = req.body;
    try {
        const result = await pool.query('SELECT * FROM users WHERE username = $1 AND password = $2', [username, password]);
        if (result.rows.length > 0) res.status(200).json({ success: true, user: result.rows[0] });
        else res.status(401).json({ success: false, message: "Invalid credentials" });
    } catch (err) { res.status(500).json({ error: err.message }); }
});

// --- SETTINGS ROUTES ---
app.get('/api/get-settings', async (req, res) => {
    try {
        const result = await pool.query('SELECT * FROM system_settings');
        const settings = {};
        result.rows.forEach(row => settings[row.key] = parseFloat(row.value));
        res.status(200).json(settings);
    } catch (err) { res.status(500).json({ error: err.message }); }
});

app.post('/api/save-settings', async (req, res) => {
    const settings = req.body;
    const client = await pool.connect();
    try {
        await client.query('BEGIN');
        for (const [key, value] of Object.entries(settings)) {
            await client.query('UPDATE system_settings SET value = $1 WHERE key = $2', [value.toString(), key]);
        }
        await client.query('COMMIT');
        res.status(200).json({ success: true });
    } catch (err) {
        await client.query('ROLLBACK'); res.status(500).json({ error: err.message });
    } finally { client.release(); }
});

// --- PATIENT ROUTES ---
app.get('/api/get-patients', async (req, res) => {
    try {
        const result = await pool.query('SELECT * FROM patients ORDER BY id ASC');
        res.status(200).json(result.rows);
    } catch (err) { res.status(500).json({ error: err.message }); }
});

app.post('/api/add-patient', async (req, res) => {
    const { name, age, gender, affected_side, dataset_type } = req.body;
    try {
        const result = await pool.query(
            'INSERT INTO patients (patient_code, name, age, gender, affected_side, dataset_type, has_model) VALUES ($1, $2, $3, $4, $5, $6, false) RETURNING *',
            [`PAT-${Date.now()}`, name, age, gender, affected_side, dataset_type]
        );
        res.status(200).json({ success: true, patient: result.rows[0] });
    } catch (err) { res.status(500).json({ error: err.message }); }
});

app.put('/api/edit-patient/:id', async (req, res) => {
    const { name, age, affected_side, notes } = req.body;
    try {
        await pool.query('UPDATE patients SET name = $1, age = $2, affected_side = $3, notes = $4 WHERE id = $5', [name, age, affected_side, notes, req.params.id]);
        res.status(200).json({ success: true });
    } catch (err) { res.status(500).json({ error: err.message }); }
});

app.delete('/api/delete-patient/:id', async (req, res) => {
    const patientId = req.params.id;
    try {
        const sessions = await pool.query('SELECT id FROM sessions WHERE patient_id = $1', [patientId]);
        if (sessions.rows.length > 0) return res.status(400).json({ success: false, message: "Cannot delete: Patient has recorded therapy sessions." });
        await pool.query('DELETE FROM patients WHERE id = $1', [patientId]);
        res.status(200).json({ success: true });
    } catch (err) { res.status(500).json({ error: err.message }); }
});

// --- VITALS ROUTES ---
app.post('/api/save-vitals', async (req, res) => {
    const { patient_id, session_id, phase, bp_systolic, bp_diastolic, heart_rate, spo2, fatigue_scale, symptoms_reported } = req.body;
    try {
        await pool.query(
            'INSERT INTO session_vitals (patient_id, session_id, phase, bp_systolic, bp_diastolic, heart_rate, spo2, fatigue_scale, symptoms_reported) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)',
            [patient_id, session_id, phase, bp_systolic, bp_diastolic, heart_rate, spo2, fatigue_scale, symptoms_reported]
        );
        res.status(200).json({ success: true });
    } catch (err) { res.status(500).json({ error: err.message }); }
});

app.get('/api/get-vitals/:session_id', async (req, res) => {
    try {
        const result = await pool.query('SELECT * FROM session_vitals WHERE session_id = $1 ORDER BY recorded_at ASC', [req.params.session_id]);
        res.status(200).json(result.rows);
    } catch (err) { res.status(500).json({ error: err.message }); }
});

app.get('/api/get-patient-vitals/:patient_id', async (req, res) => {
    try {
        const result = await pool.query('SELECT * FROM session_vitals WHERE patient_id = $1 ORDER BY recorded_at ASC', [req.params.patient_id]);
        res.status(200).json(result.rows);
    } catch (err) { res.status(500).json({ error: err.message }); }
});

app.get('/api/get-latest-session/:patient_id', async (req, res) => {
    try {
        const result = await pool.query('SELECT id FROM sessions WHERE patient_id = $1 ORDER BY session_date DESC LIMIT 1', [req.params.patient_id]);
        res.status(200).json(result.rows[0] || {});
    } catch (err) { res.status(500).json({ error: err.message }); }
});

app.get('/api/doctor-dashboard', async (req, res) => {
    try {
        const patients = await pool.query('SELECT * FROM patients ORDER BY id ASC');
        const sessions = await pool.query('SELECT * FROM sessions ORDER BY session_date DESC');
        const vitals = await pool.query('SELECT * FROM session_vitals');
        res.status(200).json({ patients: patients.rows, sessions: sessions.rows, vitals: vitals.rows });
    } catch (err) { res.status(500).json({ error: err.message }); }
});

// --- AI ROUTES ---
app.post('/api/calibrate/:id', async (req, res) => {
    const patientId = req.params.id;
    try {
        const pResult = await pool.query('SELECT dataset_type FROM patients WHERE id = $1', [patientId]);
        const datasetType = pResult.rows[0]?.dataset_type || 'healthy';

        const scriptPath = path.join(scriptDir, 'calibrate_patient.py');

        const pythonProcess = spawn(pythonPath, ['-u', scriptPath, patientId, datasetType]);
        pythonProcess.stdout.on('data', (data) => console.log(`Python: ${data}`));
        pythonProcess.stderr.on('data', (data) => console.error(`Python Error: ${data}`));

        pythonProcess.on('close', async (code) => {
            if (code === 0) {
                await pool.query('UPDATE patients SET has_model = true, calibrated_at = CURRENT_TIMESTAMP WHERE id = $1', [patientId]);
                console.log(`Calibration successful for Patient ${patientId}`);
            }
        });
        res.status(200).json({ success: true });
    } catch (err) { res.status(500).json({ error: err.message }); }
});

app.post('/api/start-therapy/:id', async (req, res) => {
    const patientId = req.params.id;
    try {
        const pResult = await pool.query('SELECT dataset_type FROM patients WHERE id = $1', [patientId]);
        const datasetType = pResult.rows[0]?.dataset_type || 'healthy';

        const scriptPath = path.join(scriptDir, 'websocket_stream.py');

        const pythonProcess = spawn(pythonPath, ['-u', scriptPath, patientId, datasetType]);
        pythonProcess.stdout.on('data', (data) => console.log(`Python: ${data}`));
        pythonProcess.stderr.on('data', (data) => console.error(`Python Error: ${data}`));

        res.status(200).json({ success: true });
    } catch (err) { res.status(500).json({ error: err.message }); }
});

app.post('/api/save-session', async (req, res) => {
    const { patient_id, duration_sec, avg_confidence, avg_latency_ms, total_trials, trials } = req.body;
    const client = await pool.connect();
    try {
        await client.query('BEGIN');
        const sessionResult = await client.query(
            'INSERT INTO sessions (patient_id, duration_sec, avg_confidence, avg_latency_ms, total_trials) VALUES ($1, $2, $3, $4, $5) RETURNING id;',
            [patient_id, duration_sec, avg_confidence, avg_latency_ms, total_trials]
        );
        const session_id = sessionResult.rows[0].id;
        for (let t of trials) {
            await client.query(
                'INSERT INTO trials (session_id, intended_action, predicted_action, confidence, reaction_time_ms, status) VALUES ($1, $2, $3, $4, $5, $6);',
                [session_id, t.intended_action, t.predicted_action, t.confidence, t.reaction_time_ms, t.status]
            );
        }
        await client.query('COMMIT');
        res.status(200).json({ success: true, session_id: session_id });
    } catch (err) {
        await client.query('ROLLBACK'); res.status(500).json({ error: err.message });
    } finally { client.release(); }
});

app.get('/api/get-sessions', async (req, res) => {
    const result = await pool.query('SELECT * FROM sessions ORDER BY session_date DESC'); res.status(200).json(result.rows);
});
app.get('/api/get-trials/:session_id', async (req, res) => {
    const result = await pool.query('SELECT * FROM trials WHERE session_id = $1 ORDER BY id ASC', [req.params.session_id]); res.status(200).json(result.rows);
});

wss.on('connection', (ws, req) => {
    const url = new URL(req.url, `http://${req.headers.host}`);
    const patientId = url.searchParams.get('patientId');
    if (patientId) console.log(`Python AI connected for Patient: ${patientId}!`);
    else console.log(`Browser UI connected!`);

    ws.on('message', (message) => {
        const messageString = message.toString();
        wss.clients.forEach((client) => {
            if (client !== ws && client.readyState === WebSocket.OPEN) client.send(messageString);
        });
    });
});

const PORT = 3000;
server.listen(PORT, () => console.log(`Clinical Server running at http://localhost:${PORT}`));