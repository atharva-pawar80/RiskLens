from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, validator
import joblib
import logging
import math
from pathlib import Path


logger = logging.getLogger(__name__)

try:
    model = joblib.load(Path(__file__).with_name("fraud_detection.pkl"))
except Exception:
    logger.exception("Unable to load fraud detection model")
    model = None


app = FastAPI()


class Transaction(BaseModel):
    Time: float
    V_features: list[float]
    Amount: float

    @validator("Time", "Amount")
    def validate_finite_number(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("value must be finite")
        return value

    @validator("V_features")
    def validate_v_features(cls, values: list[float]) -> list[float]:
        if len(values) != 28:
            raise ValueError("V_features must contain exactly 28 values")
        if not all(math.isfinite(value) for value in values):
            raise ValueError("V_features values must be finite")
        return values


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ready")
def ready():
    if model is None or not callable(getattr(model, "predict", None)):
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "detail": "Model is unavailable"},
        )
    return {"status": "ready"}


@app.post("/predict")
def predict(transaction: Transaction):
    if model is None or not callable(getattr(model, "predict", None)):
        return JSONResponse(
            status_code=503,
            content={"detail": "Model is unavailable"},
        )

    row = [transaction.Time] + transaction.V_features + [transaction.Amount]
    prediction = model.predict([row])[0]
    result = bool(prediction)

    return {"is_fraud": result}

    
            

        




@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>Transaction Risk Console</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>
  :root {
    --bg: #0B1120;
    --surface: #131B2E;
    --surface-raised: #182238;
    --border: #1F2A44;
    --text: #E7ECF5;
    --text-muted: #7C8AA8;
    --accent: #2DD4BF;
    --safe: #22C55E;
    --flagged: #EF4444;
  }
  * { box-sizing: border-box; }
  body {
    background: var(--bg);
    color: var(--text);
    font-family: 'Inter', sans-serif;
    margin: 0;
    min-height: 100vh;
  }
  .topbar {
    border-bottom: 1px solid var(--border);
    padding: 20px 40px;
    display: flex;
    align-items: center;
    justify-content: space-between;
  }
  .brand {
    display: flex;
    align-items: center;
    gap: 10px;
  }
  .brand-mark {
    width: 28px; height: 28px;
    border: 1.5px solid var(--accent);
    border-radius: 6px;
    display: flex; align-items: center; justify-content: center;
    color: var(--accent);
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    font-size: 13px;
  }
  .brand-name {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 600;
    font-size: 15px;
    letter-spacing: 0.01em;
  }
  .brand-sub {
    font-size: 11px;
    color: var(--text-muted);
    font-family: 'IBM Plex Mono', monospace;
    letter-spacing: 0.05em;
  }
  .status-pill {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    color: var(--safe);
    display: flex; align-items: center; gap: 6px;
  }
  .status-dot {
    width: 6px; height: 6px; border-radius: 50%;
    background: var(--safe);
    box-shadow: 0 0 6px var(--safe);
  }

  .wrap {
    max-width: 1040px;
    margin: 0 auto;
    padding: 56px 40px 80px;
  }
  .eyebrow {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px;
    color: var(--accent);
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 10px;
  }
  h1 {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 32px;
    font-weight: 600;
    margin: 0 0 10px 0;
    letter-spacing: -0.01em;
  }
  .lede {
    color: var(--text-muted);
    font-size: 14px;
    line-height: 1.6;
    max-width: 560px;
    margin-bottom: 40px;
  }

  .console {
    display: grid;
    grid-template-columns: 1.1fr 0.9fr;
    gap: 1px;
    background: var(--border);
    border: 1px solid var(--border);
    border-radius: 12px;
    overflow: hidden;
  }
  .panel {
    background: var(--surface);
    padding: 28px;
  }
  .panel-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 18px;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }
  .ledger-id {
    color: var(--text-muted);
    font-size: 10px;
  }

  textarea {
    width: 100%;
    min-height: 220px;
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 8px;
    color: var(--text);
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12.5px;
    line-height: 1.6;
    padding: 14px;
    resize: vertical;
  }
  textarea:focus { outline: none; border-color: var(--accent); }

  .row {
    display: flex;
    gap: 10px;
    margin-top: 16px;
  }
  .btn {
    font-family: 'Inter', sans-serif;
    font-size: 13px;
    font-weight: 600;
    padding: 10px 16px;
    border-radius: 7px;
    cursor: pointer;
    border: 1px solid var(--border);
    background: var(--surface-raised);
    color: var(--text);
    transition: border-color .15s;
  }
  .btn:hover { border-color: var(--text-muted); }
  .btn-primary {
    background: var(--accent);
    color: #06231F;
    border: none;
    margin-left: auto;
  }
  .btn-primary:hover { opacity: 0.9; }

  .verdict-panel {
    display: flex;
    flex-direction: column;
    min-height: 300px;
  }
  .verdict-empty {
    flex: 1;
    display: flex;
    align-items: center;
    justify-content: center;
    color: var(--text-muted);
    font-size: 13px;
    text-align: center;
    line-height: 1.6;
  }

  .scan-track {
    position: relative;
    height: 3px;
    background: var(--border);
    border-radius: 2px;
    overflow: hidden;
    margin-bottom: 20px;
    display: none;
  }
  .scan-track.active { display: block; }
  .scan-line {
    position: absolute;
    top: 0; left: -30%;
    width: 30%; height: 100%;
    background: linear-gradient(90deg, transparent, var(--accent), transparent);
    animation: sweep 1.1s linear infinite;
  }
  @keyframes sweep {
    to { left: 110%; }
  }
  .scan-status {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px;
    color: var(--accent);
    margin-bottom: 20px;
  }

  .stamp-wrap {
    display: none;
    align-items: center;
    flex-direction: column;
    text-align: center;
    padding: 20px 0;
  }
  .stamp-wrap.active { display: flex; }
  .stamp {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    font-size: 22px;
    letter-spacing: 0.08em;
    padding: 14px 28px;
    border-radius: 8px;
    border: 2.5px solid;
    transform: rotate(-3deg);
    margin-bottom: 18px;
  }
  .stamp.safe {
    color: var(--safe);
    border-color: var(--safe);
    box-shadow: 0 0 24px -8px var(--safe);
  }
  .stamp.flagged {
    color: var(--flagged);
    border-color: var(--flagged);
    box-shadow: 0 0 24px -8px var(--flagged);
  }
  .stamp-detail {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px;
    color: var(--text-muted);
    max-width: 280px;
    line-height: 1.7;
  }

  .footnote {
    margin-top: 24px;
    font-size: 12px;
    color: var(--text-muted);
    line-height: 1.6;
    border-top: 1px solid var(--border);
    padding-top: 18px;
  }
  .footnote code {
    background: var(--surface-raised);
    padding: 1px 6px;
    border-radius: 4px;
    font-family: 'IBM Plex Mono', monospace;
    color: var(--accent);
  }
</style>
</head>
<body>

<div class="topbar">
  <div class="brand">
    <div class="brand-mark">TR</div>
    <div>
      <div class="brand-name">Transaction Risk Console</div>
      <div class="brand-sub">FRAUD DETECTION · v1</div>
    </div>
  </div>
  <div class="status-pill"><span class="status-dot"></span>MODEL ONLINE</div>
</div>

<div class="wrap">
  <div class="eyebrow">Live Inference</div>
  <h1>Screen a transaction</h1>
  <div class="lede">
    Submit transaction data to the Random Forest risk model, trained on 170K+ labeled transactions
    with a 0.17% base fraud rate. Returns a binary risk verdict.
  </div>

  <div class="console">
    <div class="panel">
      <div class="panel-label">
        <span>Transaction Ledger Entry</span>
        <span class="ledger-id" id="ledgerId">REF-000000</span>
      </div>
      <textarea id="input" spellcheck="false"></textarea>
      <div class="row">
        <button class="btn" onclick="loadExample('legit')">Load legitimate sample</button>
        <button class="btn" onclick="loadExample('fraud')">Load fraud sample</button>
        <button class="btn btn-primary" onclick="submitTransaction()">Run screening →</button>
      </div>
    </div>

    <div class="panel verdict-panel">
      <div class="panel-label"><span>Risk Verdict</span></div>

      <div class="scan-track" id="scanTrack"><div class="scan-line"></div></div>
      <div class="scan-status" id="scanStatus" style="display:none;">ANALYZING SIGNAL PATTERN…</div>

      <div class="verdict-empty" id="emptyState">
        Submit a transaction to see the risk assessment.
      </div>

      <div class="stamp-wrap" id="stampWrap">
        <div class="stamp" id="stamp"></div>
        <div class="stamp-detail" id="stampDetail"></div>
      </div>
    </div>
  </div>

  <div class="footnote">
    Model: Random Forest · Tracked in MLflow · Served via FastAPI · Containerized with Docker.
    Precision 0.96 / Recall 0.83 on held-out fraud cases. See <code>/docs</code> for the raw API.
  </div>
</div>

<script>
const examples = {
  legit: {"Time": 136131.0, "V_features": [-0.224816, 1.821721, 0.131439, 3.348614, 2.582168, 1.628348, 1.247632, 0.055309, -1.925839, 0.424950, 1.761240, -0.522738, -0.486871, -2.660399, -0.181033, 0.386488, 2.047155, -0.182370, -1.443739, 0.026941, -0.285439, -0.430957, -0.036161, -1.723991, -1.039929, -0.213176, -0.208386, -0.207902], "Amount": 0.9},
  fraud: {"Time": 102625.0, "V_features": [-4.221221, 2.871121, -5.888716, 6.890952, -3.404894, -1.154394, -7.739928, 2.851363, -2.507569, -5.110728, 5.350890, -9.299807, 2.793140, -6.106552, -2.106947, -6.250629, -13.566325, -4.192780, 0.510570, -0.227882, 1.620591, 1.567947, -0.578007, -0.059045, -1.829169, -0.072429, 0.136734, -0.599848], "Amount": 7.59}
};

function refreshRef() {
  document.getElementById('ledgerId').innerText = 'REF-' + Math.floor(100000 + Math.random()*900000);
}
refreshRef();

function loadExample(type) {
  document.getElementById('input').value = JSON.stringify(examples[type], null, 2);
  refreshRef();
}

async function submitTransaction() {
  const raw = document.getElementById('input').value;
  const emptyState = document.getElementById('emptyState');
  const stampWrap = document.getElementById('stampWrap');
  const scanTrack = document.getElementById('scanTrack');
  const scanStatus = document.getElementById('scanStatus');

  let data;
  try {
    data = JSON.parse(raw);
  } catch (e) {
    alert('Invalid JSON — check the ledger entry format.');
    return;
  }

  emptyState.style.display = 'none';
  stampWrap.classList.remove('active');
  scanTrack.classList.add('active');
  scanStatus.style.display = 'block';

  // Simulated fetch delay to let the scan animation read
  const start = Date.now();
  let result;
  try {
    const res = await fetch('/predict', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(data)
    });
    result = await res.json();
    if (!res.ok) {
      const detail = result.detail;
      const message = Array.isArray(detail)
        ? detail.map(item => item.msg).join('; ')
        : detail || 'Prediction request failed.';
      throw new Error(message);
    }
  } catch (err) {
    scanTrack.classList.remove('active');
    scanStatus.style.display = 'none';
    emptyState.style.display = 'flex';
    emptyState.innerText = 'Prediction failed: ' + err.message;
    return;
  }

  const elapsed = Date.now() - start;
  const minDelay = 700;
  if (elapsed < minDelay) {
    await new Promise(r => setTimeout(r, minDelay - elapsed));
  }

  scanTrack.classList.remove('active');
  scanStatus.style.display = 'none';

  const stamp = document.getElementById('stamp');
  const detail = document.getElementById('stampDetail');

  if (result.is_fraud) {
    stamp.className = 'stamp flagged';
    stamp.innerText = 'FLAGGED';
    detail.innerText = 'Model classified this transaction as high risk. Recommend manual review before settlement.';
  } else {
    stamp.className = 'stamp safe';
    stamp.innerText = 'APPROVED';
    detail.innerText = 'Model classified this transaction as low risk. No fraud signal detected.';
  }
  stampWrap.classList.add('active');
}
</script>

</body>
</html>
    """
