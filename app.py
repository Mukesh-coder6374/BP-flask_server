from flask import Flask, request, jsonify
import numpy as np
import joblib
from scipy.signal import find_peaks
from scipy.fft import fft

app = Flask(__name__)

# Load model once at startup
model = joblib.load("PPG_bp_model.joblib")
print("BP Model Loaded.")

BUFFER_SIZE = 500
ppg_buffer = []

# Feature extraction
def extract_features(ppg):
    p2p_0 = np.ptp(ppg)
    peaks, _ = find_peaks(ppg)
    valleys, _ = find_peaks(-ppg)
    AI = (np.max(ppg[peaks]) - np.min(ppg[valleys])) / p2p_0 if len(peaks) > 0 and len(valleys) > 0 else 0
    bd = np.std(ppg) / np.mean(ppg) if np.mean(ppg) != 0 else 0
    bcda = np.mean(np.diff(ppg))
    sdoo = np.std(np.diff(ppg))
    fft_vals = np.abs(fft(ppg))
    fft_peaks, _ = find_peaks(fft_vals)
    top_fft = sorted(fft_vals[fft_peaks], reverse=True)[:3]
    top_fft += [0] * (3 - len(top_fft))
    return [p2p_0, AI, bd, bcda, sdoo] + top_fft

@app.route("/predict", methods=["POST"])
def predict():
    global ppg_buffer
    data = request.get_json()

    ir = data.get("ir")
    if ir is None:
        return jsonify({"error": "Missing IR value"}), 400

    try:
        ir = int(ir)
    except ValueError:
        return jsonify({"error": "Invalid IR format"}), 400

    ppg_buffer.append(ir)

    if len(ppg_buffer) >= BUFFER_SIZE:
        features = extract_features(np.array(ppg_buffer[:BUFFER_SIZE]))
        prediction = model.predict([features])[0]
        systolic, diastolic = prediction
        ppg_buffer = []

        return jsonify({
            "systolic": round(float(systolic), 2),
            "diastolic": round(float(diastolic), 2)
        })

    return jsonify({"status": "Collecting", "current_length": len(ppg_buffer)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
