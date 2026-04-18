from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
import joblib
import pandas as pd
import shap
import datetime

app = Flask(__name__)
app.config["SECRET_KEY"] = "upi_fraud_secret"
socketio = SocketIO(app)

# Load model and column names
model = joblib.load("model.pkl")
model_columns = joblib.load("model_columns.pkl")

# In-memory transaction log
transaction_log = []

# SHAP explainer (load once)
explainer = shap.TreeExplainer(model)  # Use TreeExplainer for tree-based models

def get_risk_level(prob):
    """Categorize risk based on fraud probability"""
    if prob < 0.3:
        return "🟢 Low Risk"
    elif prob < 0.7:
        return "🟡 Medium Risk"
    else:
        return "🔴 High Risk"

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/form")
def form():
    return render_template("form.html")

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html", transactions=transaction_log)

@app.route("/predict", methods=["POST"])
def predict():
    # Get values from form
    amount = float(request.form["amount"])
    session_duration = float(request.form["session_duration"])
    authentication_attempts = float(request.form["authentication_attempts"])
    transaction_velocity = float(request.form["transaction_velocity"])
    failed_transaction_count = float(request.form["failed_transaction_count"])

    # Get transaction type
    transaction_type = request.form["transaction_type"]
    type_map = {
        "PAYMENT": 0,
        "TRANSFER": 1,
        "CASH_OUT": 2,
        "DEBIT": 3
    }
    transaction_type_encoded = type_map[transaction_type]

    # Build input
    input_dict = {
        "amount": amount,
        "session_duration": session_duration,
        "authentication_attempts": authentication_attempts,
        "transaction_velocity": transaction_velocity,
        "failed_transaction_count": failed_transaction_count,
        "transaction_type": transaction_type_encoded
    }

    final_data = pd.DataFrame([input_dict])
    final_data = final_data.reindex(columns=model_columns, fill_value=0)

    # Predict label + probability
    prediction = model.predict(final_data)[0]
    fraud_prob = model.predict_proba(final_data)[0][1]
    risk_level = get_risk_level(fraud_prob)

    # SHAP explanation — top 3 reasons (handles all model types)
    shap_values = explainer.shap_values(final_data)

    if isinstance(shap_values, list):
        sv = shap_values[1][0] if len(shap_values) > 1 else shap_values[0][0]
    else:
        sv = shap_values[0]

    # Flatten if 2D (shape like (6,2) → take fraud class column)
    import numpy as np
    sv = np.array(sv)
    if sv.ndim == 2:
        sv = sv[:, 1] if sv.shape[1] > 1 else sv[:, 0]

    shap_series = pd.Series(sv, index=model_columns)
    top_factors = shap_series.abs().nlargest(3).index.tolist()
    shap_reasons = [f"{f}: {shap_series[f]:+.3f}" for f in top_factors]

    # Result message
    result = "⚠ Fraudulent Transaction" if prediction == 1 else "✅ Safe Transaction"

    # Log the transaction
    txn_record = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "amount": amount,
        "type": request.form["transaction_type"],
        "fraud_prob": round(fraud_prob * 100, 2),
        "risk_level": risk_level,
        "result": result,
        "reasons": shap_reasons
    }
    transaction_log.append(txn_record)

    # Emit to real-time dashboard via WebSocket
    socketio.emit("new_transaction", txn_record)

    return render_template(
        "predict.html",
        prediction_text=result,
        fraud_prob=round(fraud_prob * 100, 2),
        risk_level=risk_level,
        shap_reasons=shap_reasons
    )

@app.route("/api/transactions")
def get_transactions():
    """API endpoint to fetch all logged transactions as JSON"""
    return jsonify(transaction_log)

@app.route("/api/stats")
def get_stats():
    """Summary stats for dashboard"""
    total = len(transaction_log)
    frauds = sum(1 for t in transaction_log if "Fraudulent" in t["result"])
    safe = total - frauds
    return jsonify({
        "total": total,
        "frauds": frauds,
        "safe": safe,
        "fraud_rate": round((frauds / total) * 100, 2) if total > 0 else 0
    })

if __name__ == "__main__":
    socketio.run(app, debug=True)