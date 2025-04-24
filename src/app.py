# src/app.py
import mlflow
import mlflow.pyfunc
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template_string
import time
import os
import traceback

# Import cấu hình (chỉ cần PORT, HOST, N_FEATURES)
from src.config import API_HOST, API_PORT, N_FEATURES

# --- Đường dẫn tới model bên trong container ---
MODEL_PATH = "/app/model_to_serve"

model = None
model_load_error = None


def load_model_from_path(path):
    """Tải model MLflow từ đường dẫn file cục bộ trong container."""
    global model, model_load_error
    model = None
    model_load_error = None
    print(f"--- Dang tai model cho API tu path: {path} ---")
    if not os.path.exists(path) or not os.path.isdir(path):
        model_load_error = f"Thu muc model khong tim thay tai: {path}. Ban da build Docker image sau khi copy model chua?"
        print(f"LOI: {model_load_error}")
        return None
    try:
        model = mlflow.pyfunc.load_model(model_uri=path)
        print(f"Tai model tu '{path}' thanh cong!")
        return model
    except Exception as e:
        print(f"Loi khi tai model tu duong dan '{path}': {e}")
        print(traceback.format_exc())
        model_load_error = f"Loi khi tai model tu '{path}': {e}"
        return None


# --- Khoi tao Flask app ---
app = Flask(__name__)

# --- Tải model ngay khi module được import (hoặc khi app tạo) ---
load_model_from_path(MODEL_PATH)


# --- Định nghĩa Route ---
@app.route("/health", methods=["GET"])
def health_check():
    """Kiem tra trang thai cua API va model"""
    if model is None and not model_load_error:  # Thử tải lại nếu chưa có lỗi trước đó
        print("Health check: Model is None, attempting reload...")
        load_model_from_path(MODEL_PATH)

    model_status = "OK" if model is not None else "ERROR"
    return jsonify(
        {
            "status": "OK",
            "model_path_configured": MODEL_PATH,
            "model_status": model_status,
            "error_message": model_load_error,
        }
    )


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": f"Model khong san sang: {model_load_error}"}), 503
    try:
        data = request.get_json()
        if (
            not data
            or "features" not in data
            or not isinstance(data["features"], list)
            or len(data["features"]) != N_FEATURES
        ):
            return (
                jsonify(
                    {
                        "error": f"Du lieu khong hop le. Can key 'features' la list co {N_FEATURES} phan tu so."
                    }
                ),
                400,
            )

        # Validate kiểu dữ liệu features
        features = data["features"]
        if not all(isinstance(x, (int, float)) for x in features):
            return (
                jsonify(
                    {
                        "error": "Tat ca cac phan tu trong 'features' phai la so (int hoac float)."
                    }
                ),
                400,
            )

        input_df = pd.DataFrame([features])
        predictions = model.predict(input_df)
        predictions_list = (
            predictions.tolist() if isinstance(predictions, np.ndarray) else predictions
        )
        print(f"Input: {features}, Prediction: {predictions_list}")
        return jsonify(
            {"predictions": predictions_list, "model_served_from_path": MODEL_PATH}
        )
    except Exception as e:
        print(f"Loi trong qua trinh predict: {e}")
        print(traceback.format_exc())
        return jsonify({"error": f"Loi server khi du doan: {str(e)}"}), 500


# ... (Route /predict_batch tương tự, đảm bảo validation input) ...
@app.route("/predict_batch", methods=["POST"])
def predict_batch():
    if model is None:
        return jsonify({"error": f"Model khong san sang: {model_load_error}"}), 503
    try:
        data = request.get_json()
        if (
            not data
            or "instances" not in data
            or not isinstance(data["instances"], list)
        ):
            return (
                jsonify(
                    {"error": "Du lieu khong hop le. Can key 'instances' la mot list."}
                ),
                400,
            )

        all_features = []
        for i, instance in enumerate(data["instances"]):
            if (
                not isinstance(instance, dict)
                or "features" not in instance
                or not isinstance(instance["features"], list)
                or len(instance["features"]) != N_FEATURES
            ):
                return (
                    jsonify(
                        {
                            "error": f"Instance {i} khong hop le. Can key 'features' la list co {N_FEATURES} phan tu so."
                        }
                    ),
                    400,
                )
            if not all(isinstance(x, (int, float)) for x in instance["features"]):
                return (
                    jsonify({"error": f"Instance {i}: Tat ca features phai la so."}),
                    400,
                )
            all_features.append(instance["features"])

        if not all_features:
            return jsonify(
                {"predictions": [], "model_served_from_path": MODEL_PATH}
            )  # Trả về rỗng nếu input rỗng

        input_df = pd.DataFrame(all_features)
        predictions = model.predict(input_df)
        predictions_list = (
            predictions.tolist() if isinstance(predictions, np.ndarray) else predictions
        )
        print(
            f"Input batch ({len(all_features)} samples), Predictions: {predictions_list}"
        )
        return jsonify(
            {"predictions": predictions_list, "model_served_from_path": MODEL_PATH}
        )

    except Exception as e:
        print(f"Loi trong qua trinh predict_batch: {e}")
        print(traceback.format_exc())
        return jsonify({"error": f"Loi server khi du doan batch: {str(e)}"}), 500


@app.route("/", methods=["GET"])
def home():
    # Form HTML đơn giản để test
    # ... (Giữ nguyên form HTML như phiên bản trước) ...
    form_html = f"""<!DOCTYPE html><html><head><title>MLflow Flask API (Artifact)</title></head><body><h1>Du doan voi Model (tu Artifact)</h1><p>Model Path Configured: {MODEL_PATH}</p><p>Trang thai model: {'OK' if model else 'ERROR'}</p>{f'<p style="color:red;">Loi load model: {model_load_error}</p>' if model_load_error else ''}<hr><form id="predictForm"><label for="features">Nhap {N_FEATURES} features (cach nhau boi dau phay):</label><br><input type="text" id="features" name="features" size="50" value="{','.join(['0.5']*N_FEATURES)}"><br><br><button type="button" onclick="sendPrediction()">Du doan</button></form><hr><h2>Ket qua:</h2><pre id="result"></pre><script>function sendPrediction() {{ const featuresInput = document.getElementById('features').value; const featuresArray = featuresInput.split(',').map(Number); const resultDiv = document.getElementById('result'); resultDiv.textContent = 'Dang gui yeu cau...'; if (featuresArray.length !== {N_FEATURES} || featuresArray.some(isNaN)) {{ resultDiv.textContent = 'Loi: Vui long nhap dung {N_FEATURES} so cach nhau boi dau phay.'; return; }} fetch('/predict', {{ method: 'POST', headers: {{ 'Content-Type': 'application/json' }}, body: JSON.stringify({{ features: featuresArray }}) }}).then(response => response.json()).then(data => {{ resultDiv.textContent = JSON.stringify(data, null, 2); }}).catch(error => {{ resultDiv.textContent = 'Loi khi goi API: ' + error; }}); }}</script></body></html>"""
    return render_template_string(form_html)


# Chạy Flask app bằng Waitress
if __name__ == "__main__":
    from waitress import serve

    print(f"\nKhoi chay Flask server bang Waitress tren http://{API_HOST}:{API_PORT}")
    print(f"API se su dung model tu path: '{MODEL_PATH}'")
    if model is None:
        print(
            f"!!! CANH BAO: Model chua duoc load thanh cong. API co the khong hoat dong dung. !!!"
        )
        print(f"!!! Loi chi tiet: {model_load_error} !!!")
    serve(app, host=API_HOST, port=API_PORT)
