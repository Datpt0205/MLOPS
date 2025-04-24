# src/config.py
import os

print("--- Dang doc file config.py (CI Artifact Mode) ---")

# --- Cấu hình MLflow ---
# Chỉ dùng cho logging khi chạy train.py (local hoặc CI)
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "Phan Loai Local Artifact")
# Tên model chỉ mang tính tham khảo, không dùng để đăng ký registry
MLFLOW_MODEL_NAME_REFERENCE = os.getenv("MLFLOW_MODEL_NAME_REFERENCE", "BestSimpleClassifierArtifact")

print(f"MLFLOW_EXPERIMENT_NAME: {MLFLOW_EXPERIMENT_NAME}")

# --- Cấu hình API ---
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("PORT", "8000"))

print(f"API_HOST: {API_HOST}")
print(f"API_PORT: {API_PORT}")

# --- Cấu hình Huấn luyện & Dữ liệu ---
N_SAMPLES = int(os.getenv("N_SAMPLES", "1000"))
N_FEATURES = int(os.getenv("N_FEATURES", "10"))
N_CLASSES = int(os.getenv("N_CLASSES", "2"))
RANDOM_STATE_DATA = int(os.getenv("RANDOM_STATE_DATA", "42"))
TEST_SIZE = float(os.getenv("TEST_SIZE", "0.2"))
RANDOM_STATE_SPLIT = int(os.getenv("RANDOM_STATE_SPLIT", "123"))

MODEL_TYPE = os.getenv("MODEL_TYPE", "LogisticRegression")
HYPERPARAMS_TO_TUNE = {
     "LogisticRegression": { "C": [0.1, 1.0, 10.0], "solver": ["liblinear"] },
     "RandomForestClassifier": { "n_estimators": [50, 100], "max_depth": [None, 10] }
}
MODEL_PARAMS = HYPERPARAMS_TO_TUNE.get(MODEL_TYPE, {})
PRIMARY_METRIC = os.getenv("PRIMARY_METRIC", "accuracy")

print(f"MODEL_TYPE: {MODEL_TYPE}")
print("--- Doc config.py xong ---")