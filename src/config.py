# src/config.py
import os

# Import các hàm phân phối cần thiết từ scipy.stats
# Đảm bảo bạn đã cài đặt scipy: pip install scipy
from scipy.stats import uniform, loguniform, randint

print("--- Dang doc file config.py (Multi-Model Training Mode) ---")

# --- Cấu hình MLflow ---
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "So Sanh Model Artifact")
MLFLOW_MODEL_NAME_REFERENCE = os.getenv(
    "MLFLOW_MODEL_NAME_REFERENCE", "OverallBestClassifierArtifact"
)
print(f"MLFLOW_EXPERIMENT_NAME: {MLFLOW_EXPERIMENT_NAME}")

# --- Cấu hình API (Giữ nguyên nếu cần) ---
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("PORT", "8000"))
print(f"API_HOST: {API_HOST}")
print(f"API_PORT: {API_PORT}")

# --- Cấu hình Huấn luyện & Dữ liệu (Giữ nguyên) ---
N_SAMPLES = int(os.getenv("N_SAMPLES", "1000"))
N_FEATURES = int(os.getenv("N_FEATURES", "10"))
N_CLASSES = int(os.getenv("N_CLASSES", "2"))
RANDOM_STATE_DATA = int(os.getenv("RANDOM_STATE_DATA", "42"))
TEST_SIZE = float(os.getenv("TEST_SIZE", "0.2"))
RANDOM_STATE_SPLIT = int(os.getenv("RANDOM_STATE_SPLIT", "123"))

# --- Cấu hình Tìm kiếm Siêu tham số & So sánh Model ---

# Danh sách các loại model muốn huấn luyện và so sánh
# Có thể lấy từ biến môi trường, ví dụ: "LogisticRegression,RandomForestClassifier,XGBClassifier"
MODELS_TO_TRAIN_STR = os.getenv(
    "MODELS_TO_TRAIN", "LogisticRegression,RandomForestClassifier,XGBClassifier"
)
MODELS_TO_TRAIN = [
    model.strip() for model in MODELS_TO_TRAIN_STR.split(",") if model.strip()
]

# Metric chính để so sánh hiệu năng giữa các model và chọn ra model tốt nhất
PRIMARY_METRIC = os.getenv("PRIMARY_METRIC", "f1_score_weighted")  # Ưu tiên F1-score

# Số lượng tổ hợp siêu tham số ngẫu nhiên cần thử nghiệm CHO MỖI LOẠI MODEL
N_RANDOM_ITER_PER_MODEL = int(
    os.getenv("N_RANDOM_ITER_PER_MODEL", "10")
)  # Ví dụ: 10 lần thử cho mỗi model

# Định nghĩa các *phân phối* hoặc khoảng giá trị cho siêu tham số của TỪNG LOẠI MODEL
HYPERPARAMS_DISTRIBUTIONS = {
    "LogisticRegression": {
        "classifier__C": loguniform(0.01, 100),
        "classifier__solver": ["liblinear"],
        # Lưu ý: Thêm prefix 'classifier__' vì sẽ dùng trong Pipeline
    },
    "RandomForestClassifier": {
        "classifier__n_estimators": randint(50, 201),
        "classifier__max_depth": [None, 10, 20, 30],
        "classifier__min_samples_split": randint(2, 11),
        "classifier__min_samples_leaf": randint(1, 11),
    },
    "XGBClassifier": {
        # Cần cài đặt xgboost: pip install xgboost
        "classifier__n_estimators": randint(50, 251),
        "classifier__max_depth": [3, 5, 7, 10],
        "classifier__learning_rate": loguniform(0.01, 0.3),
        "classifier__subsample": uniform(0.6, 0.4),  # sample ratio from 0.6 to 1.0
        "classifier__colsample_bytree": uniform(0.6, 0.4),
        # Các tham số khác của XGBoost có thể thêm vào đây
    },
    # Thêm các model khác nếu muốn (ví dụ: SVC)
    # "SVC": {
    #     "classifier__C": loguniform(0.1, 1000),
    #     "classifier__gamma": loguniform(0.0001, 1),
    #     "classifier__kernel": ["rbf"] # Chỉ thử RBF kernel
    # }
}

print(f"Models to train and compare: {MODELS_TO_TRAIN}")
print(f"Primary metric for comparison: {PRIMARY_METRIC}")
print(f"Random search iterations per model: {N_RANDOM_ITER_PER_MODEL}")
print(
    f"Hyperparameter distributions configured for: {list(HYPERPARAMS_DISTRIBUTIONS.keys())}"
)
print("--- Doc config.py xong ---")
