# src/config.py
import os

# Import các hàm phân phối cần thiết từ scipy.stats
# Đảm bảo bạn đã cài đặt scipy: pip install scipy
from scipy.stats import uniform, loguniform, randint

print("--- Dang doc file config.py (Random Search Mode) ---")

# --- Cấu hình MLflow ---
MLFLOW_EXPERIMENT_NAME = os.getenv(
    "MLFLOW_EXPERIMENT_NAME", "Phan Loai Random Search Artifact"
)
MLFLOW_MODEL_NAME_REFERENCE = os.getenv(
    "MLFLOW_MODEL_NAME_REFERENCE", "BestSimpleClassifierRandomSearchArtifact"
)
print(f"MLFLOW_EXPERIMENT_NAME: {MLFLOW_EXPERIMENT_NAME}")

# --- Cấu hình API ---
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("PORT", "8000"))
print(f"API_HOST: {API_HOST}")
print(f"API_PORT: {API_PORT}")

# --- Cấu hình Huấn luyện & Dữ liệu ---
N_SAMPLES = int(os.getenv("N_SAMPLES", "1000"))
N_FEATURES = int(os.getenv("N_FEATURES", "10"))  # Giữ nguyên hoặc tăng nếu cần
N_CLASSES = int(os.getenv("N_CLASSES", "2"))
RANDOM_STATE_DATA = int(os.getenv("RANDOM_STATE_DATA", "42"))
TEST_SIZE = float(os.getenv("TEST_SIZE", "0.2"))
RANDOM_STATE_SPLIT = int(os.getenv("RANDOM_STATE_SPLIT", "123"))

# --- Cấu hình Tìm kiếm Siêu tham số ---
MODEL_TYPE = os.getenv("MODEL_TYPE", "LogisticRegression")
PRIMARY_METRIC = os.getenv("PRIMARY_METRIC", "f1_score_weighted")

# Số lượng tổ hợp siêu tham số ngẫu nhiên cần thử nghiệm
N_RANDOM_ITER = int(os.getenv("N_RANDOM_ITER", "15"))  # Tăng số lần thử lên 15

# Định nghĩa các *phân phối* hoặc khoảng giá trị cho siêu tham số
HYPERPARAMS_DISTRIBUTIONS = {
    "LogisticRegression": {
        # Sử dụng loguniform cho C: phù hợp khi tham số có thể thay đổi trên nhiều bậc độ lớn
        # Lấy mẫu ngẫu nhiên giá trị C trong khoảng [0.01, 100]
        "C": loguniform(0.01, 100),
        # Solver vẫn có thể là danh sách cố định nếu chỉ có ít lựa chọn
        "solver": ["liblinear"],
        # Nếu muốn thử cả 'saga', thêm vào list và đảm bảo train.py có StandardScaler
        # "solver": ["liblinear", "saga"],
    },
    "RandomForestClassifier": {
        # Có thể dùng list, range hoặc distribution
        "n_estimators": randint(50, 201),  # Số nguyên ngẫu nhiên từ 50 đến 200
        "max_depth": [None, 5, 10, 15, 20],  # Vẫn có thể dùng list
        "min_samples_split": randint(2, 11),  # Số nguyên ngẫu nhiên từ 2 đến 10
        "min_samples_leaf": randint(1, 11),  # Số nguyên ngẫu nhiên từ 1 đến 10
    },
    "SVC": {
        "C": loguniform(0.1, 1000),  # Khoảng rộng hơn cho SVC
        "gamma": loguniform(0.0001, 1),  # Gamma cũng thường dùng loguniform
        "kernel": ["rbf", "linear"],  # Có thể thử các kernel khác nhau
        # probability=True sẽ được set trong train.py
    },
}

# Lấy cấu hình phân phối cho model được chọn
PARAM_DIST = HYPERPARAMS_DISTRIBUTIONS.get(MODEL_TYPE, {})

print(f"MODEL_TYPE: {MODEL_TYPE}")
print(f"PRIMARY_METRIC: {PRIMARY_METRIC}")
print(f"N_RANDOM_ITER: {N_RANDOM_ITER}")
print(f"Parameter distributions configured: {bool(PARAM_DIST)}")
print("--- Doc config.py xong ---")

# Lưu ý: Không còn biến MODEL_PARAMS nữa, thay vào đó là PARAM_DIST


# # src/config.py
# import os
# from scipy.stats import uniform, loguniform

# print("--- Dang doc file config.py (CI Artifact Mode) ---")

# # --- Cấu hình MLflow ---
# # Chỉ dùng cho logging khi chạy train.py (local hoặc CI)
# MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "Phan Loai Local Artifact")
# # Tên model chỉ mang tính tham khảo, không dùng để đăng ký registry
# MLFLOW_MODEL_NAME_REFERENCE = os.getenv(
#     "MLFLOW_MODEL_NAME_REFERENCE", "BestSimpleClassifierArtifact"
# )

# print(f"MLFLOW_EXPERIMENT_NAME: {MLFLOW_EXPERIMENT_NAME}")

# # --- Cấu hình API ---
# API_HOST = os.getenv("API_HOST", "0.0.0.0")
# API_PORT = int(os.getenv("PORT", "8000"))

# print(f"API_HOST: {API_HOST}")
# print(f"API_PORT: {API_PORT}")

# # --- Cấu hình Huấn luyện & Dữ liệu ---
# N_SAMPLES = int(os.getenv("N_SAMPLES", "1000"))
# N_FEATURES = int(os.getenv("N_FEATURES", "10"))
# N_CLASSES = int(os.getenv("N_CLASSES", "2"))
# RANDOM_STATE_DATA = int(os.getenv("RANDOM_STATE_DATA", "42"))
# TEST_SIZE = float(os.getenv("TEST_SIZE", "0.2"))
# RANDOM_STATE_SPLIT = int(os.getenv("RANDOM_STATE_SPLIT", "123"))
# N_RANDOM_ITER = int(os.getenv("N_RANDOM_ITER", "10"))
# PARAM_DIST = HYPERPARAMS_DISTRIBUTIONS.get(MODEL_TYPE, {})

# MODEL_TYPE = os.getenv("MODEL_TYPE", "LogisticRegression")
# HYPERPARAMS_TO_TUNE = {
#     "LogisticRegression": {
#         "C": loguniform(0.01, 100),
#         "solver": ["liblinear"],
#     },  # Thay đổi danh sách giá trị C
#     "RandomForestClassifier": {"n_estimators": [50, 100], "max_depth": [None, 10]},
# }
# MODEL_PARAMS = HYPERPARAMS_TO_TUNE.get(MODEL_TYPE, {})
# PRIMARY_METRIC = os.getenv("PRIMARY_METRIC", "accuracy")

# print(f"MODEL_TYPE: {MODEL_TYPE}")
# print("--- Doc config.py xong ---")
