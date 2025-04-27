# src/train.py
import mlflow
import mlflow.sklearn
import mlflow.xgboost  # Thêm import cho XGBoost

# Model classes
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier  # Cần cài đặt: pip install xgboost

# Helper classes
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import ParameterSampler

# Utilities
import pandas as pd
import os
import traceback
import shutil
import time  # Để đo thời gian huấn luyện

# Local imports
from data_generator import get_data
from utils import evaluate_model
from config import (
    MLFLOW_EXPERIMENT_NAME,
    MODELS_TO_TRAIN,  # Danh sách các model cần huấn luyện
    HYPERPARAMS_DISTRIBUTIONS,  # Phân phối tham số cho từng model
    N_RANDOM_ITER_PER_MODEL,  # Số lần random search cho mỗi model
    PRIMARY_METRIC,  # Metric chính để so sánh
    RANDOM_STATE_SPLIT,  # Dùng cho sampler và train/test split
    # ... các cấu hình khác nếu cần
)

# Thư mục để chứa artifact của model TỐT NHẤT TỔNG THỂ
BEST_MODEL_OUTPUT_DIR = "best_overall_model_artifact"

# Mapping tên model trong config sang class thực tế và yêu cầu scaler
MODEL_CONFIG = {
    "LogisticRegression": {"class": LogisticRegression, "needs_scaler": True},
    "RandomForestClassifier": {"class": RandomForestClassifier, "needs_scaler": False},
    "XGBClassifier": {
        "class": XGBClassifier,
        "needs_scaler": False,
    },  # XGBoost thường không cần scale, nhưng có thể thử
    "SVC": {"class": SVC, "needs_scaler": True},
}


def train_and_compare_models():
    print("--- Bat dau qua trinh Huan luyen & So sanh Nhieu Model ---")
    print(f"MLflow Tracking URI: {mlflow.get_tracking_uri()}")

    try:
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
        print(f"Su dung MLflow Experiment: '{MLFLOW_EXPERIMENT_NAME}'")
    except Exception as e:
        print(f"!!! Loi khi set experiment '{MLFLOW_EXPERIMENT_NAME}': {e}")
        return  # Nên dừng nếu không set được experiment

    X_train, X_test, y_train, y_test = get_data()

    overall_best_metric_value = -float("inf")
    overall_best_run_id = None
    overall_best_model_type = None
    overall_best_params = None
    overall_experiment_id = None  # Sẽ lấy từ run đầu tiên

    # Lưu kết quả tốt nhất của từng loại model
    results_per_model_type = {}

    # --- Vòng lặp ngoài: Duyệt qua từng loại model ---
    print(f"\n=== Bat dau huan luyen va tuning cho cac model: {MODELS_TO_TRAIN} ===")
    for model_type in MODELS_TO_TRAIN:
        print(f"\n--- Dang xu ly Model Type: {model_type} ---")

        if model_type not in MODEL_CONFIG:
            print(
                f"CANH BAO: Model type '{model_type}' khong duoc dinh nghia trong MODEL_CONFIG. Bo qua."
            )
            continue
        if model_type not in HYPERPARAMS_DISTRIBUTIONS:
            print(
                f"CANH BAO: Khong tim thay cau hinh phan phoi tham so cho '{model_type}' trong HYPERPARAMS_DISTRIBUTIONS. Bo qua."
            )
            continue

        model_class = MODEL_CONFIG[model_type]["class"]
        needs_scaler = MODEL_CONFIG[model_type]["needs_scaler"]
        param_dist = HYPERPARAMS_DISTRIBUTIONS[model_type]

        # --- Vòng lặp trong: Random Search cho model hiện tại ---
        print(
            f"Thuc hien Random Search voi {N_RANDOM_ITER_PER_MODEL} lan thu nghiem..."
        )
        sampler = ParameterSampler(
            param_dist, n_iter=N_RANDOM_ITER_PER_MODEL, random_state=RANDOM_STATE_SPLIT
        )
        param_combinations = list(sampler)

        best_metric_for_this_model = -float("inf")
        best_run_id_for_this_model = None
        best_params_for_this_model = None
        start_time_model_type = time.time()

        for i, params_with_prefix in enumerate(param_combinations):
            run_name = f"{model_type}_run_{i+1}"
            # Loại bỏ prefix 'classifier__' để log và khởi tạo model
            params = {k.split("__", 1)[1]: v for k, v in params_with_prefix.items()}

            try:
                with mlflow.start_run(run_name=run_name) as run:
                    run_id = run.info.run_id
                    if overall_experiment_id is None:
                        overall_experiment_id = run.info.experiment_id

                    # Log thông tin cơ bản
                    mlflow.log_param("model_type", model_type)
                    mlflow.log_params(params)
                    mlflow.log_param("random_search_iteration", i + 1)

                    # --- Xây dựng Pipeline ---
                    steps = []
                    if needs_scaler:
                        steps.append(("scaler", StandardScaler()))

                    # Khởi tạo model instance với tham số được sample
                    model_specific_params = {}
                    if model_type == "XGBClassifier":
                        # Thêm các tham số mặc định quan trọng cho XGBoost
                        model_specific_params = {
                            "random_state": RANDOM_STATE_SPLIT,
                            "use_label_encoder": False,
                            "eval_metric": "logloss",  # Hoặc 'auc', 'error', ...
                        }
                    elif model_type == "SVC":
                        model_specific_params = {
                            "probability": True,
                            "random_state": RANDOM_STATE_SPLIT,
                        }
                    else:
                        model_specific_params = {"random_state": RANDOM_STATE_SPLIT}

                    # Ghi đè tham số mặc định bằng tham số từ random search
                    model_specific_params.update(params)
                    model_instance = model_class(**model_specific_params)

                    steps.append(("classifier", model_instance))
                    pipeline = Pipeline(steps)

                    # --- Huấn luyện & Đánh giá ---
                    start_train_time = time.time()
                    pipeline.fit(X_train, y_train)
                    train_duration = time.time() - start_train_time
                    mlflow.log_metric("training_duration_seconds", train_duration)

                    y_pred_test = pipeline.predict(X_test)
                    metrics = evaluate_model(y_test, y_pred_test)
                    print(
                        f"  Metrics: {PRIMARY_METRIC}={metrics.get(PRIMARY_METRIC, 'N/A'):.4f}"
                    )

                    mlflow.log_metrics(metrics)

                    # Log model phù hợp với loại
                    log_model_args = {"artifact_path": "model"}
                    if isinstance(pipeline.named_steps["classifier"], XGBClassifier):
                        mlflow.xgboost.log_model(
                            pipeline.named_steps["classifier"], **log_model_args
                        )
                        # Hoặc log cả pipeline nếu muốn: mlflow.sklearn.log_model(pipeline, ...)
                    else:
                        # Log pipeline cho các model sklearn khác
                        try:
                            signature = mlflow.models.infer_signature(
                                X_train, pipeline.predict(X_train)
                            )
                            log_model_args["signature"] = signature
                        except Exception as e_sig:
                            print(
                                f"  !! Loi khi infer signature: {e_sig}. Log model khong co signature."
                            )
                        mlflow.sklearn.log_model(sk_model=pipeline, **log_model_args)

                    # --- Cập nhật model tốt nhất cho loại model hiện tại ---
                    current_metric_value = metrics.get(PRIMARY_METRIC)
                    if current_metric_value is not None:
                        if current_metric_value > best_metric_for_this_model:
                            best_metric_for_this_model = current_metric_value
                            best_run_id_for_this_model = run_id
                            best_params_for_this_model = params
                            print(
                                f"  *** Run {run_id} la tot nhat cho {model_type} hien tai -> {PRIMARY_METRIC} = {best_metric_for_this_model:.4f} ***"
                            )
                    else:
                        print(
                            f"  CANH BAO: Metric chinh '{PRIMARY_METRIC}' khong tim thay trong run {run_id}."
                        )

            except Exception as e_run:
                run_id_for_error = (
                    run.info.run_id
                    if "run" in locals() and run and run.info
                    else "UNKNOWN"
                )
                print(
                    f"  !!! Loi trong MLflow Run ID {run_id_for_error} ({run_name}): {e_run}"
                )
                print(traceback.format_exc())
                if mlflow.active_run():
                    try:
                        mlflow.end_run(status="FAILED")
                    except Exception as e_log:
                        print(
                            f"  Loi khi ket thuc run {run_id_for_error} (FAILED): {e_log}"
                        )
                continue  # Chuyển sang lần lặp random search tiếp theo

        # --- Kết thúc Random Search cho model hiện tại ---
        end_time_model_type = time.time()
        total_time_model_type = end_time_model_type - start_time_model_type
        print(
            f"\n--- Ket thuc tuning cho {model_type} sau {total_time_model_type:.2f} giay ---"
        )
        if best_run_id_for_this_model:
            print(
                f"  Best run for {model_type}: ID={best_run_id_for_this_model}, {PRIMARY_METRIC}={best_metric_for_this_model:.4f}"
            )
            # Lưu kết quả tốt nhất của loại model này
            results_per_model_type[model_type] = {
                "run_id": best_run_id_for_this_model,
                "metric_value": best_metric_for_this_model,
                "params": best_params_for_this_model,
            }

            # --- So sánh với model tốt nhất tổng thể hiện tại ---
            if best_metric_for_this_model > overall_best_metric_value:
                print(
                    f"  *** {model_type} voi run {best_run_id_for_this_model} la model tot nhat tong the hien tai! ***"
                )
                overall_best_metric_value = best_metric_for_this_model
                overall_best_run_id = best_run_id_for_this_model
                overall_best_model_type = model_type
                overall_best_params = best_params_for_this_model
        else:
            print(f"  Khong tim thay run nao thanh cong cho {model_type}.")

    # --- Kết thúc vòng lặp qua tất cả các loại model ---
    print("\n\n=== Ket thuc huan luyen & so sanh tat ca model ===")

    if not overall_best_run_id:
        print("!!! Khong tim thay model nao tot nhat tong the.")
        # raise RuntimeError("Khong the xac dinh model tot nhat.")
        return

    print(f"\nModel tot nhat tong the tim duoc:")
    print(f"  Loai Model: {overall_best_model_type}")
    print(f"  Run ID: {overall_best_run_id}")
    print(f"  Metric ({PRIMARY_METRIC}): {overall_best_metric_value:.4f}")
    print(f"  Sieu tham so: {overall_best_params}")

    # --- Copy artifact của model TỐT NHẤT TỔNG THỂ ---
    if overall_best_run_id and overall_experiment_id is not None:
        source_model_path = os.path.join(
            "mlruns",
            str(overall_experiment_id),
            overall_best_run_id,
            "artifacts",
            "model",
        )
        # Kiểm tra xem artifact 'model' có tồn tại không
        if not os.path.exists(source_model_path):
            # Thử tìm artifact 'xgboost_model' nếu là XGBoost và log riêng lẻ
            source_model_path_xgb = os.path.join(
                "mlruns",
                str(overall_experiment_id),
                overall_best_run_id,
                "artifacts",
                "xgboost_model",
            )
            if overall_best_model_type == "XGBClassifier" and os.path.exists(
                source_model_path_xgb
            ):
                source_model_path = source_model_path_xgb
                print(f"  (Su dung artifact 'xgboost_model' cho XGBoost)")
            else:
                print(
                    f"!!! Khong tim thay thu muc artifact 'model' (hoac 'xgboost_model') tai: {source_model_path}"
                )
                # raise FileNotFoundError(f"Khong tim thay artifact path: {source_model_path}")
                return  # Dừng nếu không tìm thấy artifact

        destination_path = BEST_MODEL_OUTPUT_DIR

        print(
            f"\nDang chuan bi copy artifact model tot nhat tong the tu: {source_model_path}"
        )
        try:
            if os.path.exists(destination_path):
                print(f"Xoa thu muc dich cu: {destination_path}")
                shutil.rmtree(destination_path)
            shutil.copytree(source_model_path, destination_path)
            print(
                f"Da copy artifact model tot nhat tong the ({overall_best_model_type}) den '{destination_path}'"
            )
        except Exception as e_copy:
            print(f"!!! Loi khi copy artifact model tot nhat tong the: {e_copy}")
            print(traceback.format_exc())
            # raise IOError(f"Loi copy artifact tu {source_model_path}")
    else:
        print(
            "CANH BAO: Khong the xac dinh model tot nhat de copy artifact (run_id hoac experiment_id bi thieu)."
        )
        # raise ValueError("Khong tim thay overall_best_run_id hoac overall_experiment_id")

    print("\n--- Hoan thanh train.py (Multi-Model Training Mode) ---")


if __name__ == "__main__":
    # Nhớ cài đặt các thư viện cần thiết:
    # pip install scikit-learn pandas mlflow scipy xgboost
    train_and_compare_models()
