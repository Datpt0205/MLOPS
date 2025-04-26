# src/train.py
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import ParameterSampler
import pandas as pd
import os
import traceback
import shutil
import random  # Có thể không cần nếu dùng ParameterSampler với random_state

from data_generator import get_data
from utils import evaluate_model

# Import các cấu hình mới từ config.py
from config import (
    MLFLOW_EXPERIMENT_NAME,
    MODEL_TYPE,
    PARAM_DIST,  # Sử dụng cấu hình phân phối tham số
    N_RANDOM_ITER,  # Số lần lặp random search
    PRIMARY_METRIC,
    RANDOM_STATE_SPLIT,  # Có thể dùng làm random_state cho sampler
    # ... các cấu hình khác nếu cần
)

# Thư mục để chứa artifact của model tốt nhất
BEST_MODEL_OUTPUT_DIR = "best_model_artifact_output"


def train_and_log():
    print("--- Bat dau qua trinh huan luyen & logging (Random Search Mode) ---")
    print(f"MLflow Tracking URI: {mlflow.get_tracking_uri()}")

    try:
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
        print(f"Su dung MLflow Experiment: '{MLFLOW_EXPERIMENT_NAME}'")
    except Exception as e:
        print(f"!!! Loi khi set experiment '{MLFLOW_EXPERIMENT_NAME}': {e}")
        print(traceback.format_exc())
        print("Tiep tuc chay...")
        # return # Có thể muốn dừng lại nếu không set được experiment

    X_train, X_test, y_train, y_test = get_data()

    param_combinations = []
    if not PARAM_DIST:
        print(
            f"CANH BAO: Khong tim thay cau hinh phan phoi tham so cho model type '{MODEL_TYPE}'. "
            f"Chay 1 lan voi tham so mac dinh."
        )
        param_combinations = [{}]
        n_iterations = 1
    else:
        # --- Sử dụng ParameterSampler để tạo các bộ tham số ngẫu nhiên ---
        print(f"\nSu dung Random Search voi toi da {N_RANDOM_ITER} lan thu nghiem.")
        # Tạo sampler với phân phối tham số và số lần lặp mong muốn
        # random_state đảm bảo kết quả lặp lại được nếu chạy lại code
        sampler = ParameterSampler(
            PARAM_DIST, n_iter=N_RANDOM_ITER, random_state=RANDOM_STATE_SPLIT
        )
        # Tạo danh sách các bộ tham số
        param_combinations = list(sampler)
        n_iterations = len(param_combinations)
        print(f"Da tao {n_iterations} bo tham so ngau nhien.")
        if param_combinations:
            print("Vi du bo tham so dau tien:", param_combinations[0])
            if len(param_combinations) > 1:
                print("Vi du bo tham so cuoi cung:", param_combinations[-1])

    best_metric_value = -float("inf")
    best_run_id = None
    experiment_id = None
    run_results = []

    print(
        f"\nBat dau thu nghiem {n_iterations} cau hinh ngau nhien cho model {MODEL_TYPE}..."
    )

    # Vòng lặp này giờ sẽ duyệt qua các bộ tham số được lấy mẫu ngẫu nhiên
    for i, params in enumerate(param_combinations):
        run_name = f"run_{i+1}_random_{MODEL_TYPE}"  # Đặt tên run rõ ràng hơn
        try:
            # Bắt đầu một MLflow run cho mỗi bộ tham số ngẫu nhiên
            with mlflow.start_run(run_name=run_name) as run:
                run_id = run.info.run_id
                if experiment_id is None:
                    experiment_id = run.info.experiment_id
                    print(f"Experiment ID: {experiment_id}")

                print(f"\n--- Dang chay MLflow Run ID: {run_id} ({run_name}) ---")
                print(f"Sieu tham so (ngau nhien): {params}")
                mlflow.log_params(params)  # Log các tham số đã được chọn ngẫu nhiên
                mlflow.log_param("model_type", MODEL_TYPE)
                mlflow.log_param("search_iteration", i + 1)

                # --- Tạo và huấn luyện pipeline ---
                steps = []
                model_instance = None
                use_scaler = False

                # Xác định model và có cần scaler không
                if MODEL_TYPE == "LogisticRegression":
                    # Luôn đặt max_iter đủ lớn, đặc biệt khi dùng solver khác 'liblinear'
                    model_instance = LogisticRegression(
                        **params, random_state=42, max_iter=2000  # Tăng max_iter
                    )
                    # Kiểm tra solver được chọn ngẫu nhiên (nếu có nhiều lựa chọn)
                    if params.get("solver") == "saga":
                        use_scaler = True
                    # An toàn hơn là luôn dùng scaler cho Logistic Regression
                    use_scaler = True
                elif MODEL_TYPE == "RandomForestClassifier":
                    model_instance = RandomForestClassifier(
                        **params, random_state=42, n_jobs=-1
                    )
                    # RF không thường yêu cầu scaler
                elif MODEL_TYPE == "SVC":
                    # Đảm bảo probability=True nếu cần dùng predict_proba sau này
                    # hoặc để có thể tính các metric như log_loss
                    model_instance = SVC(**params, probability=True, random_state=42)
                    use_scaler = True  # SVC rất nhạy cảm với scale
                else:
                    raise ValueError(f"Loai model khong duoc ho tro: {MODEL_TYPE}")

                # Thêm scaler vào pipeline nếu cần
                if use_scaler:
                    steps.append(("scaler", StandardScaler()))
                steps.append(("classifier", model_instance))
                pipeline = Pipeline(steps)

                print("Dang huan luyen...")
                pipeline.fit(X_train, y_train)

                print("Dang danh gia...")
                y_pred_test = pipeline.predict(X_test)
                metrics = evaluate_model(y_test, y_pred_test)  # Hàm evaluate của bạn

                print("Dang log metrics...")
                mlflow.log_metrics(metrics)

                print("Dang log model artifact...")
                try:
                    # Infer signature có thể lỗi nếu X_train quá lớn hoặc có kiểu dữ liệu lạ
                    signature = mlflow.models.infer_signature(
                        X_train, pipeline.predict(X_train)
                    )
                    mlflow.sklearn.log_model(
                        sk_model=pipeline, artifact_path="model", signature=signature
                    )
                    print("Log model artifact thanh cong.")
                except Exception as e_sig:
                    print(f"!! Loi khi infer signature hoac log model: {e_sig}")
                    print("  Tiep tuc ma khong co signature...")
                    mlflow.sklearn.log_model(
                        sk_model=pipeline, artifact_path="model", signature=None
                    )
                    print("Log model artifact (khong co signature) thanh cong.")

                # --- Kết thúc huấn luyện & log artifact ---

                current_metric_value = metrics.get(PRIMARY_METRIC)
                if current_metric_value is not None:
                    run_results.append(
                        {
                            "run_id": run_id,
                            "params": params,
                            "metrics": metrics,
                            "primary_metric_value": current_metric_value,
                        }
                    )
                    # So sánh với best_metric_value tìm được cho đến nay
                    if current_metric_value > best_metric_value:
                        best_metric_value = current_metric_value
                        best_run_id = run_id
                        print(
                            f"*** Run {run_id}: Model tot hon -> {PRIMARY_METRIC} = {best_metric_value:.4f} ***"
                        )
                else:
                    print(
                        f"CANH BAO: Metric chinh '{PRIMARY_METRIC}' khong tim thay trong run {run_id}."
                    )

        except Exception as e_run:
            # Ghi nhận lỗi cụ thể của run này
            run_id_for_error = (
                run.info.run_id if "run" in locals() and run and run.info else "UNKNOWN"
            )
            print(
                f"!!! Loi trong MLflow Run ID {run_id_for_error} ({run_name}): {e_run}"
            )
            print(traceback.format_exc())
            # Cố gắng kết thúc run với trạng thái FAILED
            if mlflow.active_run():
                try:
                    mlflow.end_run(status="FAILED")
                except Exception as e_log:
                    print(f"Loi khi ket thuc run {run_id_for_error} (FAILED): {e_log}")
            print(f"--- Ket thuc Run {run_id_for_error} (FAILED) ---")
            # Không dừng toàn bộ quá trình, tiếp tục với lần lặp tiếp theo
            continue  # Quan trọng: đi đến lần lặp random search tiếp theo

    print("\n--- Ket thuc tat ca thu nghiem Random Search ---")
    if not run_results:
        print("Khong co run nao hoan thanh thanh cong.")
        # Có thể raise lỗi ở đây trong CI nếu không có run nào thành công
        # raise RuntimeError("Khong co run nao thanh cong trong qua trinh Random Search.")
        return  # Kết thúc nếu không có kết quả

    # Sắp xếp lại kết quả để tìm run tốt nhất cuối cùng
    run_results.sort(key=lambda x: x["primary_metric_value"], reverse=True)
    best_run = run_results[0]
    best_run_id = best_run["run_id"]
    best_metric_value = best_run["primary_metric_value"]
    best_params = best_run["params"]

    print(
        f"\nModel tot nhat tim duoc (dựa trên '{PRIMARY_METRIC}' = {best_metric_value:.4f}):"
    )
    print(f"  Run ID: {best_run_id}")
    print(f"  Sieu tham so tot nhat: {best_params}")

    # --- Copy artifact của model tốt nhất ra thư mục riêng ---
    if best_run_id and experiment_id is not None:
        source_model_path = os.path.join(
            "mlruns", str(experiment_id), best_run_id, "artifacts", "model"
        )
        destination_path = BEST_MODEL_OUTPUT_DIR

        print(f"\nDang chuan bi copy artifact model tot nhat tu: {source_model_path}")
        if os.path.exists(source_model_path):
            try:
                if os.path.exists(destination_path):
                    print(f"Xoa thu muc dich cu: {destination_path}")
                    shutil.rmtree(destination_path)
                shutil.copytree(source_model_path, destination_path)
                print(f"Da copy artifact model tot nhat den '{destination_path}'")
            except Exception as e_copy:
                print(f"!!! Loi khi copy artifact model tot nhat: {e_copy}")
                print(traceback.format_exc())
                # Có thể raise lỗi ở đây trong CI
                # raise IOError(f"Loi copy artifact tu {source_model_path}")
        else:
            print(
                f"!!! Khong tim thay thu muc artifact model tot nhat tai: {source_model_path}"
            )
            # Trong CI, có thể làm fail ở đây nếu muốn đảm bảo model luôn được copy
            # raise FileNotFoundError(f"Khong tim thay artifact path: {source_model_path}")
    else:
        print(
            "CANH BAO: Khong the xac dinh model tot nhat de copy artifact (best_run_id hoac experiment_id bi thieu)."
        )
        # Trong CI, có thể làm fail ở đây
        # raise ValueError("Khong tim thay best_run_id hoac experiment_id")

    print("\n--- Hoan thanh train.py (Random Search Mode) ---")


if __name__ == "__main__":
    train_and_log()
