# src/train.py
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import itertools
import pandas as pd
import os
import traceback
import shutil # Dùng để copy thư mục

from src.data_generator import get_data
from src.utils import evaluate_model
from src.config import (
    MLFLOW_EXPERIMENT_NAME, MLFLOW_MODEL_NAME_REFERENCE, # Chỉ dùng tên để tham khảo
    MODEL_TYPE, MODEL_PARAMS, PRIMARY_METRIC
)

# Thư mục để chứa artifact của model tốt nhất (sẽ được tạo bởi script này)
BEST_MODEL_OUTPUT_DIR = "best_model_artifact_output"

def train_and_log():
    print("--- Bat dau qua trinh huan luyen & logging (CI Artifact Mode) ---")
    print(f"MLflow Tracking URI (mac dinh): {mlflow.get_tracking_uri()} -> ./mlruns") # Sẽ dùng local ./mlruns

    try:
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
        print(f"Su dung MLflow Experiment: '{MLFLOW_EXPERIMENT_NAME}'")
    except Exception as e:
        print(f"!!! Loi khi set experiment '{MLFLOW_EXPERIMENT_NAME}': {e}")
        print(traceback.format_exc())
        print("Tiep tuc chay...") # Vẫn chạy tiếp trong CI

    X_train, X_test, y_train, y_test = get_data()

    if not MODEL_PARAMS:
        print(f"CANH BAO: Khong tim thay cau hinh sieu tham so cho model type '{MODEL_TYPE}'. Chay 1 lan voi tham so mac dinh.")
        param_combinations = [{}]
    else:
        param_names = list(MODEL_PARAMS.keys())
        param_values = list(MODEL_PARAMS.values())
        param_combinations = [dict(zip(param_names, prod)) for prod in itertools.product(*param_values)]

    best_metric_value = -float('inf')
    best_run_id = None
    experiment_id = None
    run_results = []

    print(f"\nBat dau thu nghiem {len(param_combinations)} cau hinh cho model {MODEL_TYPE}...")

    for params in param_combinations:
        try:
            with mlflow.start_run() as run:
                run_id = run.info.run_id
                if experiment_id is None: # Lấy experiment ID
                    experiment_id = run.info.experiment_id
                    print(f"Experiment ID: {experiment_id}")

                print(f"\n--- Dang chay MLflow Run ID: {run_id} ---")
                print(f"Sieu tham so: {params}")
                mlflow.log_params(params)
                mlflow.log_param("model_type", MODEL_TYPE)

                # --- Tạo và huấn luyện pipeline ---
                steps = []
                model_instance = None
                use_scaler = False
                if MODEL_TYPE == "LogisticRegression":
                    model_instance = LogisticRegression(**params, random_state=42, max_iter=1500)
                    if params.get('solver') == 'saga': use_scaler = True
                elif MODEL_TYPE == "RandomForestClassifier":
                    model_instance = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
                elif MODEL_TYPE == "SVC":
                    model_instance = SVC(**params, probability=True, random_state=42)
                    use_scaler = True
                else: raise ValueError(f"Loai model khong duoc ho tro: {MODEL_TYPE}")

                if use_scaler: steps.append(('scaler', StandardScaler()))
                steps.append(('classifier', model_instance))
                pipeline = Pipeline(steps)
                print("Dang huan luyen...")
                pipeline.fit(X_train, y_train)
                print("Dang danh gia...")
                y_pred_test = pipeline.predict(X_test)
                metrics = evaluate_model(y_test, y_pred_test)
                print("Dang log metrics...")
                mlflow.log_metrics(metrics)
                print("Dang log model artifact...")
                signature = mlflow.models.infer_signature(X_train, pipeline.predict(X_train))
                # Chỉ log model như là artifact thông thường
                mlflow.sklearn.log_model(sk_model=pipeline, artifact_path="model", signature=signature)
                print("Log model artifact thanh cong.")
                # --- Kết thúc huấn luyện & log artifact ---

                current_metric_value = metrics.get(PRIMARY_METRIC)
                if current_metric_value is not None:
                    run_results.append({
                        "run_id": run_id, "params": params, "metrics": metrics,
                        "primary_metric_value": current_metric_value
                     })
                    if current_metric_value > best_metric_value:
                         best_metric_value = current_metric_value
                         best_run_id = run_id
                         print(f"*** Run {run_id}: Model tot hon -> {PRIMARY_METRIC} = {best_metric_value:.4f} ***")
                else:
                     print(f"CANH BAO: Metric chinh '{PRIMARY_METRIC}' khong tim thay trong run {run_id}.")

        except Exception as e_run:
            print(f"!!! Loi trong MLflow Run ID {run_id}: {e_run}")
            print(traceback.format_exc())
            if mlflow.active_run():
                 try: mlflow.end_run(status="FAILED")
                 except Exception as e_log: print(f"Loi khi ket thuc run {run_id}: {e_log}")
            print("--- Ket thuc Run (FAILED) ---")
            continue

    print("\n--- Ket thuc tat ca thu nghiem ---")
    if not run_results:
        print("Khong co run nao hoan thanh thanh cong.")
        return

    # Tìm best run cuối cùng
    run_results.sort(key=lambda x: x['primary_metric_value'], reverse=True)
    best_run = run_results[0]
    best_run_id = best_run['run_id']
    best_metric_value = best_run['primary_metric_value']

    print(f"\nModel tot nhat tim duoc (dựa trên '{PRIMARY_METRIC}'):")
    print(f"  Run ID: {best_run_id}")

    # --- Copy artifact của model tốt nhất ra thư mục riêng ---
    if best_run_id and experiment_id is not None:
        # Đường dẫn đến thư mục artifact của model tốt nhất trong ./mlruns
        source_model_path = os.path.join("mlruns", str(experiment_id), best_run_id, "artifacts", "model")
        destination_path = BEST_MODEL_OUTPUT_DIR # Thư mục đích để upload artifact

        print(f"Dang chuan bi copy artifact model tot nhat tu: {source_model_path}")
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
        else:
            print(f"!!! Khong tim thay thu muc artifact model tot nhat tai: {source_model_path}")
            # Trong CI, có thể làm fail ở đây nếu muốn đảm bảo model luôn được copy
            # raise FileNotFoundError(f"Khong tim thay artifact path: {source_model_path}")
    else:
        print("Khong the xac dinh model tot nhat de copy artifact.")
        # Trong CI, có thể làm fail ở đây
        # raise ValueError("Khong tim thay best_run_id hoac experiment_id")

    print("\n--- Hoan thanh train.py ---")

if __name__ == "__main__":
    train_and_log()