import os
import yaml
import time
import mlflow

# Ensure MLflow tracking URI is set before importing/initializing libraries that may
# instantiate MLflow (for example the ultralytics callbacks). Use a file:// URI
# so MLflow recognizes the scheme on Windows.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
default_tracking_dir = os.path.join(project_root, "runs", "mlflow")
os.makedirs(default_tracking_dir, exist_ok=True)
file_tracking_uri = "file:///" + default_tracking_dir.replace("\\", "/")
# Make available as an environment variable (some libs read this on import)
os.environ.setdefault("MLFLOW_TRACKING_URI", file_tracking_uri)
mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI"))
mlflow.set_experiment("Experiment_19102025")

from ultralytics import YOLO,settings
settings.update({'mlflow': True})

def _scan_max_class_in_labels(image_dirs):
    """Scan corresponding labels dirs for max class index found."""
    max_class = -1
    for img_dir in image_dirs:
        # labels are expected in a sibling 'labels' folder next to 'images'
        if img_dir.endswith(os.sep + "images") or img_dir.endswith("/images"):
            label_dir = img_dir.rsplit(os.sep + "images", 1)[0] + os.sep + "labels"
        else:
            # fallback: try sibling labels
            label_dir = os.path.join(os.path.dirname(img_dir), "labels")

        if not os.path.isdir(label_dir):
            continue

        for fname in os.listdir(label_dir):
            if not fname.lower().endswith('.txt'):
                continue
            fpath = os.path.join(label_dir, fname)
            try:
                with open(fpath, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if not parts:
                            continue
                        try:
                            cls = int(float(parts[0]))
                        except Exception:
                            continue
                        if cls > max_class:
                            max_class = cls
            except Exception:
                continue
    return max_class


def make_dataset_yaml(train_augs, val_augs, yaml_path="configs/tmp_dataset.yaml"):
    """
    Create YOLOv8 dataset YAML with absolute paths pointing to the correct data folder.
    """
    # Force base_path to the project root (where 'data' folder lives)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    base_path = os.path.join(project_root, "data", "model")

    train_paths = [os.path.join(base_path, "train", aug, "images") for aug in train_augs]
    val_paths = [os.path.join(base_path, "val", aug, "images") for aug in val_augs]

    # determine nc by scanning label files for the highest class index
    # max_class_train = _scan_max_class_in_labels(train_paths)
    # max_class_val = _scan_max_class_in_labels(val_paths)
    # max_class = max(max_class_train, max_class_val)
    max_class_train = 4
    max_class_val = 4
    max_class = 4
    if max_class < 0:
        # no labels found, fall back to 3 and warn
        nc = 5
        print("⚠️  No label files found while scanning dataset; defaulting `nc` to 5.")
    else:
        nc = max_class + 1

    # create generic class names if there are more/unknown classes
    names = {i: f"class_{i}" for i in range(nc)}

    data = {
        "train": train_paths,
        "val": val_paths,
        "nc": nc,
        "names": names,
    }

    os.makedirs(os.path.dirname(yaml_path), exist_ok=True)

    with open(yaml_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False)

    print(f"✅ Dataset YAML created at: {os.path.abspath(yaml_path)}")
    print(f"🔗 Train paths: {train_paths}")
    print(f"🔗 Val paths: {val_paths}")

    return yaml_path


def train_with_mlflow(train_augs,
                    val_augs,
                    model_name="yolov8n.pt",
                    run_name=None,
                    epochs=50,
                    batch=16,
                    imgsz=640):
    """Create dataset YAML, run YOLO training and log results to MLflow.

    This function accepts augmentation folder names (relative to data/model/train and data/model/val).
    It will dynamically compute `nc` by scanning label files and write a temporary YAML used by YOLO.
    """
    # create dataset yaml and discover nc
    data_yaml = make_dataset_yaml(train_augs, val_augs, yaml_path=os.path.join(project_root, "configs", "tmp_dataset.yaml"))

    # Ensure a run_name
    run_name = run_name or f"train_{'_'.join(train_augs)}"
    
    # start MLflow run and train
    try:
        with mlflow.start_run(run_name=run_name):
            mlflow.log_param("model", model_name)
            mlflow.log_param("data_yaml", data_yaml)
            mlflow.log_param("epochs", epochs)
            mlflow.log_param("batch", batch)
            mlflow.log_param("imgsz", imgsz)

            model = YOLO(model_name)
            results = model.train(data=data_yaml, epochs=epochs, batch=batch, imgsz=imgsz)

            # Try to log common metrics if available
            try:
                metrics = getattr(results, 'metrics', None)
                if metrics is not None:
                    to_log = {}
                    for k in ("mAP50", "mAP50_95", "precision", "recall"):
                        v = getattr(metrics, k, None)
                        if v is not None:
                            # ensure numeric
                            try:
                                to_log[k] = float(v)
                            except Exception:
                                pass
                    if to_log:
                        mlflow.log_metrics(to_log)
            except Exception as e:
                print("⚠️  Could not extract metrics to log to MLflow:", e)

            # Log best weights if present
            try:
                artifact_path = os.path.join("runs", "detect", "train", "weights", "best.pt")
                if os.path.exists(artifact_path):
                    mlflow.log_artifact(artifact_path)
            except Exception as e:
                print("⚠️  Could not log artifact to MLflow:", e)

            print("✅ Training and MLflow logging complete!")
    except Exception as e:
        # Capture MLflow/registry errors but allow training to finish or report helpful message
        print("❌ Error during MLflow run or training:", e)
        raise


if __name__ == '__main__':
    augmentation_combos = [
        #["raw"],
        ["raw", "flip"]
        # ["raw", "rotation"],
        # ["raw", "flip", "rotation"],
        # ["raw", "gaussian"],
        # ["raw", "synthetic"]
    ]

    for combo in augmentation_combos:
        train_with_mlflow(
            train_augs=combo,
            val_augs=combo,
            model_name="yolov8n.pt",
            epochs=1,
            batch=16,
            imgsz=640
        )

