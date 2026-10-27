import csv
import json
import os
import shutil
import sys

import numpy as np
from model.Livio_net import Livio_net
# from model.Livio_net_no_time_attention import LivioNetNoTimeAttention

# ========================= 可修改参数 =========================
THRESHOLD_SEARCH_MIN = 0.0       # validation 模式搜索范围（ordinal score 在 [0, 2]）
THRESHOLD_SEARCH_MAX = 2.0
THRESHOLD_SEARCH_STEP = 0.005

# 验证集：训练集/验证集共用一个病例根目录，通过 CSV 中的 split 列筛选验证患者。
VALIDATION_DATA_ROOT = r""
VALIDATION_CSV_PATH = r""
VALIDATION_CSV_SPLIT = "val"
VALIDATION_PATIENT_COL = "patient_id"
VALIDATION_LABEL_COL = "label"
VALIDATION_SPLIT_COL = "split"

N_BOOTSTRAPS = 1000
BOOTSTRAP_SEED = 20260922
CLASS_LABELS = (0, 1, 2)


class CsvSplitFrameDataset:
    """从混合病例目录中按 CSV 的 split 列读取验证患者。

    CSV 默认需要 ``patient_id``, ``label``, ``split`` 三列；病例文件夹名称必须
    与 patient_id 相同。图像帧从每个患者文件夹中按排序后的文件名均匀抽取。
    """
    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    def __init__(self, data_root, csv_path, csv_split="val", frame_count=50,
                 transform=None, patient_col="patient_id", label_col="label", split_col="split"):
        self.frame_count, self.transform = frame_count, transform
        with open(csv_path, newline="", encoding="utf-8-sig") as file:
            rows = list(csv.DictReader(file))
        if not rows or not rows[0]:
            raise ValueError("验证 CSV 为空或没有表头。")
        columns = set(rows[0])
        missing = {patient_col, label_col, split_col}.difference(columns)
        if missing:
            raise ValueError(f"CSV 缺少列: {sorted(missing)}；当前列为: {sorted(columns)}")
        selected = [row for row in rows if row[split_col].strip().lower() == str(csv_split).strip().lower()]
        if not selected:
            raise ValueError(f"CSV 中没有 split={csv_split!r} 的验证病例。")
        self.samples, missing_folders = [], []
        for row in selected:
            patient_id = str(row[patient_col]).strip()
            try:
                label = int(row[label_col])
            except ValueError as error:
                raise ValueError(f"病例 {patient_id!r} 的标签不是整数: {row[label_col]!r}") from error
            if label not in CLASS_LABELS:
                raise ValueError(f"病例 {patient_id!r} 的标签必须是 0、1 或 2，当前为 {label}。")
            patient_dir = os.path.join(data_root, patient_id)
            if os.path.isdir(patient_dir):
                self.samples.append((patient_dir, label, patient_id))
            else:
                missing_folders.append(patient_id)
        if missing_folders:
            print(f"[WARNING] 有 {len(missing_folders)} 个 CSV 病例目录不存在；前 10 个: {', '.join(missing_folders[:10])}")
        if not self.samples:
            raise RuntimeError("没有可用验证病例，请检查 data_root、CSV 的 patient_id 和 split。")
        counts = [sum(label == class_id for _, label, _ in self.samples) for class_id in CLASS_LABELS]
        print(f"CSV validation loaded: n={len(self.samples)}, label0={counts[0]}, label1={counts[1]}, label2={counts[2]}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        import torch
        from PIL import Image
        patient_dir, label, patient_id = self.samples[index]
        frame_files = sorted(os.path.join(patient_dir, name) for name in os.listdir(patient_dir)
                             if os.path.isfile(os.path.join(patient_dir, name)) and
                             os.path.splitext(name)[1].lower() in self.IMAGE_EXTENSIONS)
        if not frame_files:
            raise RuntimeError(f"病例 {patient_id} 文件夹内没有图像帧: {patient_dir}")
        indices = np.linspace(0, len(frame_files) - 1, self.frame_count, dtype=int)
        frames = []
        for frame_index in indices:
            image = Image.open(frame_files[int(frame_index)]).convert("RGB")
            frames.append(self.transform(image) if self.transform is not None else image)
        return torch.stack(frames, dim=0), torch.tensor(label, dtype=torch.long), patient_dir


def _safe_divide(numerator, denominator):
    return float(numerator / denominator) if denominator else 0.0


def _confusion_matrix(y_true, y_pred, labels):
    label_to_index = {label: index for index, label in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=int)
    for true_label, pred_label in zip(y_true, y_pred):
        if true_label not in label_to_index or pred_label not in label_to_index:
            raise ValueError("")
        matrix[label_to_index[true_label], label_to_index[pred_label]] += 1
    return matrix


def _quadratic_weighted_kappa(confusion):
    confusion = np.asarray(confusion, dtype=float)
    n_classes = confusion.shape[0]
    n_samples = confusion.sum()
    if n_samples == 0:
        raise ValueError("无法用空数据计算 Cohen's kappa。")
    indices = np.arange(n_classes, dtype=float)
    weights = (indices[:, None] - indices[None, :]) ** 2
    weights /= float((n_classes - 1) ** 2)
    observed = confusion / n_samples
    expected = np.outer(confusion.sum(axis=1) / n_samples, confusion.sum(axis=0) / n_samples)
    observed_disagreement = float(np.sum(weights * observed))
    expected_disagreement = float(np.sum(weights * expected))
    if expected_disagreement == 0:
        return 1.0 if observed_disagreement == 0 else np.nan
    return 1.0 - observed_disagreement / expected_disagreement


def compute_three_class_metrics(
        y_true, y_pred, labels=CLASS_LABELS,
        class_names=("Class_0", "Class_1", "Class_2"),
):
    """计算硬分类总体和 one-vs-rest 指标。"""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    labels, class_names = tuple(labels), tuple(class_names)
    if y_true.ndim != 1 or y_pred.ndim != 1 or len(y_true) != len(y_pred) or not len(y_true):
        raise ValueError("y_true 和 y_pred 必须是等长、非空的一维患者级标签。")
    if len(labels) != len(class_names):
        raise ValueError("labels 与 class_names 的长度必须相同。")
    confusion = _confusion_matrix(y_true, y_pred, labels)
    total = int(confusion.sum())
    class_metrics, class_f1_values = {}, []
    for index, class_name in enumerate(class_names):
        tp = int(confusion[index, index])
        fp = int(confusion[:, index].sum() - tp)
        fn = int(confusion[index, :].sum() - tp)
        tn = total - tp - fp - fn
        f1 = _safe_divide(2 * tp, 2 * tp + fp + fn)
        class_f1_values.append(f1)
        class_metrics[class_name] = {
            "precision": _safe_divide(tp, tp + fp),
            "recall": _safe_divide(tp, tp + fn),
            "specificity": _safe_divide(tn, tn + fp),
            "f1": f1, "support": int(confusion[index, :].sum()),
        }
    return {
        "quadratic_weighted_kappa": float(_quadratic_weighted_kappa(confusion)),
        "accuracy": _safe_divide(np.trace(confusion), total),
        "macro_f1": float(np.mean(class_f1_values)),
        "classes": class_metrics, "confusion_matrix": confusion,
    }


def _percentile_interval(values, alpha=0.05):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan, np.nan
    return float(np.percentile(values, 100 * alpha / 2)), float(np.percentile(values, 100 * (1 - alpha / 2)))


def bootstrap_three_class_metrics(
        y_true, y_pred, labels=CLASS_LABELS, class_names=("Class_0", "Class_1", "Class_2"),
        n_bootstraps=N_BOOTSTRAPS, seed=BOOTSTRAP_SEED, alpha=0.05,
):
    """以患者为单位、按真实类别分层的 percentile bootstrap CI。"""
    y_true, y_pred = np.asarray(y_true, dtype=int), np.asarray(y_pred, dtype=int)
    labels, class_names = tuple(labels), tuple(class_names)
    if len(y_true) != len(y_pred) or not len(y_true) or n_bootstraps <= 0:
        raise ValueError("标签必须等长非空，且 n_bootstraps 必须大于 0。")
    strata = [np.flatnonzero(y_true == label) for label in labels]
    if any(not len(indices) for indices in strata):
        raise ValueError("真实标签中缺少某类别，无法进行三分类分层 bootstrap。")
    samples = {"quadratic_weighted_kappa": [], "accuracy": [], "macro_f1": [],
               "classes": {name: {key: [] for key in ("precision", "recall", "specificity", "f1")}
                           for name in class_names}}
    rng = np.random.default_rng(seed)
    for _ in range(n_bootstraps):
        sampled_indices = np.concatenate([rng.choice(x, size=len(x), replace=True) for x in strata])
        rng.shuffle(sampled_indices)
        metrics = compute_three_class_metrics(y_true[sampled_indices], y_pred[sampled_indices], labels, class_names)
        for key in ("quadratic_weighted_kappa", "accuracy", "macro_f1"):
            samples[key].append(metrics[key])
        for name in class_names:
            for key in ("precision", "recall", "specificity", "f1"):
                samples["classes"][name][key].append(metrics["classes"][name][key])
    intervals = {key: _percentile_interval(samples[key], alpha)
                 for key in ("quadratic_weighted_kappa", "accuracy", "macro_f1")}
    intervals["classes"] = {name: {key: _percentile_interval(samples["classes"][name][key], alpha)
                                   for key in ("precision", "recall", "specificity", "f1")}
                            for name in class_names}
    return intervals


def ordinal_scores(probabilities):
    """返回 E[class] = 0*P0 + 1*P1 + 2*P2。"""
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim != 2 or probabilities.shape[1] != 3:
        raise ValueError("probabilities 必须为形状 [n_patients, 3]。")
    return probabilities @ np.asarray(CLASS_LABELS, dtype=float)


def threshold_predictions(scores, threshold_t1, threshold_t2):
    """score < T1 为 0，T1 <= score < T2 为 1，score >= T2 为 2。"""
    if not threshold_t1 < threshold_t2:
        raise ValueError("必须满足 threshold_T1 < threshold_T2。")
    scores = np.asarray(scores, dtype=float)
    return np.where(scores < threshold_t1, 0, np.where(scores < threshold_t2, 1, 2)).astype(int)


def search_optimal_thresholds(y_true, scores, step=THRESHOLD_SEARCH_STEP,
                              search_min=THRESHOLD_SEARCH_MIN, search_max=THRESHOLD_SEARCH_MAX,
                              labels=CLASS_LABELS, class_names=("Class_0", "Class_1", "Class_2")):
    """仅验证集调用：网格搜索 T1<T2，以 QWK 最大为准。"""
    if step <= 0 or search_min >= search_max:
        raise ValueError("阈值网格的范围或步长无效。")
    values = np.arange(search_min, search_max + step * 0.5, step, dtype=float)
    best = None
    for threshold_t1 in values:
        for threshold_t2 in values[values > threshold_t1]:
            predictions = threshold_predictions(scores, threshold_t1, threshold_t2)
            metrics = compute_three_class_metrics(y_true, predictions, labels, class_names)
            candidate = (metrics["quadratic_weighted_kappa"], -threshold_t1, -threshold_t2)
            if best is None or candidate > best["rank"]:
                best = {"threshold_T1": float(threshold_t1), "threshold_T2": float(threshold_t2),
                        "predictions": predictions, "metrics": metrics, "rank": candidate}
    del best["rank"]
    return best


def evaluate_model(model, dataloader, device):
    """推理一次；保留概率、argmax 和患者路径。"""
    import torch
    from tqdm import tqdm
    model.eval()
    criterion, total_loss, total_patients = torch.nn.CrossEntropyLoss(), 0.0, 0
    all_true, all_argmax, all_probabilities, all_paths = [], [], [], []
    with torch.no_grad():
        for videos, labels, paths in tqdm(dataloader, desc="Patient-level inference"):
            videos, labels = videos.to(device), labels.to(device).long()
            output = model(videos)
            logits = output[0] if isinstance(output, tuple) else output
            if logits.ndim != 2 or logits.shape[1] != 3:
                raise RuntimeError(f"模型必须输出形状 [batch, 3]，当前为 {tuple(logits.shape)}。")
            probabilities = torch.softmax(logits, dim=1)
            batch_size = int(labels.shape[0])
            total_loss += float(criterion(logits, labels).item()) * batch_size
            total_patients += batch_size
            all_true.extend(labels.cpu().numpy().astype(int))
            all_argmax.extend(probabilities.argmax(dim=1).cpu().numpy().astype(int))
            all_probabilities.extend(probabilities.cpu().numpy())
            all_paths.extend(paths)
    return (np.asarray(all_true, dtype=int), np.asarray(all_argmax, dtype=int),
            np.asarray(all_probabilities, dtype=float), all_paths,
            total_loss / total_patients if total_patients else np.nan)


def _metric_rows(point, intervals, class_names):
    for metric in ("quadratic_weighted_kappa", "accuracy", "macro_f1"):
        lower, upper = intervals[metric]
        yield "Overall", metric, point[metric], lower, upper, ""
    for class_name in class_names:
        for metric in ("precision", "recall", "specificity", "f1"):
            lower, upper = intervals["classes"][class_name][metric]
            yield class_name, metric, point["classes"][class_name][metric], lower, upper, point["classes"][class_name]["support"]


def save_metrics_csv(point, intervals, class_names, output_csv, pretty=False):
    with open(output_csv, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Scope", "Metric", "Estimate (95% CI)", "Support"] if pretty else
                        ["Scope", "Metric", "Estimate", "95CI_Lower", "95CI_Upper", "Support"])
        for scope, metric, estimate, lower, upper, support in _metric_rows(point, intervals, class_names):
            writer.writerow([scope, metric, f"{estimate:.4f} ({lower:.4f}-{upper:.4f})", support] if pretty else
                            [scope, metric, estimate, lower, upper, support])


def save_confusion_matrix_csv(confusion, class_names, output_csv):
    with open(output_csv, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["True/Pred"] + list(class_names))
        writer.writerows([[name] + [int(value) for value in row] for name, row in zip(class_names, confusion)])


def save_predictions_csv(paths, y_true, probabilities, argmax_pred, threshold_pred, scores, threshold_t1, threshold_t2, output_csv):
    with open(output_csv, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Patient", "True_Label", "Probability_Class_0", "Probability_Class_1", "Probability_Class_2",
                         "ordinal_score", "threshold_T1", "threshold_T2", "argmax_pred", "threshold_pred"])
        for path, true_label, probability, argmax, threshold, score in zip(paths, y_true, probabilities, argmax_pred, threshold_pred, scores):
            writer.writerow([os.path.basename(path), int(true_label), *map(float, probability), float(score),
                             float(threshold_t1), float(threshold_t2), int(argmax), int(threshold)])


def save_misclassified_cases(paths, y_true, y_pred, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for path, true_label, pred_label in zip(paths, y_true, y_pred):
        if int(true_label) != int(pred_label):
            category_dir = os.path.join(output_dir, f"true_{int(true_label)}_pred_{int(pred_label)}")
            os.makedirs(category_dir, exist_ok=True)
            destination = os.path.join(category_dir, os.path.basename(path))
            if not os.path.exists(destination):
                shutil.copytree(path, destination)


def _load_state_dict(checkpoint):
    return checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint


def classify_and_evaluate(model_path, validation_data_root, validation_csv_path, validation_csv_split,
                          test_data_path, test_split, frame_count=50, model_dir=None,
                          class_names=("Class_0", "Class_1", "Class_2"), batch_size=4, num_workers=2,
                          device_name="cuda:2", n_bootstraps=N_BOOTSTRAPS, bootstrap_seed=BOOTSTRAP_SEED,
                          save_misclassified=False, threshold_search_min=THRESHOLD_SEARCH_MIN,
                          threshold_search_max=THRESHOLD_SEARCH_MAX, threshold_search_step=THRESHOLD_SEARCH_STEP):
    """CSV 验证集确定阈值后，在独立测试集上固定评估。"""
    import torch
    from torch.utils.data import DataLoader
    from torchvision import transforms
    from dataset import test_Dataset
    from model.ConvNext_ema_time2 import ConvNeXt_ema_time2
    if model_dir is None or len(class_names) != 3:
        raise ValueError("必须指定 model_dir，且 class_names 必须包含三个类别。")
    os.makedirs(model_dir, exist_ok=True)
    device = torch.device("cpu" if device_name.startswith("cuda") and not torch.cuda.is_available() else device_name)
    model = Livio_net(num_classes=3)
    # model = LivioNetNoTimeAttention(num_classes=3)
    model.load_state_dict(_load_state_dict(torch.load(model_path, map_location=device)))
    model.to(device)
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    # 1) 仅用 CSV 指定的验证患者寻找 T1/T2。
    validation_dataset = CsvSplitFrameDataset(
        validation_data_root, validation_csv_path, validation_csv_split, frame_count, transform,
        VALIDATION_PATIENT_COL, VALIDATION_LABEL_COL, VALIDATION_SPLIT_COL,
    )
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    val_true, val_argmax, val_probabilities, val_paths, val_loss = evaluate_model(model, validation_loader, device)
    val_scores = ordinal_scores(val_probabilities)
    optimized = search_optimal_thresholds(val_true, val_scores, threshold_search_step, threshold_search_min,
                                          threshold_search_max, CLASS_LABELS, class_names)
    threshold_t1, threshold_t2, val_threshold = optimized["threshold_T1"], optimized["threshold_T2"], optimized["predictions"]
    with open(os.path.join(model_dir, "thresholds.json"), "w", encoding="utf-8") as file:
        json.dump({"threshold_T1": threshold_t1, "threshold_T2": threshold_t2,
                   "selection_split": validation_csv_split, "selection_objective": "quadratic_weighted_kappa",
                   "selection_qwk": optimized["metrics"]["quadratic_weighted_kappa"]}, file, indent=2)

    # 2) 独立测试集只应用上一步固定的阈值；这里不会调用任何阈值搜索函数。
    test_dataset = test_Dataset(root_dir=test_data_path, frame_count=frame_count, transform=transform, split=test_split)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    y_true, argmax_pred, probabilities, paths, average_loss = evaluate_model(model, test_loader, device)
    scores = ordinal_scores(probabilities)
    threshold_pred = threshold_predictions(scores, threshold_t1, threshold_t2)

    val_argmax_point = compute_three_class_metrics(val_true, val_argmax, CLASS_LABELS, class_names)
    val_threshold_point = compute_three_class_metrics(val_true, val_threshold, CLASS_LABELS, class_names)
    argmax_point = compute_three_class_metrics(y_true, argmax_pred, CLASS_LABELS, class_names)
    threshold_point = compute_three_class_metrics(y_true, threshold_pred, CLASS_LABELS, class_names)
    val_argmax_ci = bootstrap_three_class_metrics(val_true, val_argmax, CLASS_LABELS, class_names, n_bootstraps, bootstrap_seed)
    val_threshold_ci = bootstrap_three_class_metrics(val_true, val_threshold, CLASS_LABELS, class_names, n_bootstraps, bootstrap_seed)
    argmax_ci = bootstrap_three_class_metrics(y_true, argmax_pred, CLASS_LABELS, class_names, n_bootstraps, bootstrap_seed)
    threshold_ci = bootstrap_three_class_metrics(y_true, threshold_pred, CLASS_LABELS, class_names, n_bootstraps, bootstrap_seed)
    save_predictions_csv(val_paths, val_true, val_probabilities, val_argmax, val_threshold, val_scores, threshold_t1, threshold_t2,
                         os.path.join(model_dir, "validation_patient_predictions.csv"))
    save_predictions_csv(paths, y_true, probabilities, argmax_pred, threshold_pred, scores, threshold_t1, threshold_t2,
                         os.path.join(model_dir, "test_patient_predictions.csv"))
    for prefix, point, intervals in (("validation_argmax", val_argmax_point, val_argmax_ci),
                                     ("validation_threshold", val_threshold_point, val_threshold_ci),
                                     ("test_argmax", argmax_point, argmax_ci),
                                     ("test_threshold", threshold_point, threshold_ci)):
        save_metrics_csv(point, intervals, class_names, os.path.join(model_dir, f"metrics_{prefix}_full.csv"))
        save_metrics_csv(point, intervals, class_names, os.path.join(model_dir, f"metrics_{prefix}_pretty.csv"), pretty=True)
        save_confusion_matrix_csv(point["confusion_matrix"], class_names, os.path.join(model_dir, f"confusion_matrix_{prefix}.csv"))
    if save_misclassified:
        save_misclassified_cases(paths, y_true, threshold_pred, os.path.join(model_dir, "wrong_threshold"))
    with open(os.path.join(model_dir, "log_three_class.log"), "w", encoding="utf-8") as log:
        log.write(f"Validation CSV: {validation_csv_path}; split={validation_csv_split}\n")
        log.write(f"Validation loss: {val_loss:.6f}; Test data: {test_data_path}/{test_split}; Test loss: {average_loss:.6f}\n")
        log.write(f"Thresholds selected only from validation CSV: T1={threshold_t1:.6f}, T2={threshold_t2:.6f}\n\n")
        for name, point, intervals in (("validation_argmax", val_argmax_point, val_argmax_ci),
                                       ("validation_threshold", val_threshold_point, val_threshold_ci),
                                       ("test_argmax", argmax_point, argmax_ci),
                                       ("test_threshold", threshold_point, threshold_ci)):
            log.write(f"[{name}]\n")
            for scope, metric, estimate, lower, upper, support in _metric_rows(point, intervals, class_names):
                log.write(f"{scope} {metric}: {estimate:.4f} (95% CI: {lower:.4f}-{upper:.4f})\n")
            log.write(np.array2string(point["confusion_matrix"]) + "\n\n")
    print(f"Validation-selected T1={threshold_t1:.6f}, T2={threshold_t2:.6f}")
    print(f"Test argmax QWK={argmax_point['quadratic_weighted_kappa']:.4f}; Test threshold QWK={threshold_point['quadratic_weighted_kappa']:.4f}")
    print(f"All outputs were saved to: {model_dir}")
    return {"validation_argmax": (val_argmax_point, val_argmax_ci), "validation_threshold": (val_threshold_point, val_threshold_ci),
            "test_argmax": (argmax_point, argmax_ci), "test_threshold": (threshold_point, threshold_ci),
            "threshold_T1": threshold_t1, "threshold_T2": threshold_t2}


if __name__ == "__main__":
    sys.path.append(r"")
    model_path = r""

    validation_data_root = r""
    validation_csv_path = r""
    validation_csv_split = "val"

    test_data_path = r""
    test_split = "spe50"
    model_dir = r""
    classify_and_evaluate(model_path, validation_data_root, validation_csv_path, validation_csv_split,
                          test_data_path, test_split, frame_count=50, model_dir=model_dir,
                          class_names=("S0", "S1", "S2"), batch_size=4, num_workers=2,
                          device_name="cuda:2", n_bootstraps=N_BOOTSTRAPS,
                          bootstrap_seed=BOOTSTRAP_SEED, save_misclassified=False,
                          threshold_search_min=THRESHOLD_SEARCH_MIN, threshold_search_max=THRESHOLD_SEARCH_MAX,
                          threshold_search_step=THRESHOLD_SEARCH_STEP)
