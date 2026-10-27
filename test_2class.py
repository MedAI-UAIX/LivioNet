import csv
import os
import shutil
import sys

import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.stats import norm
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

sys.path.append(r'/home/user02/ZengY_task/L_scan/XX_net')

from dataset2 import test_Dataset
from pathlib import Path
from model.Livio_net import Livio_net
from model.convNext_lstm import SingleTaskConvNeXtLSTM
from model.resnet import MultiTaskResNet
from model.resnet_lstm import SingleTaskResNetLSTM
from model.ConvNext import ConvNeXt
from model.ConvNext_ema import ConvNeXt_ema
from model.singelHFnet import singelHFnet
from model.resnet_ema import ResNet_ema
from model.ConvNext_ema_time2 import ConvNeXt_ema_time2
from model.ConvNext_time import ConvNeXt_time
from model.R2plus1D import R2plus1D
from model.RHnet import RHnet
from model.Livio_net_no_time_attention import LivioNetNoTimeAttention
from model.ETS_ConvNext import ETS_ConvNeXt


os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

# =========================
# 全局参数
# =========================
N_BOOTSTRAPS = 1000
BOOTSTRAP_SEED = 20260921
TARGET_SENSITIVITY = 0.90
TARGET_SPECIFICITY = 0.90


# =========================================================
# DeLong 95% CI for AUC
# =========================================================
def compute_midrank(x):
    """Compute midranks (DeLong implementation helper)."""
    x = np.asarray(x)
    order = np.argsort(x)
    sorted_x = x[order]
    n = len(x)
    ranks = np.zeros(n, dtype=float)

    i = 0
    while i < n:
        j = i
        while j < n and sorted_x[j] == sorted_x[i]:
            j += 1
        ranks[i:j] = 0.5 * (i + j - 1)
        i = j

    out = np.empty(n, dtype=float)
    out[order] = ranks + 1.0
    return out


def fast_delong(predictions_sorted_transposed, label_1_count):
    """
    Fast DeLong covariance calculation.
    Rows = classifiers; columns = examples sorted with positives first.
    """
    m = int(label_1_count)
    n = predictions_sorted_transposed.shape[1] - m
    k = predictions_sorted_transposed.shape[0]

    if m <= 0 or n <= 0:
        raise ValueError('DeLong AUC CI requires both positive and negative samples.')

    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]

    tx = np.empty((k, m), dtype=float)
    ty = np.empty((k, n), dtype=float)
    tz = np.empty((k, m + n), dtype=float)

    for r in range(k):
        tx[r, :] = compute_midrank(positive_examples[r, :])
        ty[r, :] = compute_midrank(negative_examples[r, :])
        tz[r, :] = compute_midrank(predictions_sorted_transposed[r, :])

    aucs = tz[:, :m].sum(axis=1) / (m * n) - (m + 1.0) / (2.0 * n)

    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m

    sx = np.cov(v01)
    sy = np.cov(v10)
    delong_cov = sx / m + sy / n
    return aucs, delong_cov


def delong_auc_ci(y_true, y_score, alpha=0.05):
    """Return AUC and two-sided 95% CI calculated with DeLong's method."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)

    if len(np.unique(y_true)) < 2:
        raise ValueError('AUC requires both positive and negative samples.')

    order = np.argsort(-y_true)  # positives first
    label_1_count = int(np.sum(y_true == 1))
    predictions_sorted = y_score[np.newaxis, order]

    aucs, covariance = fast_delong(predictions_sorted, label_1_count)
    auc_value = float(aucs[0])

    covariance = np.asarray(covariance)
    if covariance.ndim == 0:
        variance = float(covariance)
    elif covariance.ndim == 1:
        variance = float(covariance[0])
    else:
        variance = float(covariance[0, 0])

    variance = max(variance, 0.0)
    se = np.sqrt(variance)
    z = norm.ppf(1 - alpha / 2)
    lower = max(0.0, auc_value - z * se)
    upper = min(1.0, auc_value + z * se)
    return auc_value, (lower, upper)


# =========================================================
# Threshold selection on validation/development data
# =========================================================
def select_three_thresholds(y_true, y_score,
                            target_sensitivity=0.90,
                            target_specificity=0.90):
    """
    Determine three operating thresholds from a validation/development set:
      1) SEN90: among thresholds with sensitivity >= target, choose the one
         with the highest specificity; ties -> higher threshold.
      2) Youden: maximize sensitivity + specificity - 1.
      3) SPE90: among thresholds with specificity >= target, choose the one
         with the highest sensitivity; ties -> lower threshold.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)

    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    specificity = 1.0 - fpr

    # sklearn may include inf as the first threshold; exclude non-finite values
    finite = np.isfinite(thresholds)
    thresholds = thresholds[finite]
    tpr = tpr[finite]
    specificity = specificity[finite]

    if len(thresholds) == 0:
        raise RuntimeError('No finite ROC thresholds were available.')

    # High-sensitivity operating point
    sen_candidates = np.where(tpr >= target_sensitivity)[0]
    if len(sen_candidates) == 0:
        raise RuntimeError(
            f'No threshold achieved sensitivity >= {target_sensitivity:.2f}.'
        )
    best_spe = np.max(specificity[sen_candidates])
    sen_candidates = sen_candidates[
        np.isclose(specificity[sen_candidates], best_spe)
    ]
    idx_sen = sen_candidates[np.argmax(thresholds[sen_candidates])]

    # Youden operating point
    youden = tpr + specificity - 1.0
    max_youden = np.max(youden)
    youden_candidates = np.where(np.isclose(youden, max_youden))[0]
    # tie-breaker: choose the point with the most balanced SEN/SPE
    if len(youden_candidates) > 1:
        balance = np.abs(tpr[youden_candidates] - specificity[youden_candidates])
        idx_youden = youden_candidates[np.argmin(balance)]
    else:
        idx_youden = youden_candidates[0]

    # High-specificity operating point
    spe_candidates = np.where(specificity >= target_specificity)[0]
    if len(spe_candidates) == 0:
        raise RuntimeError(
            f'No threshold achieved specificity >= {target_specificity:.2f}.'
        )
    best_sen = np.max(tpr[spe_candidates])
    spe_candidates = spe_candidates[np.isclose(tpr[spe_candidates], best_sen)]
    idx_spe = spe_candidates[np.argmin(thresholds[spe_candidates])]

    return {
        'SEN90': {
            'threshold': float(thresholds[idx_sen]),
            'validation_sensitivity': float(tpr[idx_sen]),
            'validation_specificity': float(specificity[idx_sen]),
            'selection_rule': f'SEN>={target_sensitivity:.0%}, then max SPE',
        },
        'Youden': {
            'threshold': float(thresholds[idx_youden]),
            'validation_sensitivity': float(tpr[idx_youden]),
            'validation_specificity': float(specificity[idx_youden]),
            'selection_rule': 'maximum Youden index',
        },
        'SPE90': {
            'threshold': float(thresholds[idx_spe]),
            'validation_sensitivity': float(tpr[idx_spe]),
            'validation_specificity': float(specificity[idx_spe]),
            'selection_rule': f'SPE>={target_specificity:.0%}, then max SEN',
        },
    }


# =========================================================
# Metrics + Bootstrap CI (except AUC)
# =========================================================
def metric_values_from_predictions(y_true, y_pred):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    n = tp + tn + fp + fn

    def safe_div(a, b):
        return a / b if b > 0 else np.nan

    acc = safe_div(tp + tn, n)
    sen = safe_div(tp, tp + fn)
    spe = safe_div(tn, tn + fp)
    ppv = safe_div(tp, tp + fp)
    npv = safe_div(tn, tn + fn)
    f1 = safe_div(2 * tp, 2 * tp + fp + fn)

    return {
        'tp': int(tp),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'ACC': acc,
        'SEN': sen,
        'SPE': spe,
        'PPV': ppv,
        'NPV': npv,
        'F1': f1,
    }


def bootstrap_metric_ci(y_true, y_score, threshold, metric_name,
                        n_bootstraps=1000, alpha=0.05, seed=20260921):
    """Participant-level percentile bootstrap CI for one threshold-dependent metric."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    n = len(y_true)
    rng = np.random.default_rng(seed)
    values = []

    for _ in range(n_bootstraps):
        idx = rng.integers(0, n, n)
        yt = y_true[idx]
        ys = y_score[idx]
        yp = (ys >= threshold).astype(int)
        metric_dict = metric_values_from_predictions(yt, yp)
        value = metric_dict[metric_name]
        if np.isfinite(value):
            values.append(float(value))

    if not values:
        return (np.nan, np.nan)

    lower = float(np.percentile(values, 100 * alpha / 2))
    upper = float(np.percentile(values, 100 * (1 - alpha / 2)))
    return lower, upper


def compute_metrics(y_true, y_score, threshold,
                    n_bootstraps=1000, seed=20260921):
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    y_pred = (y_score >= threshold).astype(int)

    point = metric_values_from_predictions(y_true, y_pred)
    auc_value, auc_ci = delong_auc_ci(y_true, y_score)

    result = {
        'tp': point['tp'],
        'tn': point['tn'],
        'fp': point['fp'],
        'fn': point['fn'],
        'threshold': float(threshold),
        'AUC': (auc_value, auc_ci),
    }

    for i, metric_name in enumerate(('ACC', 'SEN', 'SPE', 'PPV', 'NPV', 'F1')):
        ci = bootstrap_metric_ci(
            y_true,
            y_score,
            threshold,
            metric_name,
            n_bootstraps=n_bootstraps,
            seed=seed + i,
        )
        result[metric_name] = (float(point[metric_name]), ci)

    return result


# =========================================================
# Validation dataset read from CSV split
# =========================================================
class CsvSplitFrameDataset(Dataset):
    """
    Validation/development dataset when train and val cases are stored under
    the same root directory and are separated by a CSV column named `split`.

    Expected CSV columns:
        patient_id, label, split

    Each patient is expected to correspond to a folder under data_root whose
    folder name exactly matches patient_id. The patient folder contains image
    frames. Exactly `frame_count` frames are sampled uniformly from the sorted
    frame list. If fewer than frame_count frames are present, indices are
    repeated uniformly so that the model still receives a fixed-length input.
    """
    IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

    def __init__(self, data_root, csv_path, csv_split='val', frame_count=50,
                 transform=None, patient_col='patient_id', label_col='label',
                 split_col='split'):
        self.data_root = data_root
        self.frame_count = frame_count
        self.transform = transform

        df = pd.read_csv(csv_path)
        required = {patient_col, label_col, split_col}
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(
                f'CSV缺少列: {sorted(missing)}。当前列为: {list(df.columns)}'
            )

        df = df[df[split_col].astype(str).str.lower() == str(csv_split).lower()].copy()
        if len(df) == 0:
            raise ValueError(f'CSV中没有找到 split={csv_split!r} 的病例。')

        self.samples = []
        missing_folders = []
        for _, row in df.iterrows():
            patient_id = str(row[patient_col])
            label = int(row[label_col])
            patient_dir = os.path.join(data_root, patient_id)
            if not os.path.isdir(patient_dir):
                missing_folders.append(patient_id)
                continue
            self.samples.append((patient_dir, label, patient_id))

        if missing_folders:
            preview = ', '.join(missing_folders[:10])
            print(
                f'[WARNING] CSV中有 {len(missing_folders)} 个病例未在 data_root 下找到对应文件夹。'
                f'前10个: {preview}'
            )

        if len(self.samples) == 0:
            raise RuntimeError('没有可用的验证病例，请检查 data_root 与 CSV patient_id 是否对应。')

        labels = [x[1] for x in self.samples]
        n0 = sum(x == 0 for x in labels)
        n1 = sum(x == 1 for x in labels)
        print(f'Validation split loaded: n={len(self.samples)}, label0={n0}, label1={n1}')

    def __len__(self):
        return len(self.samples)

    def _get_frame_files(self, patient_dir):
        files = []
        for name in os.listdir(patient_dir):
            p = os.path.join(patient_dir, name)
            if os.path.isfile(p) and Path(name).suffix.lower() in self.IMAGE_EXTENSIONS:
                files.append(p)
        files.sort()
        return files

    def _sample_indices(self, n_frames):
        if n_frames <= 0:
            raise RuntimeError('患者文件夹内没有图像帧。')
        # np.linspace naturally repeats indices when n_frames < frame_count
        return np.linspace(0, n_frames - 1, self.frame_count, dtype=int)

    def __getitem__(self, idx):
        patient_dir, label, patient_id = self.samples[idx]
        frame_files = self._get_frame_files(patient_dir)
        if len(frame_files) == 0:
            raise RuntimeError(f'病例 {patient_id} 文件夹内没有找到图像帧: {patient_dir}')

        indices = self._sample_indices(len(frame_files))
        frames = []
        for i in indices:
            img = Image.open(frame_files[int(i)]).convert('RGB')
            if self.transform is not None:
                img = self.transform(img)
            frames.append(img)

        video = torch.stack(frames, dim=0)
        return video, torch.tensor(label, dtype=torch.float32), patient_dir


# =========================================================
# Model inference
# =========================================================
def evaluate_model(model, dataloader, criterion, device, desc='Evaluating'):
    model.eval()
    total_loss = 0.0
    all_true = []
    all_proba = []
    all_paths = []

    with torch.no_grad():
        for videos, labels, paths in tqdm(dataloader, desc=desc):
            videos = videos.to(device)
            labels = labels.to(device).float()

            output, weights = model(videos)
            probability = torch.sigmoid(output).view(-1)
            loss = criterion(output, labels.view(-1, 1))

            total_loss += loss.item()
            all_true.extend(labels.cpu().numpy())
            all_proba.extend(probability.cpu().numpy())
            all_paths.extend(paths)

    average_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0.0
    return (
        np.asarray(all_true).astype(int),
        np.asarray(all_proba, dtype=float),
        average_loss,
        all_paths,
    )


def build_loader(data_path, split, frame_count, transform,
                 batch_size=4, num_workers=2):
    dataset = test_Dataset(
        root_dir=data_path,
        frame_count=frame_count,
        transform=transform,
        split=split,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )


def build_csv_split_loader(data_root, csv_path, csv_split, frame_count, transform,
                           batch_size=4, num_workers=2):
    dataset = CsvSplitFrameDataset(
        data_root=data_root,
        csv_path=csv_path,
        csv_split=csv_split,
        frame_count=frame_count,
        transform=transform,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )


# =========================================================
# Save outputs
# =========================================================
def save_probabilities_csv(paths, y_true, y_score, threshold,
                           output_csv, misclassified_dir=None):
    y_pred = (np.asarray(y_score) >= threshold).astype(int)

    if misclassified_dir:
        os.makedirs(misclassified_dir, exist_ok=True)
        for path, true_label, pred_label in zip(paths, y_true, y_pred):
            if int(true_label) == int(pred_label):
                continue
            category_dir = os.path.join(
                misclassified_dir,
                f'true_{int(true_label)}_pred_{int(pred_label)}',
            )
            os.makedirs(category_dir, exist_ok=True)
            destination = os.path.join(category_dir, os.path.basename(path))
            if not os.path.exists(destination):
                shutil.copytree(path, destination)

    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Patient', 'True_Label', 'Pred_Label', 'Probability', 'Threshold'
        ])
        for path, true_label, pred_label, probability in zip(
                paths, y_true, y_pred, y_score):
            writer.writerow([
                os.path.basename(path),
                int(true_label),
                int(pred_label),
                float(probability),
                float(threshold),
            ])


def save_threshold_source_csv(paths, y_true, y_score, output_csv):
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Patient', 'True_Label', 'Probability'])
        for path, true_label, probability in zip(paths, y_true, y_score):
            writer.writerow([
                os.path.basename(path),
                int(true_label),
                float(probability),
            ])


def save_thresholds_csv(threshold_info, output_csv):
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Operating_Point', 'Threshold', 'Selection_Rule',
            'Validation_SEN', 'Validation_SPE'
        ])
        for name, info in threshold_info.items():
            writer.writerow([
                name,
                info['threshold'],
                info['selection_rule'],
                info['validation_sensitivity'],
                info['validation_specificity'],
            ])


def save_metrics_summary_csv(all_metrics, threshold_info, output_csv):
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Operating_Point', 'Threshold',
            'ACC', 'ACC_95CI_L', 'ACC_95CI_U',
            'AUC', 'AUC_95CI_L', 'AUC_95CI_U',
            'SEN', 'SEN_95CI_L', 'SEN_95CI_U',
            'SPE', 'SPE_95CI_L', 'SPE_95CI_U',
            'PPV', 'PPV_95CI_L', 'PPV_95CI_U',
            'NPV', 'NPV_95CI_L', 'NPV_95CI_U',
            'F1', 'F1_95CI_L', 'F1_95CI_U',
            'TP', 'TN', 'FP', 'FN',
            'Threshold_Source_SEN', 'Threshold_Source_SPE',
        ])

        for name in ('SEN90', 'Youden', 'SPE90'):
            metrics = all_metrics[name]
            row = [name, metrics['threshold']]
            for metric_name in ('ACC', 'AUC', 'SEN', 'SPE', 'PPV', 'NPV', 'F1'):
                value, ci = metrics[metric_name]
                row.extend([value, ci[0], ci[1]])
            row.extend([
                metrics['tp'], metrics['tn'], metrics['fp'], metrics['fn'],
                threshold_info[name]['validation_sensitivity'],
                threshold_info[name]['validation_specificity'],
            ])
            writer.writerow(row)


def save_pretty_metrics_csv(all_metrics, output_csv):
    """A compact table convenient for direct viewing in Excel."""
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Operating_Point', 'Threshold',
            'ACC (95% CI)', 'AUC (95% CI, DeLong)',
            'SEN (95% CI)', 'SPE (95% CI)',
            'PPV (95% CI)', 'NPV (95% CI)', 'F1 (95% CI)'
        ])
        for name in ('SEN90', 'Youden', 'SPE90'):
            metrics = all_metrics[name]
            row = [name, f"{metrics['threshold']:.6f}"]
            for metric_name in ('ACC', 'AUC', 'SEN', 'SPE', 'PPV', 'NPV', 'F1'):
                value, ci = metrics[metric_name]
                row.append(f'{value:.4f} ({ci[0]:.4f}-{ci[1]:.4f})')
            writer.writerow(row)


# =========================================================
# Main evaluation pipeline
# =========================================================
def classify_and_evaluate_three_thresholds(
        model_path,
        threshold_data_root,
        threshold_csv_path,
        threshold_split,
        test_data_path,
        test_split,
        frame_count=64,
        model_dir=None,
        batch_size=4,
        device_name='cuda:2',
        n_bootstraps=1000,
        target_sensitivity=0.90,
        target_specificity=0.90,
        save_misclassified=False,
):
    """
    IMPORTANT:
    - Thresholds are determined ONLY from rows with threshold_split in threshold_csv_path.
    - The three thresholds are then fixed and applied to the independent test set.
    - Do not point threshold_data_path/split to the final test set unless you
      intentionally want a test-set-derived exploratory analysis.
    """
    if model_dir is None:
        raise ValueError('必须指定 model_dir')

    os.makedirs(model_dir, exist_ok=True)

    if device_name.startswith('cuda') and not torch.cuda.is_available():
        device_name = 'cpu'
    device = torch.device(device_name)

    model = ConvNeXt_ema_time2()
    # model = Livio_net()
    # model = SingleTaskConvNeXtLSTM() #convnext+lstm
    # model = MultiTaskResNet() #resnet
    # model = ConvNeXt() #convnext
    # model = ConvNeXt_ema()
    # model = MultiTaskSwinTransformer() #swint
    # model = ConvNeXt_ema_lstm()
    # model = ResNet_ema_lstm()
    # model = EfficientNet()
    # model = DenseNet()
    # model = MobileNetV2()
    #model = MultiArchVideoModel(model_name='shufflenet_v2') #inception_v3  shufflenet_v2
    # model = R2plus1D()
    # model = LivioNetNoTimeAttention(num_classes=1)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225],
        ),
    ])

    criterion = torch.nn.BCEWithLogitsLoss()

    # 1) Validation/development set: determine thresholds
    threshold_loader = build_csv_split_loader(
    data_root=threshold_data_root,
    csv_path=threshold_csv_path,
    csv_split=threshold_split,
    frame_count=frame_count,
    transform=transform,
    batch_size=batch_size,
    )
    val_true, val_score, val_loss, val_paths = evaluate_model(
        model, threshold_loader, criterion, device,
        desc='Threshold-set inference'
    )

    threshold_info = select_three_thresholds(
        val_true,
        val_score,
        target_sensitivity=target_sensitivity,
        target_specificity=target_specificity,
    )

    save_threshold_source_csv(
        val_paths,
        val_true,
        val_score,
        os.path.join(model_dir, 'threshold_source_probabilities.csv'),
    )
    save_thresholds_csv(
        threshold_info,
        os.path.join(model_dir, 'selected_thresholds.csv'),
    )

    # 2) Independent test set: apply the fixed thresholds
    test_loader = build_loader(
    data_path=test_data_path,
    split=test_split,
    frame_count=frame_count,
    transform=transform,
    batch_size=batch_size,
    )
    test_true, test_score, test_loss, test_paths = evaluate_model(
        model, test_loader, criterion, device,
        desc='Test-set inference'
    )

    all_metrics = {}
    for name in ('SEN90', 'Youden', 'SPE90'):
        threshold = threshold_info[name]['threshold']
        metrics = compute_metrics(
            test_true,
            test_score,
            threshold,
            n_bootstraps=n_bootstraps,
            seed=BOOTSTRAP_SEED,
        )
        all_metrics[name] = metrics

        misclassified_dir = None
        if save_misclassified:
            misclassified_dir = os.path.join(model_dir, 'wrong', name)

        save_probabilities_csv(
            test_paths,
            test_true,
            test_score,
            threshold,
            os.path.join(model_dir, f'test_predictions_{name}.csv'),
            misclassified_dir=misclassified_dir,
        )

    save_metrics_summary_csv(
        all_metrics,
        threshold_info,
        os.path.join(model_dir, 'metrics_three_thresholds_full.csv'),
    )
    save_pretty_metrics_csv(
        all_metrics,
        os.path.join(model_dir, 'metrics_three_thresholds_pretty.csv'),
    )

    # 3) Text log
    log_path = os.path.join(model_dir, 'log_three_thresholds.log')
    with open(log_path, 'w') as logfile:
        logfile.write(f'Threshold data root: {threshold_data_root}\n')
        logfile.write(f'Threshold CSV: {threshold_csv_path}; split={threshold_split}\n')
        logfile.write(f'Threshold-set loss: {val_loss:.6f}\n')
        logfile.write(f'Test data: {test_data_path}/{test_split}\n')
        logfile.write(f'Test loss: {test_loss:.6f}\n')
        logfile.write(f'Bootstrap repetitions: {n_bootstraps}\n')
        logfile.write('AUC 95% CI method: DeLong\n')
        logfile.write('Other 95% CI method: participant-level percentile bootstrap\n\n')

        for name in ('SEN90', 'Youden', 'SPE90'):
            info = threshold_info[name]
            metrics = all_metrics[name]
            logfile.write(
                f'[{name}] threshold={info["threshold"]:.6f}; '
                f'threshold-source SEN={info["validation_sensitivity"]:.4f}; '
                f'threshold-source SPE={info["validation_specificity"]:.4f}; '
                f'rule={info["selection_rule"]}\n'
            )
            for metric_name in ('ACC', 'AUC', 'SEN', 'SPE', 'PPV', 'NPV', 'F1'):
                value, ci = metrics[metric_name]
                logfile.write(
                    f'{metric_name}: {value:.4f} '
                    f'(95% CI: {ci[0]:.4f}-{ci[1]:.4f})\n'
                )
            logfile.write(
                f'Confusion matrix counts: TP={metrics["tp"]}, '
                f'TN={metrics["tn"]}, FP={metrics["fp"]}, FN={metrics["fn"]}\n\n'
            )

    # Console summary
    print('\n================ Selected thresholds ================')
    for name, info in threshold_info.items():
        print(
            f'{name}: threshold={info["threshold"]:.6f}, '
            f'validation SEN={info["validation_sensitivity"]:.4f}, '
            f'validation SPE={info["validation_specificity"]:.4f}'
        )

    print('\n================ Test results ================')
    for name in ('SEN90', 'Youden', 'SPE90'):
        print(f'\n[{name}] threshold={all_metrics[name]["threshold"]:.6f}')
        for metric_name in ('ACC', 'AUC', 'SEN', 'SPE', 'PPV', 'NPV', 'F1'):
            value, ci = all_metrics[name][metric_name]
            print(
                f'{metric_name}: {value:.4f} '
                f'(95% CI: {ci[0]:.4f}-{ci[1]:.4f})'
            )

    print(f'\nAll tables were saved to: {model_dir}')


if __name__ == '__main__':
    model_path = r''    
    threshold_data_root = r''   

    threshold_csv_path = r''    
    threshold_split = 'val'

    test_data_path = r''     
    test_split = 'spe50'

    frame_count = 50
    model_dir = r''    
    classify_and_evaluate_three_thresholds(
        model_path=model_path,
        threshold_data_root=threshold_data_root,
        threshold_csv_path=threshold_csv_path,
        threshold_split=threshold_split,
        test_data_path=test_data_path,
        test_split=test_split,
        frame_count=frame_count,
        model_dir=model_dir,
        batch_size=4,
        device_name='cuda:2',
        n_bootstraps=N_BOOTSTRAPS,
        target_sensitivity=TARGET_SENSITIVITY,
        target_specificity=TARGET_SPECIFICITY,
        save_misclassified=False, 
    )
