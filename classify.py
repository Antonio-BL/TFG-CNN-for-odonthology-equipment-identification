"""Tool classifier: frozen ResNet18 features + LogisticRegression, with optional fine-tuning."""

from __future__ import annotations

import json
import pathlib
from typing import Any, Callable

import joblib

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from torchvision.models import ResNet18_Weights, resnet18
from classifier_config import ClassifierConfig
from progress import fold_bar, img_bar, epoch_bar

__all__ = [
    "ToolClassifier",
    "build_backbone",
    "extract_features",
    "load_dataset",
    "train_and_evaluate",
    "finetune_and_evaluate",
    "build_classifier",
    "classify_crop",
]


# ──────────────────────────────────────────────────────────────────────────────
# Transforms
# ──────────────────────────────────────────────────────────────────────────────

class GaussianNoise:
    """Additive Gaussian noise in tensor space (applied after ImageNet normalisation)."""

    def __init__(self, std: float = 0.05) -> None:
        self.std = std

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor + torch.randn_like(tensor) * self.std


# ──────────────────────────────────────────────────────────────────────────────
# ToolClassifier wrapper
# ──────────────────────────────────────────────────────────────────────────────

class ToolClassifier:
    """Bundles backbone + downstream classifier for single-call pipeline inference.

    mode="frozen"    → classifier is a fitted sklearn estimator
    mode="finetuned" → classifier is a fine-tuned nn.Module with its own head
    """

    def __init__(
        self,
        backbone: nn.Module,
        classifier: Any,
        preprocess: Callable[[Image.Image], torch.Tensor],
        class_names: list[str],
        mode: str = "frozen",
    ) -> None:
        self.backbone = backbone
        self.classifier = classifier
        self.preprocess = preprocess
        self.class_names = class_names
        self.mode = mode


# ──────────────────────────────────────────────────────────────────────────────
# Core helpers
# ──────────────────────────────────────────────────────────────────────────────

def build_backbone() -> tuple[nn.Module, Callable]:
    """Return a frozen ResNet18 (fc=Identity, eval) on CUDA if available, plus its preprocessing transform.

    The default ResNet18 transform preserves aspect ratio and centre-crops to
    224x224 — for the very elongated tool crops produced by _extract_tool_crop
    (median aspect 6:1, up to 50:1) that throws away the tool tips and leaves
    only the handle. We replace it with a direct squash to 224x224 so the full
    tool shape is always visible. Distortion is acceptable because training and
    inference both run through the same transform.
    """
    weights = ResNet18_Weights.IMAGENET1K_V1
    model = resnet18(weights=weights)
    model.fc = nn.Identity()
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    preprocess = T.Compose([
        T.Resize((224, 224), interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return model.to(device), preprocess


def extract_features(backbone: nn.Module, img_tensor: torch.Tensor) -> np.ndarray:
    """Return L2-normalised 512-d features for a single image tensor (3,H,W) or batch."""
    if img_tensor.dim() == 3:
        img_tensor = img_tensor.unsqueeze(0)
    device = next(backbone.parameters()).device
    with torch.no_grad():
        feats = backbone(img_tensor.to(device)).cpu().numpy()
    norms = np.linalg.norm(feats, axis=1, keepdims=True)
    return feats / np.maximum(norms, 1e-12)


def load_dataset(cfg: ClassifierConfig) -> tuple[list[Image.Image], np.ndarray, list[str]]:
    """Load PIL images from ./Tools/HerramientaN/; label = 0-based position in sorted folder list.

    Returns (images, labels_array, class_names) where class_names are sorted folder names.
    """
    root = pathlib.Path(cfg.data_dir)
    folders = sorted(
        (f for f in root.iterdir() if f.is_dir() and f.name.startswith("Herramienta")),
        key=lambda p: int(p.name[len("Herramienta"):]),
    )
    images: list[Image.Image] = []
    labels: list[int] = []
    class_names: list[str] = []
    for label_idx, folder in enumerate(folders):
        class_names.append(folder.name)
        img_paths = sorted(p for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp") for p in folder.glob(ext))
        for path in img_paths:
            images.append(Image.open(path).convert("RGB"))
            labels.append(label_idx)
    return images, np.array(labels, dtype=np.int64), class_names


def _build_aug_pipeline(
    preprocess: Callable, cfg: ClassifierConfig
) -> Callable[[Image.Image], torch.Tensor]:
    """Build the training augmentation pipeline: PIL → augmented normalised tensor."""
    pil_aug = T.Compose([
        T.RandomRotation(degrees=cfg.aug_rotation_degrees),
        T.RandomHorizontalFlip(),
        T.ColorJitter(
            brightness=cfg.aug_brightness,
            contrast=cfg.aug_contrast,
            saturation=cfg.aug_saturation,
            hue=cfg.aug_hue,
        ),
    ])
    noise = GaussianNoise(cfg.aug_noise_std)

    def pipeline(img: Image.Image) -> torch.Tensor:
        return noise(preprocess(pil_aug(img)))

    return pipeline


def _augs_per_class(train_labels: np.ndarray, base_aug: int) -> dict[int, int]:
    """Return how many augmented copies each sample needs to roughly balance classes.

    The largest class keeps cfg.aug_per_image copies (so n_max samples -> n_max * (1+base_aug)
    features); smaller classes get extra copies until their total matches that target. With
    Herramienta0/1 at n=25 and Herramienta6/7 at n=7, base_aug=4 -> target=125 -> class 6/7
    get 17 copies each instead of 4, lifting their F1 off zero.
    """
    from collections import Counter
    counts = Counter(train_labels.tolist())
    n_max = max(counts.values())
    target = n_max * (1 + base_aug)
    return {cls: max(0, (target // n) - 1) for cls, n in counts.items()}


def _save_results(results: dict, cfg: ClassifierConfig, tag: str) -> pathlib.Path:
    """Save metrics to cfg.results_dir/<tag>_metrics.json and confusion matrix as .npy."""
    out = pathlib.Path(cfg.results_dir)
    out.mkdir(parents=True, exist_ok=True)

    np.save(out / f"{tag}_confusion_matrix.npy", np.array(results["confusion_matrix"]))

    payload = {k: v for k, v in results.items() if k != "confusion_matrix"}
    json_path = out / f"{tag}_metrics.json"
    with open(json_path, "w") as fh:
        json.dump(payload, fh, indent=2)

    return json_path


# ──────────────────────────────────────────────────────────────────────────────
# Frozen-features cross-validation
# ──────────────────────────────────────────────────────────────────────────────

def train_and_evaluate(cfg: ClassifierConfig) -> dict:
    """Stratified 5-fold CV: frozen ResNet18 features + sklearn LogisticRegression.

    Training images receive on-the-fly augmentation (aug_per_image copies each);
    held-out test images are never augmented.

    Returns dict with mean_accuracy, std_accuracy, fold_accuracies,
    confusion_matrix (combined), classification_report, and label_names.
    Results are also written to cfg.results_dir.
    """
    backbone, preprocess = build_backbone()
    aug_pipeline = _build_aug_pipeline(preprocess, cfg)
    images, labels, class_names = load_dataset(cfg)

    skf = StratifiedKFold(n_splits=cfg.n_folds, shuffle=True, random_state=42)
    fold_accs: list[float] = []
    all_true: list[int] = []
    all_pred: list[int] = []
    splits = list(skf.split(np.zeros(len(labels)), labels))

    fbar = fold_bar(splits, "CV folds")
    for fold_num, (train_idx, test_idx) in enumerate(fbar):
        augs = _augs_per_class(labels[train_idx], cfg.aug_per_image)
        n_total = len(test_idx) + sum(1 + augs[int(labels[i])] for i in train_idx)
        with img_bar(n_total, f"  fold {fold_num + 1} features") as bar:
            # ── Test: no augmentation ─────────────────────────────────────────
            test_feats_list = []
            for i in test_idx:
                test_feats_list.append(extract_features(backbone, preprocess(images[i])))
                bar.update(1)
            test_feats = np.vstack(test_feats_list)
            test_labels = labels[test_idx]

            # ── Train: original + per-class augmented copies ──────────────────
            train_feats: list[np.ndarray] = []
            train_labels_list: list[int] = []
            for i in train_idx:
                lbl = int(labels[i])
                train_feats.append(extract_features(backbone, preprocess(images[i])))
                train_labels_list.append(lbl)
                bar.update(1)
                for _ in range(augs[lbl]):
                    train_feats.append(extract_features(backbone, aug_pipeline(images[i])))
                    train_labels_list.append(lbl)
                    bar.update(1)

        clf = LogisticRegression(max_iter=200, C=cfg.reg_C, solver="saga",
                                 n_jobs=-1, class_weight="balanced")
        clf.fit(np.vstack(train_feats), np.array(train_labels_list))

        preds = clf.predict(test_feats)
        acc = accuracy_score(test_labels, preds)
        fold_accs.append(acc)
        all_true.extend(test_labels.tolist())
        all_pred.extend(preds.tolist())
        fbar.set_postfix(acc=f"{acc:.3f}", mean=f"{np.mean(fold_accs):.3f}")

    results = {
        "mean_accuracy": float(np.mean(fold_accs)),
        "std_accuracy": float(np.std(fold_accs)),
        "fold_accuracies": [float(a) for a in fold_accs],
        "confusion_matrix": confusion_matrix(all_true, all_pred).tolist(),
        "classification_report": classification_report(
            all_true, all_pred, target_names=class_names, output_dict=True
        ),
        "class_names": class_names,
    }
    _save_results(results, cfg, "frozen_features")
    return results


# ──────────────────────────────────────────────────────────────────────────────
# Fine-tuning cross-validation (optional comparison path)
# ──────────────────────────────────────────────────────────────────────────────

class _AugDataset(Dataset):
    def __init__(
        self,
        images: list[Image.Image],
        labels: np.ndarray,
        transform: Callable[[Image.Image], torch.Tensor],
    ) -> None:
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        return self.transform(self.images[idx]), int(self.labels[idx])


def finetune_and_evaluate(cfg: ClassifierConfig) -> dict:
    """Optional: stratified 5-fold CV fine-tuning a fresh ResNet18 + n_classes head.

    Each fold trains a new model from ImageNet weights; augmentation is identical to
    train_and_evaluate so results are directly comparable. Returns the same metrics
    dict format and writes to cfg.results_dir with the 'finetuned' tag.
    """
    weights = ResNet18_Weights.IMAGENET1K_V1
    preprocess = weights.transforms()
    aug_pipeline = _build_aug_pipeline(preprocess, cfg)
    images, labels, class_names = load_dataset(cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    skf = StratifiedKFold(n_splits=cfg.n_folds, shuffle=True, random_state=42)
    fold_accs: list[float] = []
    all_true: list[int] = []
    all_pred: list[int] = []
    splits = list(skf.split(np.zeros(len(labels)), labels))

    fbar = fold_bar(splits, "CV folds (finetune)")
    for fold_num, (train_idx, test_idx) in enumerate(fbar):
        model = resnet18(weights=weights)
        model.fc = nn.Linear(512, cfg.n_classes)
        model = model.to(device)

        train_ds = _AugDataset([images[i] for i in train_idx], labels[train_idx], aug_pipeline)
        test_ds = _AugDataset([images[i] for i in test_idx], labels[test_idx], preprocess)
        train_loader = DataLoader(train_ds, batch_size=cfg.finetune_batch_size, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=cfg.finetune_batch_size)

        optimizer = optim.Adam(model.parameters(), lr=cfg.finetune_lr)
        criterion = nn.CrossEntropyLoss()

        model.train()
        ebar = epoch_bar(cfg.finetune_epochs, f"  fold {fold_num + 1} epochs")
        for _ in ebar:
            running_loss = 0.0
            for imgs, lbls in train_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                optimizer.zero_grad()
                loss = criterion(model(imgs), lbls)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            ebar.set_postfix(loss=f"{running_loss / len(train_loader):.3f}")
        ebar.close()

        model.eval()
        fold_true: list[int] = []
        fold_pred: list[int] = []
        with torch.no_grad():
            for imgs, lbls in test_loader:
                preds = model(imgs.to(device)).argmax(dim=1).cpu().numpy()
                fold_pred.extend(preds.tolist())
                fold_true.extend(lbls.numpy().tolist())

        acc = accuracy_score(fold_true, fold_pred)
        fold_accs.append(acc)
        all_true.extend(fold_true)
        all_pred.extend(fold_pred)
        fbar.set_postfix(acc=f"{acc:.3f}", mean=f"{np.mean(fold_accs):.3f}")

    results = {
        "mean_accuracy": float(np.mean(fold_accs)),
        "std_accuracy": float(np.std(fold_accs)),
        "fold_accuracies": [float(a) for a in fold_accs],
        "confusion_matrix": confusion_matrix(all_true, all_pred).tolist(),
        "classification_report": classification_report(
            all_true, all_pred, target_names=class_names, output_dict=True
        ),
        "class_names": class_names,
    }
    _save_results(results, cfg, "finetuned")
    return results


# ──────────────────────────────────────────────────────────────────────────────
# Build a production-ready classifier (trains on all data)
# ──────────────────────────────────────────────────────────────────────────────

def build_classifier(cfg: ClassifierConfig) -> ToolClassifier:
    """Return a ToolClassifier, loading from disk if already built.

    The cache lives under cfg.results_dir so multiple model versions can
    coexist (e.g. resultados/v0/cnn_results vs resultados/v1/cnn_results).
    """
    backbone, preprocess = build_backbone()

    cache_dir = pathlib.Path(cfg.results_dir)
    clf_cache = cache_dir / "classifier.joblib"
    cn_cache = cache_dir / "class_names.json"

    if clf_cache.exists() and cn_cache.exists():
        print(f"Loading classifier from cache: {clf_cache}")
        clf = joblib.load(clf_cache)
        class_names = json.loads(cn_cache.read_text())
        return ToolClassifier(backbone, clf, preprocess, class_names, mode="frozen")

    aug_pipeline = _build_aug_pipeline(preprocess, cfg)
    images, labels, class_names = load_dataset(cfg)

    augs = _augs_per_class(labels, cfg.aug_per_image)

    all_feats: list[np.ndarray] = []
    all_labels: list[int] = []
    n_total = sum(1 + augs[int(lbl)] for lbl in labels)
    with img_bar(n_total, "Building classifier") as bar:
        for img, lbl in zip(images, labels):
            lbl = int(lbl)
            all_feats.append(extract_features(backbone, preprocess(img)))
            all_labels.append(lbl)
            bar.update(1)
            for _ in range(augs[lbl]):
                all_feats.append(extract_features(backbone, aug_pipeline(img)))
                all_labels.append(lbl)
                bar.update(1)

    clf = LogisticRegression(max_iter=200, C=cfg.reg_C, solver="saga",
                             n_jobs=-1, class_weight="balanced")
    clf.fit(np.vstack(all_feats), np.array(all_labels))

    cache_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, clf_cache)
    cn_cache.write_text(json.dumps(class_names))

    return ToolClassifier(backbone, clf, preprocess, class_names, mode="frozen")


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline inference
# ──────────────────────────────────────────────────────────────────────────────

def classify_crop(model: ToolClassifier, crop_rgb: np.ndarray) -> tuple[int, float]:
    """Classify a single RGB crop and return (label_index, confidence).

    crop_rgb: np.ndarray (H, W, 3), uint8 or float32 in [0, 1].
    label_index is 0-based; use model.class_names[label_index] for the folder name.
    """
    if crop_rgb.dtype != np.uint8:
        crop_rgb = (np.clip(crop_rgb, 0.0, 1.0) * 255).astype(np.uint8)
    pil_img = Image.fromarray(crop_rgb, mode="RGB")
    tensor = model.preprocess(pil_img)

    if model.mode == "finetuned":
        net: nn.Module = model.classifier
        device = next(net.parameters()).device
        net.eval()
        with torch.no_grad():
            logits = net(tensor.unsqueeze(0).to(device))
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    else:
        feats = extract_features(model.backbone, tensor)
        probs = model.classifier.predict_proba(feats)[0]

    label = int(probs.argmax())
    return label, float(probs[label])
