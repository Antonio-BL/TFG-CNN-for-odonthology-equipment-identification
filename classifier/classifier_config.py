from dataclasses import dataclass


@dataclass
class ClassifierConfig:
    """All tunable knobs for the tool classifier."""

    # ── Backbone / input ──────────────────────────────────────────────────────
    backbone: str = "resnet18"
    img_size: int = 224

    # ── Cross-validation ──────────────────────────────────────────────────────
    n_folds: int = 5

    # ── On-the-fly augmentation (applied to training images only) ─────────────
    aug_per_image: int = 4          # extra augmented copies per training image
    aug_rotation_degrees: float = 30.0
    aug_brightness: float = 0.3
    aug_contrast: float = 0.3
    aug_saturation: float = 0.3
    aug_hue: float = 0.1
    aug_noise_std: float = 0.05     # std of additive Gaussian noise (post-normalize)

    # ── Frozen-features classifier ────────────────────────────────────────────
    reg_C: float = 1.0              # LogisticRegression regularisation strength

    # ── Fine-tuning (optional comparison path) ────────────────────────────────
    n_classes: int = 10
    finetune_lr: float = 1e-3
    finetune_epochs: int = 20
    finetune_batch_size: int = 16

    # ── I/O ───────────────────────────────────────────────────────────────────
    data_dir: str = "./pipeline_crops"
    results_dir: str = "./cnn_results"
