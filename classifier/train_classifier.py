"""Standalone script: evaluate the frozen-ResNet18 + LogisticRegression classifier
via stratified 5-fold CV, then optionally run the fine-tuning comparison.

Run from the project root with the project venv active:
    python train_classifier.py
    python train_classifier.py --finetune
"""

import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import argparse

from classifier.classifier_config import ClassifierConfig
from classifier.classify import train_and_evaluate, finetune_and_evaluate, build_classifier


def main() -> None:
    parser = argparse.ArgumentParser(description='Train and evaluate tool classifier')
    parser.add_argument('--finetune', action='store_true',
                        help='Also run the fine-tuning comparison path')
    parser.add_argument('--C', type=float, default=1.0,
                        help='LogisticRegression regularisation C (default: 1.0)')
    parser.add_argument('--aug', type=int, default=4,
                        help='Augmented copies per training image (default: 4)')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Override training data directory '
                             '(default: ClassifierConfig.data_dir)')
    parser.add_argument('--results-dir', type=str, default=None,
                        help='Override results / cache directory '
                             '(default: ClassifierConfig.results_dir)')
    parser.add_argument('--skip-build', action='store_true',
                        help='Skip building the final all-data classifier '
                             '(only run cross-validated evaluation)')
    args = parser.parse_args()

    cfg_kwargs = {'reg_C': args.C, 'aug_per_image': args.aug}
    if args.data_dir:
        cfg_kwargs['data_dir'] = args.data_dir
    if args.results_dir:
        cfg_kwargs['results_dir'] = args.results_dir
    cfg = ClassifierConfig(**cfg_kwargs)

    print('=' * 60)
    print('  Tool Classifier — Frozen ResNet18 Features + LR')
    print(f'  data_dir    : {cfg.data_dir}')
    print(f'  results_dir : {cfg.results_dir}')
    print(f'  C={cfg.reg_C}  aug_per_image={cfg.aug_per_image}  folds={cfg.n_folds}')
    print('=' * 60)

    results = train_and_evaluate(cfg)

    print()
    print(f"  Accuracy : {results['mean_accuracy']:.3f} ± {results['std_accuracy']:.3f}")
    print(f"  Per-fold : {[f'{a:.3f}' for a in results['fold_accuracies']]}")
    print()
    print("  Classification report:")
    report = results['classification_report']
    for cls, metrics in report.items():
        if isinstance(metrics, dict):
            print(f"    {cls:<20s}  "
                  f"P={metrics['precision']:.2f}  "
                  f"R={metrics['recall']:.2f}  "
                  f"F1={metrics['f1-score']:.2f}  "
                  f"n={int(metrics['support'])}")
    print()
    print(f"  Results saved to: {cfg.results_dir}/")

    if not args.skip_build:
        print()
        print('=' * 60)
        print('  Building final classifier on all data (for inference)')
        print('=' * 60)
        build_classifier(cfg)
        print(f"  classifier.joblib + class_names.json saved to: {cfg.results_dir}/")

    if args.finetune:
        print()
        print('=' * 60)
        print('  Fine-tuning comparison (ResNet18 + 10-way head)')
        print('=' * 60)
        ft_results = finetune_and_evaluate(cfg)
        print()
        print(f"  Accuracy : {ft_results['mean_accuracy']:.3f} ± {ft_results['std_accuracy']:.3f}")
        print(f"  Per-fold : {[f'{a:.3f}' for a in ft_results['fold_accuracies']]}")
        print()
        delta = ft_results['mean_accuracy'] - results['mean_accuracy']
        print(f"  Δ vs frozen features: {delta:+.3f}")


if __name__ == '__main__':
    main()
