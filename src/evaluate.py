from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc, f1_score
)
from src.utils import plot_confusion_matrix, plot_loss_curves, plot_roc_overlay, log_experiment

CLASS_NAMES = ['normal', 'ddos']


@torch.no_grad()
def predict_dl(model, X, batch_size=1024):
    """Run inference on numpy array X, return (y_pred, y_prob_ddos)."""
    device = next(model.parameters()).device
    model.eval()
    preds, probs = [], []
    for i in range(0, len(X), batch_size):
        xb = torch.tensor(X[i:i + batch_size], dtype=torch.float32).to(device)
        logits = model(xb)
        prob = torch.softmax(logits, dim=1)[:, 1]
        preds.append(logits.argmax(1).cpu().numpy())
        probs.append(prob.cpu().numpy())
    return np.concatenate(preds), np.concatenate(probs)


def evaluate_dl_model(model, X_test, y_test, model_name, history=None,
                      figures_dir='results/figures', csv_path='results/metrics_summary.csv'):
    y_pred, y_prob = predict_dl(model, X_test)
    _report_and_save(y_test, y_pred, y_prob, model_name, history, figures_dir, csv_path)
    return y_pred, y_prob


def evaluate_rf_model(rf, X_test, y_test,
                      figures_dir='results/figures', csv_path='results/metrics_summary.csv'):
    y_pred = rf.predict(X_test)
    y_prob = rf.predict_proba(X_test)[:, 1]
    _report_and_save(y_test, y_pred, y_prob, 'RandomForest', None, figures_dir, csv_path)
    return y_pred, y_prob


def _report_and_save(y_test, y_pred, y_prob, model_name, history,
                     figures_dir, csv_path):
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    report = classification_report(y_test, y_pred, target_names=CLASS_NAMES, output_dict=True)
    print(f'\n=== {model_name} — Classification Report ===')
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    fpr_val = fp / max(fp + tn, 1)

    plot_confusion_matrix(
        cm, CLASS_NAMES,
        title=f'Confusion Matrix — {model_name}',
        save_path=figures_dir / f'cm_{model_name.lower()}.png',
    )

    fpr_arr, tpr_arr, _ = roc_curve(y_test, y_prob, pos_label=1)
    roc_auc = auc(fpr_arr, tpr_arr)

    if history:
        plot_loss_curves(
            history['train_loss'], history['val_loss'],
            title=f'Loss Curves — {model_name}',
            save_path=figures_dir / f'loss_{model_name.lower()}.png',
        )

    record = {
        'model': model_name,
        'accuracy': round(report['accuracy'], 4),
        'precision_ddos': round(report['ddos']['precision'], 4),
        'recall_ddos': round(report['ddos']['recall'], 4),
        'f1_ddos': round(report['ddos']['f1-score'], 4),
        'fpr': round(fpr_val, 4),
        'roc_auc': round(roc_auc, 4),
    }
    log_experiment(record, csv_path)
    print(f'  FPR={fpr_val:.4f}  ROC-AUC={roc_auc:.4f}')
    return fpr_arr, tpr_arr, roc_auc


def compare_roc(results: dict, save_path='results/figures/roc_comparison.png'):
    """results: {'ModelName': (fpr, tpr, auc), ...}"""
    plot_roc_overlay(results, save_path=save_path)
