"""
Local GPU training script — reads splits from Google Drive (E:), saves checkpoints to Drive.
Usage:
    python train_local.py --model lstm
    python train_local.py --model gru
    python train_local.py --model cnn
    python train_local.py --model transformer
"""
import argparse
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent))

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')
if device.type == 'cuda':
    print(f'GPU: {torch.cuda.get_device_name(0)}')

with open('config.yaml') as f:
    cfg = yaml.safe_load(f)

DRIVE_ROOT = Path(cfg['data']['drive_root'])
SPLITS_DIR = Path(cfg['data']['splits_drive'])
CKPT_DIR   = Path('results/checkpoints')
CKPT_DIR.mkdir(parents=True, exist_ok=True)
Path('results/figures').mkdir(parents=True, exist_ok=True)

print(f'DRIVE_ROOT: {DRIVE_ROOT}')
print(f'SPLITS_DIR: {SPLITS_DIR}')


def load_splits(seq=True):
    if seq:
        X_train = np.load(SPLITS_DIR / 'X_train_seq.npy')
        y_train = np.load(SPLITS_DIR / 'y_train_seq.npy')
        X_val   = np.load(SPLITS_DIR / 'X_val_seq.npy')
        y_val   = np.load(SPLITS_DIR / 'y_val_seq.npy')
        X_test  = np.load(SPLITS_DIR / 'X_test_seq.npy')
        y_test  = np.load(SPLITS_DIR / 'y_test_seq.npy')
    else:
        X_train = np.load(SPLITS_DIR / 'X_train.npy')
        y_train = np.load(SPLITS_DIR / 'y_train.npy')
        X_val   = np.load(SPLITS_DIR / 'X_val.npy')
        y_val   = np.load(SPLITS_DIR / 'y_val.npy')
        X_test  = np.load(SPLITS_DIR / 'X_test.npy')
        y_test  = np.load(SPLITS_DIR / 'y_test.npy')
    print(f'Train: {X_train.shape}  Val: {X_val.shape}  Test: {X_test.shape}')
    return X_train, y_train, X_val, y_val, X_test, y_test


def run_lstm():
    from src.models.lstm_model import build_lstm
    from src.train import train_model
    from src.evaluate import evaluate_dl_model
    from src.utils import log_experiment

    X_train, y_train, X_val, y_val, X_test, y_test = load_splits(seq=True)
    n_features = X_train.shape[2]

    model = build_lstm(cfg, n_features)
    print(model)
    history, _ = train_model(
        model, X_train, y_train, X_val, y_val,
        cfg=cfg, model_key='lstm', checkpoint_dir=str(CKPT_DIR),
    )
    log_experiment({
        'exp_id': 'lstm_01_baseline', 'model': 'LSTM',
        'hidden_size': cfg['lstm']['hidden_size'],
        'num_layers': cfg['lstm']['num_layers'],
        'dropout': cfg['lstm']['dropout'],
        'seq_len': cfg['lstm']['seq_len'],
        'lr': cfg['lstm']['learning_rate'],
        'batch_size': cfg['lstm']['batch_size'],
        'best_val_f1': round(max(history['val_f1']), 4),
        'notes': 'baseline — local GPU',
    }, csv_path=str(DRIVE_ROOT / 'metrics_summary.csv'))

    best = build_lstm(cfg, n_features).to(device)
    best.load_state_dict(torch.load(str(CKPT_DIR / 'best_lstm.pt'), map_location=device))
    y_pred, y_prob = evaluate_dl_model(
        best, X_test, y_test, model_name='LSTM', history=history,
        figures_dir='results/figures', csv_path=str(DRIVE_ROOT / 'metrics_summary.csv'),
    )
    np.save('results/lstm_y_prob.npy', y_prob)
    shutil.copy(str(CKPT_DIR / 'best_lstm.pt'), str(DRIVE_ROOT / 'best_lstm.pt'))
    print(f'Saved best_lstm.pt to Drive')


def run_gru():
    from src.models.gru_model import build_gru
    from src.train import train_model
    from src.evaluate import evaluate_dl_model
    from src.utils import log_experiment

    X_train, y_train, X_val, y_val, X_test, y_test = load_splits(seq=True)
    n_features = X_train.shape[2]

    model = build_gru(cfg, n_features)
    print(model)
    history, _ = train_model(
        model, X_train, y_train, X_val, y_val,
        cfg=cfg, model_key='gru', checkpoint_dir=str(CKPT_DIR),
    )
    log_experiment({
        'exp_id': 'gru_01_baseline', 'model': 'GRU',
        'hidden_size': cfg['gru']['hidden_size'],
        'num_layers': cfg['gru']['num_layers'],
        'dropout': cfg['gru']['dropout'],
        'seq_len': cfg['gru']['seq_len'],
        'lr': cfg['gru']['learning_rate'],
        'batch_size': cfg['gru']['batch_size'],
        'best_val_f1': round(max(history['val_f1']), 4),
        'notes': 'baseline — local GPU',
    }, csv_path=str(DRIVE_ROOT / 'metrics_summary.csv'))

    best = build_gru(cfg, n_features).to(device)
    best.load_state_dict(torch.load(str(CKPT_DIR / 'best_gru.pt'), map_location=device))
    y_pred, y_prob = evaluate_dl_model(
        best, X_test, y_test, model_name='GRU', history=history,
        figures_dir='results/figures', csv_path=str(DRIVE_ROOT / 'metrics_summary.csv'),
    )
    np.save('results/gru_y_prob.npy', y_prob)
    shutil.copy(str(CKPT_DIR / 'best_gru.pt'), str(DRIVE_ROOT / 'best_gru.pt'))
    print(f'Saved best_gru.pt to Drive')


def run_cnn():
    from src.models.cnn_model import build_cnn
    from src.train import train_model
    from src.evaluate import evaluate_dl_model
    from src.utils import log_experiment

    X_train, y_train, X_val, y_val, X_test, y_test = load_splits(seq=True)
    n_features = X_train.shape[2]

    model = build_cnn(cfg, n_features)
    print(model)
    history, _ = train_model(
        model, X_train, y_train, X_val, y_val,
        cfg=cfg, model_key='cnn', checkpoint_dir=str(CKPT_DIR),
    )
    log_experiment({
        'exp_id': 'cnn_01_baseline', 'model': 'CNN',
        'num_filters': cfg['cnn']['num_filters'],
        'kernel_size': cfg['cnn']['kernel_size'],
        'dropout': cfg['cnn']['dropout'],
        'best_val_f1': round(max(history['val_f1']), 4),
        'notes': 'baseline — local GPU',
    }, csv_path=str(DRIVE_ROOT / 'metrics_summary.csv'))

    best = build_cnn(cfg, n_features).to(device)
    best.load_state_dict(torch.load(str(CKPT_DIR / 'best_cnn.pt'), map_location=device))
    y_pred, y_prob = evaluate_dl_model(
        best, X_test, y_test, model_name='CNN', history=history,
        figures_dir='results/figures', csv_path=str(DRIVE_ROOT / 'metrics_summary.csv'),
    )
    np.save('results/cnn_y_prob.npy', y_prob)
    shutil.copy(str(CKPT_DIR / 'best_cnn.pt'), str(DRIVE_ROOT / 'best_cnn.pt'))
    print(f'Saved best_cnn.pt to Drive')


def run_transformer():
    from src.models.transformer_model import build_transformer
    from src.train import train_model
    from src.evaluate import evaluate_dl_model
    from src.utils import log_experiment

    X_train, y_train, X_val, y_val, X_test, y_test = load_splits(seq=True)
    n_features = X_train.shape[2]

    model = build_transformer(cfg, n_features)
    print(model)
    history, _ = train_model(
        model, X_train, y_train, X_val, y_val,
        cfg=cfg, model_key='transformer', checkpoint_dir=str(CKPT_DIR),
    )
    log_experiment({
        'exp_id': 'transformer_01_baseline', 'model': 'Transformer',
        'd_model': cfg['transformer']['d_model'],
        'nhead': cfg['transformer']['nhead'],
        'num_layers': cfg['transformer']['num_layers'],
        'dropout': cfg['transformer']['dropout'],
        'best_val_f1': round(max(history['val_f1']), 4),
        'notes': 'baseline — local GPU',
    }, csv_path=str(DRIVE_ROOT / 'metrics_summary.csv'))

    best = build_transformer(cfg, n_features).to(device)
    best.load_state_dict(torch.load(str(CKPT_DIR / 'best_transformer.pt'), map_location=device))
    y_pred, y_prob = evaluate_dl_model(
        best, X_test, y_test, model_name='Transformer', history=history,
        figures_dir='results/figures', csv_path=str(DRIVE_ROOT / 'metrics_summary.csv'),
    )
    np.save('results/transformer_y_prob.npy', y_prob)
    shutil.copy(str(CKPT_DIR / 'best_transformer.pt'), str(DRIVE_ROOT / 'best_transformer.pt'))
    print(f'Saved best_transformer.pt to Drive')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, choices=['lstm', 'gru', 'cnn', 'transformer'])
    args = parser.parse_args()

    {'lstm': run_lstm, 'gru': run_gru, 'cnn': run_cnn, 'transformer': run_transformer}[args.model]()
