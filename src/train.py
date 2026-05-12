import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score
try:
    from tqdm.notebook import tqdm
except Exception:
    from tqdm import tqdm


def make_dataloaders(X_train, y_train, X_val, y_val, batch_size):
    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    val_ds = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.long),
    )
    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=pin)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=0, pin_memory=pin)
    return train_loader, val_loader


def train_one_epoch(model, loader, optimizer, criterion, device, scaler):
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device, non_blocking=True), y_batch.to(device, non_blocking=True)
        optimizer.zero_grad()
        with autocast(enabled=scaler is not None):
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * len(y_batch)
        correct += (logits.argmax(1) == y_batch).sum().item()
        n += len(y_batch)
    return total_loss / n, correct / n


@torch.no_grad()
def evaluate_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    all_preds, all_targets = [], []
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        total_loss += loss.item() * len(y_batch)
        preds = logits.argmax(1)
        correct += (preds == y_batch).sum().item()
        n += len(y_batch)
        all_preds.append(preds.cpu().numpy())
        all_targets.append(y_batch.cpu().numpy())
    preds_all = np.concatenate(all_preds)
    targets_all = np.concatenate(all_targets)
    f1 = f1_score(targets_all, preds_all, average='binary', pos_label=1)
    return total_loss / n, correct / n, f1


def train_model(model, X_train, y_train, X_val, y_val, cfg, model_key,
                checkpoint_dir='results/checkpoints'):
    """
    Train LSTM or GRU model. Returns history dict and path to best checkpoint.
    cfg:       full config dict
    model_key: 'lstm' or 'gru'
    """
    mcfg = cfg[model_key]
    batch_size = mcfg['batch_size']
    lr = mcfg['learning_rate']
    epochs = mcfg['epochs']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    scaler = GradScaler() if device.type == 'cuda' else None

    # class weights to handle imbalance
    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    weight = torch.tensor([1.0, n_neg / max(n_pos, 1)], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )

    train_loader, val_loader = make_dataloaders(X_train, y_train, X_val, y_val, batch_size)

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_path = checkpoint_dir / f'best_{model_key}.pt'

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'val_f1': []}
    best_val_f1 = -1.0

    for epoch in tqdm(range(1, epochs + 1), desc=f'Training {model_key.upper()}'):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
        val_loss, val_acc, val_f1 = evaluate_epoch(model, val_loader, criterion, device)
        scheduler.step(val_f1)

        history['train_loss'].append(tr_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(tr_acc)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), best_path)

        if epoch % 10 == 0 or epoch == 1:
            elapsed = time.time() - t0
            print(f'  Epoch {epoch:>3}/{epochs} | '
                  f'train_loss={tr_loss:.4f} acc={tr_acc:.4f} | '
                  f'val_loss={val_loss:.4f} acc={val_acc:.4f} f1={val_f1:.4f} | '
                  f'{elapsed:.1f}s')

    print(f'  Best val F1: {best_val_f1:.4f} — checkpoint: {best_path}')
    return history, str(best_path)
