import random
import numpy as np
import pandas as pd
import yaml
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Label normalization — maps all raw label variants to canonical binary labels
_LABEL_MAP = {
    # benign variants
    'benign': 'normal',
    'BENIGN': 'normal',
    'Benign': 'normal',
    # attack variants (all map to ddos)
    'DrDoS_DNS': 'ddos', 'drdos_dns': 'ddos',
    'DrDoS_LDAP': 'ddos', 'drdos_ldap': 'ddos',
    'DrDoS_MSSQL': 'ddos', 'drdos_mssql': 'ddos',
    'DrDoS_NetBIOS': 'ddos', 'drdos_netbios': 'ddos',
    'DrDoS_NTP': 'ddos', 'drdos_ntp': 'ddos',
    'DrDoS_SNMP': 'ddos', 'drdos_snmp': 'ddos',
    'DrDoS_SSDP': 'ddos', 'drdos_ssdp': 'ddos',
    'DrDoS_UDP': 'ddos', 'drdos_udp': 'ddos',
    'Syn': 'ddos', 'syn': 'ddos',
    'TFTP': 'ddos', 'tftp': 'ddos',
    'UDP': 'ddos', 'udp': 'ddos',
    'UDP-lag': 'ddos', 'UDPLag': 'ddos', 'udplag': 'ddos',
    'LDAP': 'ddos', 'ldap': 'ddos',
    'MSSQL': 'ddos', 'mssql': 'ddos',
    'NetBIOS': 'ddos', 'netbios': 'ddos',
    'Portmap': 'ddos', 'portmap': 'ddos',
    'WebDDoS': 'ddos', 'webddos': 'ddos',
}

LABEL_ENCODING = {'normal': 0, 'ddos': 1}


def load_config(config_path='config.yaml'):
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_kaggle_parquets(kaggle_dir):
    """Load and concatenate all Kaggle parquet files into one DataFrame."""
    kaggle_dir = Path(kaggle_dir)
    frames = []
    for p in sorted(kaggle_dir.glob('*.parquet')):
        df = pd.read_parquet(p)
        frames.append(df)
        print(f'  Loaded {p.name}: {len(df):,} rows')
    if not frames:
        raise FileNotFoundError(
            f'No .parquet files found in {kaggle_dir!r}. '
            f'Directory exists: {kaggle_dir.exists()}. '
            f'Contents: {list(kaggle_dir.iterdir()) if kaggle_dir.exists() else "N/A"}'
        )
    combined = pd.concat(frames, ignore_index=True)
    print(f'  Combined: {len(combined):,} rows')
    return combined


def normalize_labels(df, label_col='Label'):
    """Map raw label strings to binary 'normal'/'ddos', drop unknowns."""
    df = df.copy()
    df[label_col] = df[label_col].astype(str).map(_LABEL_MAP)
    unknown = df[label_col].isna().sum()
    if unknown > 0:
        print(f'  Warning: {unknown} rows with unknown labels dropped')
        df = df.dropna(subset=[label_col])
    return df


def select_features(df, features, label_col='Label'):
    """Keep only target features + label, drop rows with nan/inf."""
    cols = features + [label_col]
    df = df[cols].copy()
    # cast to float64 for safety before inf check
    for c in features:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    # drop inf and nan
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df


def split_data(df, features, label_col='Label', train_r=0.70, val_r=0.15, seed=42):
    """Stratified 70/15/15 split. Returns (X_train, X_val, X_test, y_train, y_val, y_test)."""
    X = df[features].values.astype(np.float32)
    y = df[label_col].map(LABEL_ENCODING).values.astype(np.int64)

    test_r = 1.0 - train_r - val_r
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=(1.0 - train_r), stratify=y, random_state=seed
    )
    val_fraction = val_r / (val_r + test_r)
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=(1.0 - val_fraction), stratify=y_tmp, random_state=seed
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def scale_features(X_train, X_val, X_test):
    """Fit StandardScaler on train only, transform all splits."""
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    return X_train, X_val, X_test, scaler


def make_sequences(X, y, seq_len):
    """Sliding window → (n_samples, seq_len, n_features). Drops remainder."""
    n = len(X)
    n_seq = n - seq_len + 1
    X_seq = np.stack([X[i:i + seq_len] for i in range(n_seq)], axis=0)
    y_seq = y[seq_len - 1:]  # label is the last step in each window
    return X_seq.astype(np.float32), y_seq.astype(np.int64)


def run_preprocessing(config_path='config.yaml', kaggle_dir=None, splits_dir=None):
    """
    kaggle_dir: override for parquet source dir (e.g. Google Drive path on Colab)
    splits_dir: override for output splits dir (e.g. Google Drive path on Colab)
    """
    cfg = load_config(config_path)
    features = cfg['features']
    seed = cfg['seed']
    if kaggle_dir is None:
        kaggle_dir = cfg['data']['kaggle_path']
    if splits_dir is None:
        splits_dir = Path(cfg['data']['splits_path'])
    splits_dir = Path(splits_dir)
    splits_dir.mkdir(parents=True, exist_ok=True)

    print('Loading parquets...')
    df = load_kaggle_parquets(kaggle_dir)

    print('\nNormalizing labels...')
    df = normalize_labels(df)
    print(f"  Label distribution:\n{df['Label'].value_counts()}")

    print('\nSelecting features...')
    df = select_features(df, features)
    print(f'  After feature selection + inf/nan drop: {len(df):,} rows, {len(features)} features')

    print('\nSplitting 70/15/15 stratified...')
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        df, features, seed=seed,
        train_r=cfg['data']['train_ratio'],
        val_r=cfg['data']['val_ratio'],
    )
    print(f'  Train: {len(X_train):,}  Val: {len(X_val):,}  Test: {len(X_test):,}')

    print('\nScaling features (StandardScaler fit on train)...')
    X_train, X_val, X_test, scaler = scale_features(X_train, X_val, X_test)
    joblib.dump(scaler, splits_dir / 'scaler.pkl')
    print('  Scaler saved.')

    print('\nSaving flat splits (for Random Forest)...')
    np.save(splits_dir / 'X_train.npy', X_train)
    np.save(splits_dir / 'X_val.npy', X_val)
    np.save(splits_dir / 'X_test.npy', X_test)
    np.save(splits_dir / 'y_train.npy', y_train)
    np.save(splits_dir / 'y_val.npy', y_val)
    np.save(splits_dir / 'y_test.npy', y_test)

    seq_len = cfg['lstm']['seq_len']
    print(f'\nBuilding sequences (seq_len={seq_len}) for LSTM/GRU...')
    X_train_seq, y_train_seq = make_sequences(X_train, y_train, seq_len)
    X_val_seq, y_val_seq = make_sequences(X_val, y_val, seq_len)
    X_test_seq, y_test_seq = make_sequences(X_test, y_test, seq_len)
    print(f'  Train seq: {X_train_seq.shape}  Val: {X_val_seq.shape}  Test: {X_test_seq.shape}')

    np.save(splits_dir / 'X_train_seq.npy', X_train_seq)
    np.save(splits_dir / 'X_val_seq.npy', X_val_seq)
    np.save(splits_dir / 'X_test_seq.npy', X_test_seq)
    np.save(splits_dir / 'y_train_seq.npy', y_train_seq)
    np.save(splits_dir / 'y_val_seq.npy', y_val_seq)
    np.save(splits_dir / 'y_test_seq.npy', y_test_seq)

    print('\nDone. All splits saved to', splits_dir)
    return {
        'n_features': len(features),
        'seq_len': seq_len,
        'train_size': len(X_train),
        'val_size': len(X_val),
        'test_size': len(X_test),
    }


if __name__ == '__main__':
    run_preprocessing()
