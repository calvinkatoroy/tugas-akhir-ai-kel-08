from sklearn.ensemble import RandomForestClassifier
import joblib
from pathlib import Path


def build_rf(cfg):
    rf_cfg = cfg['random_forest']
    return RandomForestClassifier(
        n_estimators=rf_cfg['n_estimators'],
        max_depth=rf_cfg.get('max_depth'),
        min_samples_split=rf_cfg['min_samples_split'],
        class_weight=rf_cfg['class_weight'],
        random_state=cfg['seed'],
        n_jobs=rf_cfg['n_jobs'],
    )


def train_rf(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model


def save_rf(model, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_rf(path):
    return joblib.load(path)
