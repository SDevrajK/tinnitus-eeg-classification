"""Train elastic-net regularized logistic regression classifiers (SGD) with nested CV."""

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from _common import load_config, load_scaled_features, RESULTS_DIR, RANDOM_SEED

FEATURE_PREFIXES = ("power_", "wpli_", "plzc_")

def feature_columns(df):
    return [col for col in df.columns if col.startswith(FEATURE_PREFIXES)]

def main():
    config = load_config()
    cv_folds = int(config["tier1"]["cv_folds"])
    l1_grid = list(config["tier1"]["elastic_net"]["l1_ratio_grid"])
    alpha_grid = list(config["tier1"]["elastic_net"]["alpha_grid"])
    max_iter = int(config["tier1"]["elastic_net"]["max_iter"])
    tol = float(config["tier1"]["elastic_net"]["tol"])
    scoring = config["tier1"]["elastic_net"]["scoring"]

    MODELS_DIR = RESULTS_DIR / "models"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    rows = []

    for dataset_key in ("torres_torres", "ibarra_zarate", "raeisi", "wang"):
        df = load_scaled_features(dataset_key)
        feat_cols = feature_columns(df)
        X = df[feat_cols].to_numpy()
        y = (df["Group"] == "Tinnitus").astype(int).to_numpy()
        groups = df["Subject_ID"].to_numpy()

        base = SGDClassifier(loss="log_loss", penalty="elasticnet", max_iter=max_iter, tol=tol, random_state=RANDOM_SEED)

        outer = GroupKFold(n_splits=cv_folds)
        inner = GroupKFold(n_splits=cv_folds)
        param_grid = {"alpha": alpha_grid, "l1_ratio": l1_grid}

        fold_ba = []
        fold_auc = []
        for train_idx, val_idx in outer.split(X, y, groups=groups):
            gs = GridSearchCV(base, param_grid, cv=inner, scoring=scoring, n_jobs=-1)
            gs.fit(X[train_idx], y[train_idx], groups=groups[train_idx])
            y_pred = gs.predict(X[val_idx])
            y_prob = gs.predict_proba(X[val_idx])[:, 1]
            fold_ba.append(balanced_accuracy_score(y[val_idx], y_pred))
            fold_auc.append(roc_auc_score(y[val_idx], y_prob))

        nested_ba = float(np.mean(fold_ba))
        nested_auc = float(np.mean(fold_auc))

        final_search = GridSearchCV(base, param_grid, cv=GroupKFold(n_splits=cv_folds), scoring=scoring, n_jobs=-1)
        final_search.fit(X, y, groups=groups)

        best_alpha = float(final_search.best_params_["alpha"])
        best_l1 = float(final_search.best_params_["l1_ratio"])
        model = final_search.best_estimator_

        joblib.dump(model, MODELS_DIR / f"{dataset_key}_elastic_net.joblib")

        rows.append({
            "dataset": dataset_key,
            "best_alpha": best_alpha,
            "best_l1_ratio": best_l1,
            "nested_balanced_accuracy": nested_ba,
            "nested_roc_auc": nested_auc,
        })
        print(f"{dataset_key}: best_alpha={best_alpha}, best_l1_ratio={best_l1}, nested_balanced_accuracy={nested_ba:.4f}, nested_roc_auc={nested_auc:.4f}")

    pd.DataFrame(rows).to_csv(RESULTS_DIR / "elastic_net_best_params.csv", index=False)

if __name__ == "__main__":
    main()