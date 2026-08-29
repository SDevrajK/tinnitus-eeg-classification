"""
Dataset-of-origin confound check using 4-class multinomial logistic regression.
FR 15-17: Check for dataset-of-origin confound in feature space.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, ConfusionMatrixDisplay, confusion_matrix
from _common import load_config, load_scaled_features, RESULTS_DIR, FIGURES_DIR, RANDOM_SEED

DATASET_KEYS = ("torres_torres", "ibarra_zarate", "raeisi", "wang")
FEATURE_PREFIXES = ("power_", "wpli_", "plzc_")

def feature_columns(df):
    """Return list of feature columns (those starting with prefixes)."""
    return [col for col in df.columns if col.startswith(FEATURE_PREFIXES)]

def main():
    config = load_config()
    cv_folds = int(config["tier1"]["cv_folds"])

    # Pool all 4 datasets
    parts = []
    for dataset_key in DATASET_KEYS:
        df = load_scaled_features(dataset_key)
        df["dataset_label"] = dataset_key
        parts.append(df)
    pool = pd.concat(parts, ignore_index=True)

    X = pool[feature_columns(pool)].to_numpy()
    y = pool["dataset_label"].to_numpy()  # string labels
    groups = pool["Subject_ID"].to_numpy()

    # Subject-grouped CV
    gkf = GroupKFold(n_splits=cv_folds)
    y_true_all = []
    y_pred_all = []
    for (tr, va) in gkf.split(X, y, groups=groups):
        clf = LogisticRegression(solver="lbfgs", C=1.0, max_iter=1000, random_state=RANDOM_SEED)
        clf.fit(X[tr], y[tr])
        y_pred_all.append(clf.predict(X[va]))
        y_true_all.append(y[va])
    y_true = np.concatenate(y_true_all)
    y_pred = np.concatenate(y_pred_all)

    ba = balanced_accuracy_score(y_true, y_pred)

    # Confusion matrix figure
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, labels=DATASET_KEYS, display_labels=DATASET_KEYS, cmap="Blues", colorbar=True, ax=ax)
    fig.suptitle(f"Dataset-of-origin confound (balanced accuracy = {ba:.3f})")
    fig.tight_layout()
    out_dir = FIGURES_DIR / "tier1" / "logistic-classifier"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "dataset_of_origin_confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Save balanced accuracy
    pd.DataFrame({"balanced_accuracy": [ba]}).to_csv(RESULTS_DIR / "dataset_of_origin_balanced_accuracy.csv", index=False)

    print(f"Balanced accuracy = {ba:.4f}")
    print("Confusion matrix (rows=true, cols=predicted):")
    print(confusion_matrix(y_true, y_pred, labels=DATASET_KEYS))

if __name__ == "__main__":
    main()