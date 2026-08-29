"""Phase 4 Tier 0 permutation t-tests + Benjamini-Hochberg FDR, using scipy (permutation_test + false_discovery_control)."""

import numpy as np
import pandas as pd
import joblib
from scipy.stats import permutation_test, ttest_ind, false_discovery_control
from _common import RANDOM_SEED, RESULTS_DIR, load_config, load_features

DATASET_KEYS = ("torres_torres", "ibarra_zarate", "raeisi", "wang")


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Return all feature columns (power_*, wpli_*, plzc_*)."""
    return [c for c in df.columns if c.startswith(("power_", "wpli_", "plzc_"))]


def t_statistic(x: np.ndarray, y: np.ndarray, axis) -> float:
    """Welch two-sample t-statistic, vectorized over `axis` — the statistic used by scipy's permutation_test."""
    return ttest_ind(x, y, equal_var=False, axis=axis).statistic


def main() -> None:
    config = load_config()
    alpha = config["phase4"]["alpha"]
    n_perm = config["phase4"]["n_permutations"]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_results = []

    for dataset_key in DATASET_KEYS:
        df = load_features(dataset_key)
        feature_cols = feature_columns(df)
        assert len(feature_cols) == 468, f"Expected 468 features, got {len(feature_cols)}"

        # Subject-level aggregation to avoid pseudo-replication (mean per subject per feature)
        subject_means = df.groupby("Subject_ID", sort=True)[feature_cols].mean()
        subject_groups = df.drop_duplicates("Subject_ID").set_index("Subject_ID")["Group"]
        tinnitus = subject_means.loc[subject_groups == "Tinnitus"]
        control = subject_means.loc[subject_groups == "Control"]

        def run_feature(feature_name):
            x = tinnitus[feature_name].values
            y = control[feature_name].values
            if len(x) < 2 or len(y) < 2:
                return {"dataset": dataset_key, "feature": feature_name,
                        "t_stat": np.nan, "p_perm": np.nan}
            res = permutation_test(
                (x, y),
                t_statistic,
                permutation_type="independent",
                alternative="two-sided",
                n_resamples=n_perm,
                vectorized=True,
                random_state=RANDOM_SEED,
            )
            return {"dataset": dataset_key, "feature": feature_name,
                    "t_stat": res.statistic, "p_perm": res.pvalue}

        results = joblib.Parallel(n_jobs=config["phase4"]["n_jobs"])(
            joblib.delayed(run_feature)(name) for name in feature_cols
        )
        all_results.extend(results)

    result_df = pd.DataFrame(all_results)  # columns: dataset, feature, t_stat, p_perm

    # Benjamini-Hochberg FDR within each dataset via scipy.stats.false_discovery_control
    # Benjamini, Y. & Hochberg, Y. (1995). Controlling the false discovery rate: a practical and powerful approach to multiple testing.
    fdr_frames = []
    for ds in DATASET_KEYS:
        sub = result_df[result_df["dataset"] == ds].copy()
        sub["p_fdr"] = false_discovery_control(sub["p_perm"].to_numpy(), method="bh")
        sub["fdr_significant"] = sub["p_fdr"] < alpha
        fdr_frames.append(sub)

    result_df = pd.concat(fdr_frames, ignore_index=True)
    result_df = result_df[["dataset", "feature", "t_stat", "p_perm", "p_fdr", "fdr_significant"]]
    result_df.to_csv(RESULTS_DIR / "permutation_ttest.csv", index=False)

    # Summary
    print("FDR Correction Summary:")
    for ds in DATASET_KEYS:
        n_sig = int(result_df[result_df["dataset"] == ds]["fdr_significant"].sum())
        print(f"  {ds}: N_significant={n_sig}")
    print("Feature Family Summary:")
    for fam in ("power_", "wpli_", "plzc_"):
        n_sig = int(result_df[result_df["feature"].str.startswith(fam)]["fdr_significant"].sum())
        print(f"  {fam}: N_significant={n_sig}")
    print("Saved", RESULTS_DIR / "permutation_ttest.csv")


if __name__ == "__main__":
    main()