from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "results" / "figures" / "revised_initial"
DESKTOP_FIG_DIR = Path(r"C:\Users\HPGZZ\Desktop\小论文\论文图片")

TABLE2 = PROJECT_ROOT / "results" / "tables" / "table2_main_model_performance.csv"
NO_SSL = PROJECT_ROOT / "results" / "metrics" / "updated_sub05_sub28_no_ssl_exact6cfg_standard10_per_seed.csv"
BARLOW = PROJECT_ROOT / "results" / "metrics" / "barlow_cnn_10seed_per_seed.csv"
NO_SSL_RESIDUAL = (
    PROJECT_ROOT
    / "results"
    / "metrics"
    / "no_ssl_residualaware_highrank_swa_clsalpha1_rerun_20260611_10seed_per_seed.csv"
)
FINAL = (
    PROJECT_ROOT
    / "final_model_ablation_explainability_results_20260610"
    / "independent_train_gpu_main"
    / "results"
    / "tables"
    / "final_Residual_ssl_cnn.csv"
)

STANDARD_SEEDS = [0, 1, 2, 3, 4, 5, 7, 13, 21, 42]

COLORS = {
    "ink": "#263238",
    "muted": "#667085",
    "grid": "#D7DEE8",
    "blue": "#4C78A8",
    "teal": "#72B7B2",
    "green": "#2AA876",
    "gold": "#E0A83B",
    "red": "#E15759",
    "violet": "#7B6CB5",
    "orange": "#F28E2B",
}

MODEL_COLORS = {
    "Residual-aware\nSSL-CNN": COLORS["green"],
    "Barlow CNN": COLORS["violet"],
    "No-SSL CNN": COLORS["orange"],
    "Logistic L1": COLORS["blue"],
    "SVM RBF": COLORS["red"],
    "Logistic L2": COLORS["teal"],
    "No-SSL residual-aware": COLORS["gold"],
    "Residual-aware CNN": COLORS["gold"],
    "Residual-aware SSL-CNN": COLORS["green"],
}

SEED_STABILITY_COLORS = {
    "No-SSL CNN": "#4C78A8",
    "Barlow CNN": "#4E8D7C",
    "No-SSL residual-aware": "#D9A43A",
    "Residual-aware SSL-CNN": "#D66A5C",
}


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.titlesize": 8,
            "axes.labelsize": 7,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.7,
            "axes.edgecolor": COLORS["ink"],
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    renamed = {
        col: col.strip().lower().replace(" ", "_").replace("-", "_")
        for col in frame.columns
    }
    frame = frame.rename(columns=renamed).copy()
    if "brier" in frame.columns and "brier_score" not in frame.columns:
        frame = frame.rename(columns={"brier": "brier_score"})
    return frame


def read_seed_table(path: Path) -> pd.DataFrame:
    frame = normalize_columns(pd.read_csv(path))
    if "seed" not in frame.columns:
        raise ValueError(f"{path} does not contain a seed column.")
    frame["seed"] = frame["seed"].astype(int)
    observed = sorted(frame["seed"].unique().tolist())
    if observed != STANDARD_SEEDS:
        raise ValueError(f"{path.name} uses seeds {observed}; expected {STANDARD_SEEDS}.")
    return frame.sort_values("seed").reset_index(drop=True)


def save_pub(fig: plt.Figure, base: Path, *, dpi: int = 600) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".tiff"), dpi=dpi, bbox_inches="tight")


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.14,
        1.03,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        fontweight="bold",
        color="black",
    )


def build_brier_source() -> pd.DataFrame:
    table2 = pd.read_csv(TABLE2)
    no_ssl = read_seed_table(NO_SSL)
    barlow = read_seed_table(BARLOW)
    no_ssl_residual = read_seed_table(NO_SSL_RESIDUAL)
    final = read_seed_table(FINAL)

    model_col = "model_name" if "model_name" in table2.columns else "source_model"

    def table2_brier(model_name: str) -> float:
        match = table2.loc[table2[model_col].eq(model_name)]
        if match.empty:
            raise ValueError(f"Model not found in {TABLE2}: {model_name}")
        return float(match.iloc[0]["brier_score"])

    rows = [
        {
            "display": "Residual-aware CNN",
            "model_key": "no_ssl_residual_aware_cnn",
            "brier_score": float(no_ssl_residual["brier_score"].mean()),
            "source_file": str(NO_SSL_RESIDUAL),
            "summary": "10-seed mean",
        },
        {
            "display": "Residual-aware\nSSL-CNN",
            "model_key": "final_residual_barlow_cnn",
            "brier_score": float(final["brier_score"].mean()),
            "source_file": str(FINAL),
            "summary": "10-seed mean",
        },
        {
            "display": "Barlow CNN",
            "model_key": "barlow_cnn",
            "brier_score": float(barlow["brier_score"].mean()),
            "source_file": str(BARLOW),
            "summary": "10-seed mean",
        },
        {
            "display": "No-SSL CNN",
            "model_key": "no_ssl_cnn",
            "brier_score": float(no_ssl["brier_score"].mean()),
            "source_file": str(NO_SSL),
            "summary": "10-seed mean",
        },
        {
            "display": "Logistic L1",
            "model_key": "ML_EEG_updated_no_selector_logistic_l1",
            "brier_score": table2_brier("ML_EEG_updated_no_selector_logistic_l1"),
            "source_file": str(TABLE2),
            "summary": "single LOSO",
        },
    ]
    return pd.DataFrame(rows).sort_values("brier_score", ascending=True).reset_index(drop=True)


def make_brier_figure(source: pd.DataFrame) -> None:
    source_path = OUT_DIR / "figure4c_b_brier_calibration_source_data.csv"
    source.to_csv(source_path, index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(3.65, 3.55), constrained_layout=True)
    panel_label(ax, "b")

    y = np.arange(len(source))
    values = source["brier_score"].astype(float).to_numpy()
    labels = source["display"].tolist()
    colors = [MODEL_COLORS.get(label, COLORS["blue"]) for label in labels]

    ax.hlines(y, 0, values, color=COLORS["grid"], linewidth=1.05, zorder=1)
    ax.scatter(values, y, s=46, color=colors, edgecolor="white", linewidth=0.6, zorder=3)
    for yi, value in zip(y, values, strict=True):
        ax.text(value + 0.010, yi, f"{value:.3f}", va="center", ha="left", fontsize=6.4, color=COLORS["ink"])

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(0, max(0.26, float(values.max()) + 0.045))
    ax.set_xticks(np.arange(0, 0.30, 0.05))
    ax.set_xlabel("Brier score")
    ax.set_title("Calibration error\n(lower is better)", loc="left", pad=6)
    ax.grid(axis="x", color=COLORS["grid"], linewidth=0.5, zorder=0)
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.5, zorder=0)
    ax.tick_params(axis="y", length=2.5)

    save_pub(fig, OUT_DIR / "figure4c_b_brier_calibration")
    plt.close(fig)


def build_seed_source() -> pd.DataFrame:
    frames = []
    for label, path, metric_col in [
        ("No-SSL CNN", NO_SSL, "accuracy"),
        ("Barlow CNN", BARLOW, "accuracy"),
        ("No-SSL residual-aware", NO_SSL_RESIDUAL, "accuracy"),
        ("Residual-aware SSL-CNN", FINAL, "accuracy"),
    ]:
        frame = read_seed_table(path)
        if metric_col not in frame.columns:
            raise ValueError(f"{path.name} does not contain {metric_col}.")
        keep = frame[["seed", metric_col]].copy()
        keep = keep.rename(columns={metric_col: "accuracy"})
        keep["model"] = label
        keep["source_file"] = str(path)
        frames.append(keep)
    return pd.concat(frames, ignore_index=True)


def make_seed_stability_figure(source: pd.DataFrame) -> None:
    source_path = OUT_DIR / "figure5a_seed_stability_source_data.csv"
    source.to_csv(source_path, index=False, encoding="utf-8-sig")

    order = ["No-SSL CNN", "Barlow CNN", "No-SSL residual-aware", "Residual-aware SSL-CNN"]
    fig, ax = plt.subplots(figsize=(4.8, 3.4), constrained_layout=True)
    panel_label(ax, "a")

    positions = np.arange(1, len(order) + 1)
    grouped = [
        source.loc[source["model"].eq(label), "accuracy"].astype(float).to_numpy()
        for label in order
    ]

    violins = ax.violinplot(grouped, positions=positions, widths=0.72, showmeans=False, showmedians=False, showextrema=False)
    for body, label in zip(violins["bodies"], order, strict=True):
        color = SEED_STABILITY_COLORS[label]
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.16)
        body.set_linewidth(0.9)

    for pos, label, values in zip(positions, order, grouped, strict=True):
        color = SEED_STABILITY_COLORS[label]
        values = np.asarray(values, dtype=float)
        ax.vlines(pos, values.min(), values.max(), color=color, linewidth=1.25, zorder=2)
        ax.hlines([values.min(), values.max()], pos - 0.18, pos + 0.18, color=color, linewidth=1.25, zorder=2)
        jitter = np.linspace(-0.065, 0.065, len(values))
        ax.scatter(
            np.full_like(values, pos, dtype=float) + jitter,
            values,
            s=20,
            color=color,
            edgecolor="white",
            linewidth=0.45,
            zorder=3,
        )
        ax.scatter(
            [pos],
            [np.median(values)],
            marker="D",
            s=34,
            facecolor="white",
            edgecolor=color,
            linewidth=1.05,
            zorder=4,
        )

    ax.set_title("Random-seed stability", loc="left", pad=6)
    ax.set_ylabel("Seed-level accuracy")
    ax.set_xticks(positions)
    ax.set_xticklabels(["No-SSL\nCNN", "Barlow\nSSL-CNN", "No-SSL\nresidual", "Final\nSSL-CNN"])
    ax.set_ylim(0.50, 0.93)
    ax.set_yticks(np.arange(0.50, 0.95, 0.05))
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.55, zorder=0)
    ax.text(0.02, 0.02, "Diamonds mark medians", transform=ax.transAxes, fontsize=6.2, color=COLORS["muted"])

    save_pub(fig, OUT_DIR / "figure5a_seed_stability")
    plt.close(fig)


def copy_desktop_outputs() -> None:
    DESKTOP_FIG_DIR.mkdir(parents=True, exist_ok=True)
    for stem in ["figure4c_b_brier_calibration", "figure5a_seed_stability"]:
        for suffix in [".png", ".svg"]:
            src = OUT_DIR / f"{stem}{suffix}"
            if src.exists():
                (DESKTOP_FIG_DIR / src.name).write_bytes(src.read_bytes())


def main() -> None:
    configure_matplotlib()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    brier = build_brier_source()
    seed = build_seed_source()
    make_brier_figure(brier)
    make_seed_stability_figure(seed)
    copy_desktop_outputs()
    print(f"brier_png={OUT_DIR / 'figure4c_b_brier_calibration.png'}")
    print(f"seed_png={OUT_DIR / 'figure5a_seed_stability.png'}")
    print(f"final_brier_mean={brier.loc[brier['model_key'].eq('final_residual_barlow_cnn'), 'brier_score'].iloc[0]:.6f}")
    print(f"final_seed_accuracy_mean={seed.loc[seed['model'].eq('Residual-aware SSL-CNN'), 'accuracy'].mean():.6f}")


if __name__ == "__main__":
    main()
