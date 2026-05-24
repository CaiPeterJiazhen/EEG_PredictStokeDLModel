from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from eeg_recovery.config import load_path_config
from eeg_recovery.training.ensemble_calibration import (
    THRESHOLD_METRICS,
    apply_oof_ensemble_weight_threshold_calibration,
    apply_oof_threshold_calibration,
    ensemble_prediction_frames,
    evaluate_prediction_frame,
    load_prediction_frame,
)


DEFAULT_PREDICTIONS = {
    "ntxent_seed2": (
        "results/predictions/"
        "dl_loso_predictions_multitask_ssl_all-patient_psd_masked_wpli_edge_gated_cnn_"
        "finetune_seed2_psd20_fc20_fcnode0_02_fcedge0_05_fcband0_02_cons0_0_"
        "ctr0_01_temp0_2_noise0_02_mask0_05_bs8_sup100.csv"
    ),
    "vicreg_seed2": (
        "results/predictions/"
        "dl_loso_predictions_mtvicreg_all-patient_psdfcwpli_gated_finetune_s2_"
        "p20_f20_fn0_02_fe0_05_fb0_02_w0_001_vi25_0_vv25_0_vc1_0_bs8_sup100.csv"
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Ensemble saved LOSO prediction CSVs and apply leave-one-subject-out "
            "threshold calibration."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default=PROJECT_ROOT / "configs" / "paths.example.yaml")
    parser.add_argument(
        "--prediction",
        action="append",
        default=None,
        help=(
            "Prediction CSV to include. Use name=path for stable output columns. "
            "Repeat for multiple models. If omitted, uses the best NT-Xent and "
            "best VICReg prediction files."
        ),
    )
    parser.add_argument(
        "--weight",
        action="append",
        default=None,
        help="Optional ensemble weight as name=value. Names must match --prediction names.",
    )
    parser.add_argument("--fixed-threshold", type=float, default=0.5)
    parser.add_argument(
        "--calibration-metric",
        choices=sorted(THRESHOLD_METRICS),
        default="balanced_accuracy",
    )
    parser.add_argument(
        "--weight-grid",
        default="0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0",
        help=(
            "Comma-separated first-model weights for strict OOF weight+threshold "
            "calibration. Used only when exactly two prediction files are supplied."
        ),
    )
    parser.add_argument("--output-prefix", default="ntxent_vicreg_seed2_ensemble")
    args = parser.parse_args()

    path_config = load_path_config(args.config)
    output_root = Path(path_config.output_root)
    prediction_specs = _resolve_prediction_specs(args.prediction, output_root)
    weights = _parse_weights(args.weight)
    weight_grid = _parse_weight_grid(args.weight_grid)

    prediction_frames = {
        name: load_prediction_frame(path, model_name=name)
        for name, path in prediction_specs.items()
    }
    ensemble = ensemble_prediction_frames(prediction_frames, weights=weights)

    fixed_predictions = ensemble.copy()
    fixed_predictions["decision_rule"] = "fixed_threshold"
    fixed_predictions["threshold"] = args.fixed_threshold
    fixed_predictions["y_pred"] = (
        fixed_predictions["y_score"].astype(float) >= args.fixed_threshold
    ).astype(int)

    calibrated_predictions = apply_oof_threshold_calibration(
        ensemble,
        metric=args.calibration_metric,
    )
    calibrated_predictions["decision_rule"] = f"oof_{args.calibration_metric}"
    weight_threshold_predictions = None
    if len(prediction_frames) == 2:
        weight_threshold_predictions = apply_oof_ensemble_weight_threshold_calibration(
            prediction_frames,
            weight_grid=weight_grid,
            metric=args.calibration_metric,
        )
        weight_threshold_predictions["decision_rule"] = (
            f"oof_weight_threshold_{args.calibration_metric}"
        )

    metric_rows = []
    for name, frame in prediction_frames.items():
        metric_rows.append(
            evaluate_prediction_frame(
                frame,
                model=name,
                threshold=args.fixed_threshold,
                decision_rule="fixed_threshold",
            )
        )
        calibrated_frame = apply_oof_threshold_calibration(frame, metric=args.calibration_metric)
        metric_rows.append(
            evaluate_prediction_frame(
                calibrated_frame,
                model=name,
                y_pred_column="y_pred",
                decision_rule=f"oof_{args.calibration_metric}",
            )
        )

    metric_rows.append(
        evaluate_prediction_frame(
            fixed_predictions,
            model="ensemble_mean",
            threshold=args.fixed_threshold,
            decision_rule="fixed_threshold",
        )
    )
    metric_rows.append(
        evaluate_prediction_frame(
            calibrated_predictions,
            model="ensemble_mean",
            y_pred_column="y_pred",
            decision_rule=f"oof_{args.calibration_metric}",
        )
    )
    if weight_threshold_predictions is not None:
        metric_rows.append(
            evaluate_prediction_frame(
                weight_threshold_predictions,
                model="ensemble_oof_weight_threshold",
                y_pred_column="y_pred",
                decision_rule=f"oof_weight_threshold_{args.calibration_metric}",
            )
        )
    metrics = pd.DataFrame(metric_rows)
    metrics.insert(0, "run_name", _safe_filename_token(args.output_prefix))
    metrics.insert(1, "calibration_metric", args.calibration_metric)

    output_paths = _write_outputs(
        output_root=output_root,
        output_prefix=args.output_prefix,
        fixed_predictions=fixed_predictions,
        calibrated_predictions=calibrated_predictions,
        weight_threshold_predictions=weight_threshold_predictions,
        metrics=metrics,
        fixed_threshold=args.fixed_threshold,
        calibration_metric=args.calibration_metric,
    )
    print(f"Wrote fixed-threshold ensemble predictions: {output_paths['fixed_predictions']}")
    print(f"Wrote OOF-calibrated ensemble predictions: {output_paths['calibrated_predictions']}")
    if "weight_threshold_predictions" in output_paths:
        print(
            "Wrote OOF weight+threshold ensemble predictions: "
            f"{output_paths['weight_threshold_predictions']}"
        )
    print(f"Wrote ensemble metrics: {output_paths['metrics']}")


def _resolve_prediction_specs(
    prediction_args: list[str] | None,
    output_root: Path,
) -> dict[str, Path]:
    if not prediction_args:
        return {
            name: _resolve_prediction_path(path, output_root)
            for name, path in DEFAULT_PREDICTIONS.items()
        }

    specs: dict[str, Path] = {}
    for raw in prediction_args:
        if "=" in raw:
            name, path_text = raw.split("=", 1)
            name = name.strip()
            path_text = path_text.strip()
        else:
            path_text = raw.strip()
            name = _safe_filename_token(Path(path_text).stem)
        if not name:
            raise SystemExit(f"Invalid empty prediction name in argument: {raw}")
        if name in specs:
            raise SystemExit(f"Duplicate prediction name: {name}")
        specs[name] = _resolve_prediction_path(path_text, output_root)
    return specs


def _resolve_prediction_path(path_text: str | Path, output_root: Path) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        path = output_root / path
    if not path.exists():
        raise SystemExit(f"Prediction file does not exist: {path}")
    return path


def _parse_weights(weight_args: list[str] | None) -> dict[str, float] | None:
    if not weight_args:
        return None
    weights: dict[str, float] = {}
    for raw in weight_args:
        if "=" not in raw:
            raise SystemExit(f"Weight must use name=value syntax: {raw}")
        name, value = raw.split("=", 1)
        name = name.strip()
        if not name:
            raise SystemExit(f"Invalid empty weight name in argument: {raw}")
        if name in weights:
            raise SystemExit(f"Duplicate weight name: {name}")
        try:
            weights[name] = float(value)
        except ValueError as exc:
            raise SystemExit(f"Invalid weight value in argument: {raw}") from exc
    return weights


def _parse_weight_grid(value: str) -> list[float]:
    try:
        grid = [float(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise SystemExit(f"Invalid --weight-grid value: {value}") from exc
    if not grid:
        raise SystemExit("--weight-grid must contain at least one numeric value.")
    if any(weight < 0.0 or weight > 1.0 for weight in grid):
        raise SystemExit("--weight-grid values must be between 0 and 1.")
    return grid


def _write_outputs(
    *,
    output_root: Path,
    output_prefix: str,
    fixed_predictions: pd.DataFrame,
    calibrated_predictions: pd.DataFrame,
    weight_threshold_predictions: pd.DataFrame | None,
    metrics: pd.DataFrame,
    fixed_threshold: float,
    calibration_metric: str,
) -> dict[str, Path]:
    safe = _safe_filename_token(output_prefix)
    threshold_token = _number_token(fixed_threshold)
    prediction_dir = output_root / "results" / "predictions"
    metric_dir = output_root / "results" / "metrics"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    metric_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "fixed_predictions": prediction_dir / f"ensemble_predictions_{safe}_fixed{threshold_token}.csv",
        "calibrated_predictions": prediction_dir
        / f"ensemble_predictions_{safe}_oof_{calibration_metric}.csv",
        "metrics": metric_dir / f"ensemble_metrics_{safe}.csv",
    }
    if weight_threshold_predictions is not None:
        paths["weight_threshold_predictions"] = (
            prediction_dir / f"ensemble_predictions_{safe}_oof_weight_threshold_{calibration_metric}.csv"
        )
    fixed_predictions.to_csv(paths["fixed_predictions"], index=False)
    calibrated_predictions.to_csv(paths["calibrated_predictions"], index=False)
    if weight_threshold_predictions is not None:
        weight_threshold_predictions.to_csv(paths["weight_threshold_predictions"], index=False)
    metrics.to_csv(paths["metrics"], index=False)
    return paths


def _safe_filename_token(value: str) -> str:
    token = re.sub(r"[^0-9A-Za-z]+", "_", value).strip("_").lower()
    return token or "ensemble"


def _number_token(value: float) -> str:
    return str(value).replace(".", "_")


if __name__ == "__main__":
    main()
