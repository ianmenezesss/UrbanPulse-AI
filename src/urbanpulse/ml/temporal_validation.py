"""Expanding-window validation and error analysis for the ML models."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from urbanpulse.ml.baseline import (
    DEFAULT_DATA_PATH,
    TARGET,
    calculate_metrics,
    load_processed_data,
    temporal_train_test_split,
)
from urbanpulse.ml.random_forest import fit_predict_random_forest
from urbanpulse.ml.ridge import fit_predict_ridge

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_REPORT_DIR = PROJECT_ROOT / "reports" / "temporal_validation"
DEFAULT_FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
WEEKDAY_NAMES = ("Segunda", "Terça", "Quarta", "Quinta", "Sexta", "Sábado", "Domingo")
MODEL_PREDICTORS = {"Ridge": fit_predict_ridge, "Random Forest": fit_predict_random_forest}


@dataclass
class TemporalWindow:
    number: int
    train: pd.DataFrame
    validation: pd.DataFrame


@dataclass
class TemporalValidationResult:
    window_metrics: pd.DataFrame
    summary_metrics: pd.DataFrame
    predictions: pd.DataFrame
    errors_by_hour: pd.DataFrame
    errors_by_weekday: pd.DataFrame


def expanding_windows(
    development: pd.DataFrame,
    *,
    n_splits: int = 5,
    initial_train_fraction: float = 0.5,
) -> list[TemporalWindow]:
    """Create contiguous validation blocks with an expanding past-only train set.

    The input must contain only the initial 80% of the full series. The final
    20% is isolated by evaluate_temporal_validation and never passed here.
    """
    if n_splits < 1:
        raise ValueError("n_splits must be at least 1")
    if not 0 < initial_train_fraction < 1:
        raise ValueError("initial_train_fraction must be strictly between 0 and 1")
    if "datetime" not in development.columns:
        raise ValueError("Development data is missing required column: datetime")
    if development["datetime"].isna().any():
        raise ValueError("datetime contains missing values")

    ordered = development.sort_values("datetime", kind="stable").reset_index(drop=True)
    initial_train_size = int(len(ordered) * initial_train_fraction)
    remaining = len(ordered) - initial_train_size
    if initial_train_size < 2 or remaining < n_splits:
        raise ValueError("Not enough rows for the requested expanding validation windows")

    block_sizes = np.full(n_splits, remaining // n_splits, dtype=int)
    block_sizes[: remaining % n_splits] += 1
    windows: list[TemporalWindow] = []
    validation_start = initial_train_size
    for number, block_size in enumerate(block_sizes, start=1):
        validation_end = validation_start + int(block_size)
        windows.append(
            TemporalWindow(
                number=number,
                train=ordered.iloc[:validation_start].copy(),
                validation=ordered.iloc[validation_start:validation_end].copy(),
            )
        )
        validation_start = validation_end
    return windows


def summarize_window_metrics(window_metrics: pd.DataFrame) -> pd.DataFrame:
    """Return mean and sample standard deviation of metrics across windows."""
    metrics = ("mae", "rmse", "r2")
    missing = sorted({"model", *metrics}.difference(window_metrics.columns))
    if missing:
        raise ValueError(f"Window metrics are missing required columns: {', '.join(missing)}")
    summary = window_metrics.groupby("model")[list(metrics)].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    return summary.reset_index()


def _grouped_error_metrics(predictions: pd.DataFrame, group_column: str) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for (model, category), group in predictions.groupby(
        ["model", group_column], sort=True, observed=True
    ):
        errors = group["error"].to_numpy(dtype=float)
        rows.append(
            {
                "model": str(model),
                group_column: category,
                "observations": len(group),
                "mae": float(np.mean(np.abs(errors))),
                "rmse": float(np.sqrt(np.mean(np.square(errors)))),
                "mean_error": float(np.mean(errors)),
            }
        )
    return pd.DataFrame(rows)


def evaluate_temporal_validation(
    frame: pd.DataFrame,
    *,
    n_splits: int = 5,
    initial_train_fraction: float = 0.5,
) -> TemporalValidationResult:
    """Evaluate Ridge and Random Forest using only the initial 80% of records."""
    development, _reserved_final_test = temporal_train_test_split(frame)
    windows = expanding_windows(
        development,
        n_splits=n_splits,
        initial_train_fraction=initial_train_fraction,
    )

    metric_rows: list[dict[str, object]] = []
    prediction_rows: list[pd.DataFrame] = []
    weekday_names = dict(enumerate(WEEKDAY_NAMES))
    for window in windows:
        actual = window.validation[TARGET]
        for model_name, predict in MODEL_PREDICTORS.items():
            predicted = predict(window.train, window.validation)
            metrics = calculate_metrics(actual, pd.Series(predicted, index=actual.index))
            metric_rows.append(
                {
                    "window": window.number,
                    "model": model_name,
                    "train_rows": len(window.train),
                    "validation_rows": len(window.validation),
                    "train_start": window.train["datetime"].iloc[0],
                    "train_end": window.train["datetime"].iloc[-1],
                    "validation_start": window.validation["datetime"].iloc[0],
                    "validation_end": window.validation["datetime"].iloc[-1],
                    "mae": metrics.mae,
                    "rmse": metrics.rmse,
                    "r2": metrics.r2,
                }
            )
            dates = window.validation["datetime"]
            prediction_rows.append(
                pd.DataFrame(
                    {
                        "window": window.number,
                        "datetime": dates.to_numpy(),
                        "hour": dates.dt.hour.to_numpy(),
                        "weekday": dates.dt.dayofweek.to_numpy(),
                        "weekday_name": dates.dt.dayofweek.map(weekday_names).to_numpy(),
                        "model": model_name,
                        "actual": actual.to_numpy(dtype=float),
                        "predicted": np.asarray(predicted, dtype=float),
                        "error": np.asarray(predicted, dtype=float) - actual.to_numpy(dtype=float),
                    }
                )
            )

    window_metrics = pd.DataFrame(metric_rows)
    predictions = pd.concat(prediction_rows, ignore_index=True)
    summary_metrics = summarize_window_metrics(window_metrics)
    errors_by_hour = _grouped_error_metrics(predictions, "hour")
    errors_by_weekday = _grouped_error_metrics(predictions, "weekday")
    errors_by_weekday["weekday_name"] = errors_by_weekday["weekday"].map(weekday_names)
    return TemporalValidationResult(
        window_metrics=window_metrics,
        summary_metrics=summary_metrics,
        predictions=predictions,
        errors_by_hour=errors_by_hour,
        errors_by_weekday=errors_by_weekday,
    )


def save_figures(
    result: TemporalValidationResult,
    output_dir: Path = DEFAULT_FIGURES_DIR,
) -> list[Path]:
    """Save actual-vs-predicted, residual, hourly-error, and weekday-error charts."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    models = list(MODEL_PREDICTORS)

    figure, axes = plt.subplots(1, len(models), figsize=(13, 5), constrained_layout=True)
    lower = min(result.predictions["actual"].min(), result.predictions["predicted"].min())
    upper = max(result.predictions["actual"].max(), result.predictions["predicted"].max())
    for axis, model in zip(axes, models):
        subset = result.predictions[result.predictions["model"] == model]
        axis.scatter(subset["actual"], subset["predicted"], s=9, alpha=0.25)
        axis.plot([lower, upper], [lower, upper], color="black", linestyle="--", linewidth=1)
        axis.set(title=model, xlabel="Aluguéis reais por hora", ylabel="Aluguéis previstos por hora")
        axis.grid(alpha=0.2)
    figure.suptitle("Validação temporal: valores reais e previstos")
    path = output_dir / "temporal_validation_actual_vs_predicted.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    paths.append(path)

    figure, axis = plt.subplots(figsize=(10, 5), constrained_layout=True)
    for model in models:
        errors = result.predictions.loc[result.predictions["model"] == model, "error"]
        axis.hist(errors, bins=45, alpha=0.5, density=True, label=model)
    axis.axvline(0, color="black", linestyle="--", linewidth=1)
    axis.set(
        title="Distribuição dos erros fora da amostra",
        xlabel="Erro (previsto − real), aluguéis/hora",
        ylabel="Densidade",
    )
    axis.legend()
    axis.grid(axis="y", alpha=0.2)
    path = output_dir / "temporal_validation_error_distribution.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    paths.append(path)

    figure, axis = plt.subplots(figsize=(11, 5), constrained_layout=True)
    for model in models:
        subset = result.errors_by_hour[result.errors_by_hour["model"] == model]
        axis.plot(subset["hour"], subset["mae"], marker="o", label=model)
    axis.set(
        title="Erro absoluto médio por hora do dia",
        xlabel="Hora do dia",
        ylabel="MAE (aluguéis/hora)",
        xticks=range(24),
    )
    axis.legend()
    axis.grid(alpha=0.25)
    path = output_dir / "temporal_validation_error_by_hour.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    paths.append(path)

    figure, axis = plt.subplots(figsize=(11, 5), constrained_layout=True)
    for model in models:
        subset = result.errors_by_weekday[result.errors_by_weekday["model"] == model]
        axis.plot(subset["weekday"], subset["mae"], marker="o", label=model)
    axis.set(
        title="Erro absoluto médio por dia da semana",
        xlabel="Dia da semana",
        ylabel="MAE (aluguéis/hora)",
        xticks=range(7),
        xticklabels=WEEKDAY_NAMES,
    )
    axis.legend()
    axis.grid(alpha=0.25)
    path = output_dir / "temporal_validation_error_by_weekday.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    paths.append(path)
    return paths


def save_reports(
    result: TemporalValidationResult,
    report_dir: Path = DEFAULT_REPORT_DIR,
    figures_dir: Path = DEFAULT_FIGURES_DIR,
) -> list[Path]:
    """Persist metric tables, out-of-sample predictions, error tables, and plots."""
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        "window_metrics.csv": result.window_metrics,
        "summary_metrics.csv": result.summary_metrics,
        "out_of_sample_predictions.csv": result.predictions,
        "error_by_hour.csv": result.errors_by_hour,
        "error_by_weekday.csv": result.errors_by_weekday,
    }
    paths: list[Path] = []
    for filename, table in tables.items():
        path = report_dir / filename
        table.to_csv(path, index=False)
        paths.append(path)
    paths.extend(save_figures(result, output_dir=figures_dir))
    return paths


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run expanding-window temporal validation for Ridge and Random Forest."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_DATA_PATH, help="Prepared Parquet path.")
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR, help="CSV report directory.")
    parser.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES_DIR, help="Figure output directory.")
    args = parser.parse_args(argv)
    try:
        result = evaluate_temporal_validation(load_processed_data(args.input))
        paths = save_reports(result, args.report_dir, args.figures_dir)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"ERROR: {exc}\n")

    print("Metrics by validation window:")
    print(
        result.window_metrics[
            ["window", "model", "train_rows", "validation_rows", "mae", "rmse", "r2"]
        ].to_string(index=False)
    )
    print("\nMean and sample standard deviation across windows:")
    print(result.summary_metrics.to_string(index=False))
    print("\nSaved reports and figures:")
    for path in paths:
        print(f"  {path}")
    print(
        "Weather-feature assumption: weather forecasts for each validation hour are "
        "available at prediction time."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
