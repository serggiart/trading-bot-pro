"""Automated walk-forward self-calibration package."""

from src.calibration.self_calibrator import (
    CalibrationReport,
    StrategySelfCalibrator,
    TickerCalibrationScore,
    compute_objective_score,
)

__all__ = [
    "CalibrationReport",
    "StrategySelfCalibrator",
    "TickerCalibrationScore",
    "compute_objective_score",
]
